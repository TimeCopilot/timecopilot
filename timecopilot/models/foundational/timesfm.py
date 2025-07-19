from contextlib import contextmanager

import pandas as pd
import timesfm
import torch
import utilsforecast.processing as ufp
from timesfm.timesfm_base import DEFAULT_QUANTILES as DEFAULT_QUANTILES_TFM

from ..utils.forecaster import Forecaster, QuantileConverter


class TimesFM(Forecaster):
    """
    TimesFM is a large time series model for time series forecasting, supporting both
    probabilistic and point forecasts. See the [official repo](https://github.com/
    google-research/timesfm) for more details.
    """

    def __init__(
        self,
        repo_id: str = "google/timesfm-1.0-200m-pytorch",
        context_length: int = 512,
        per_core_batch_size: int = 64,
        num_layers: int = 20,
        model_dims: int = 1280,
        alias: str = "TimesFM",
        max_length: int | None = None,
    ):
        """
        Args:
            repo_id (str, optional): The Hugging Face Hub model ID or local path to
                load the TimesFM model from. Examples include
                "google/timesfm-1.0-200m-pytorch". Defaults to
                "google/timesfm-1.0-200m-pytorch". See the full list of models at
                [Hugging Face](https://huggingface.co/collections/google/timesfm-release-
                66e4be5fdb56e960c1e482a6).
            context_length (int, optional): Maximum context length (input window size)
                for the model. Defaults to 512. For TimesFM 2.0 models, max is 2048
                (must be a multiple of 32). See [TimesFM docs](https://github.com/google-
                research/timesfm#loading-the-model) for details.
            per_core_batch_size (int, optional): Batch size per device/core for
                inference. Defaults to 64. Adjust based on available memory and model
                size.
            num_layers (int, optional): Number of transformer layers in the model.
                Defaults to 20. Should match the configuration of the pretrained
                checkpoint.
            model_dims (int, optional): Model hidden dimension size. Defaults to 1280.
                Should match the configuration of the pretrained checkpoint.
            alias (str, optional): Name to use for the model in output DataFrames and
                logs. Defaults to "TimesFM".
            max_length (int, optional): Maximum number of observations to use from the
                end of each time series for training and inference. If None, all
                observations are used. This can significantly improve inference times
                for long time series by reducing the amount of data processed.

        Notes:
            **Academic Reference:**

            - Paper: [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688)

            **Resources:**

            - GitHub: [google-research/timesfm](https://github.com/google-research/timesfm)
            - HuggingFace: [google/timesfm-release](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6)

            **Technical Details:**

            - Only PyTorch checkpoints are currently supported. JAX is not supported.
            - TimesFM 2.0 models are not yet supported. See
              [issue #269](https://github.com/google-research/timesfm/issues/269)
              for more details.
            - The model is loaded onto the best available device (GPU if available,
              otherwise CPU).
        """
        super().__init__(max_length=max_length)
        if "pytorch" not in repo_id:
            raise ValueError(
                "TimesFM only supports pytorch models, "
                "if you'd like to use jax, please open an issue"
            )

        if "2.0" in repo_id:
            raise ValueError(
                "TimesFM 2.0 is not supported yet, "
                "see https://github.com/google-research/timesfm/issues/269"
                "please use TimesFM 1.0"
            )

        self.repo_id = repo_id
        self.context_length = context_length
        self.per_core_batch_size = per_core_batch_size
        self.num_layers = num_layers
        self.model_dims = model_dims
        self.alias = alias

    @contextmanager
    def get_predictor(
        self,
        prediction_length: int,
        quantiles: list[float] | None = None,
    ) -> timesfm.TimesFm:
        backend = "gpu" if torch.cuda.is_available() else "cpu"
        tfm_hparams = timesfm.TimesFmHparams(
            backend=backend,
            horizon_len=prediction_length,
            quantiles=quantiles,
            context_len=self.context_length,
            num_layers=self.num_layers,
            model_dims=self.model_dims,
            per_core_batch_size=self.per_core_batch_size,
        )
        tfm_checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id=self.repo_id)
        tfm = timesfm.TimesFm(
            hparams=tfm_hparams,
            checkpoint=tfm_checkpoint,
        )
        try:
            yield tfm
        finally:
            del tfm
            torch.cuda.empty_cache()

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts for time series data using the model.

        This method produces point forecasts and, optionally, prediction
        intervals or quantile forecasts. The input DataFrame can contain one
        or multiple time series in stacked (long) format.

        Args:
            df (pd.DataFrame):
                DataFrame containing the time series to forecast. It must
                include as columns:

                    - "unique_id": an ID column to distinguish multiple series.
                    - "ds": a time column indicating timestamps or periods.
                    - "y": a target column with the observed values.

            h (int):
                Forecast horizon specifying how many future steps to predict.
            freq (str, optional):
                Frequency of the time series (e.g. "D" for daily, "M" for
                monthly). See [Pandas frequency aliases](https://pandas.pydata.org/
                pandas-docs/stable/user_guide/timeseries.html#offset-aliases) for
                valid values. If not provided, the frequency will be inferred
                from the data.
            level (list[int | float], optional):
                Confidence levels for prediction intervals, expressed as
                percentages (e.g. [80, 95]). If provided, the returned
                DataFrame will include lower and upper interval columns for
                each specified level.
            quantiles (list[float], optional):
                List of quantiles to forecast, expressed as floats between 0
                and 1. Should not be used simultaneously with `level`. When
                provided, the output DataFrame will contain additional columns
                named in the format "model-q-{percentile}", where {percentile}
                = 100 × quantile value.

        Returns:
            pd.DataFrame:
                DataFrame containing forecast results. Includes:

                    - point forecasts for each timestamp and series.
                    - prediction intervals if `level` is specified.
                    - quantile forecasts if `quantiles` is specified.

                For multi-series data, the output retains the same unique
                identifiers as the input DataFrame.
        """
        freq = self._maybe_infer_freq(df, freq)
        df = self._maybe_truncate_series(df)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        if qc.quantiles is not None and len(qc.quantiles) != len(DEFAULT_QUANTILES_TFM):
            raise ValueError(
                "TimesFM only supports the default quantiles, "
                "please use the default quantiles or default level, "
                "see https://github.com/google-research/timesfm/issues/286"
            )
        with self.get_predictor(
            prediction_length=h,
            quantiles=qc.quantiles or DEFAULT_QUANTILES_TFM,
        ) as predictor:
            fcst_df = predictor.forecast_on_df(
                inputs=df,
                freq=freq,
                value_name="y",
                model_name=self.alias,
                num_jobs=1,
            )
        if qc.quantiles is not None:
            for q in qc.quantiles:
                fcst_df = ufp.assign_columns(
                    fcst_df,
                    f"{self.alias}-q-{int(q * 100)}",
                    fcst_df[f"{self.alias}-q-{q}"],
                )
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        else:
            fcst_df = fcst_df[["unique_id", "ds", self.alias]]
        return fcst_df
