from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto as TotoModel
from tqdm import tqdm

from ..utils.forecaster import Forecaster, QuantileConverter
from ...utils.df_utils import to_pandas
from .utils import TimeSeriesDataset


class Toto(Forecaster):
    """
    Toto is a foundation model for multivariate time series forecasting, optimized
    for observability and high-dimensional data. See the
    [official repo](https://github.com/DataDog/toto) for more details.
    """

    def __init__(
        self,
        repo_id: str = "Datadog/Toto-Open-Base-1.0",
        context_length: int = 4096,
        batch_size: int = 16,
        num_samples: int = 128,
        samples_per_batch: int = 8,
        alias: str = "Toto",
    ):
        """
        Args:
            repo_id (str, optional): The Hugging Face Hub model ID or local path to
                load the Toto model from. Examples include "Datadog/Toto-Open-Base-1.0".
                Defaults to "Datadog/Toto-Open-Base-1.0". See the full list of models
                at [Hugging Face](https://huggingface.co/Datadog).
            context_length (int, optional): Maximum context length (input window size)
                for the model. Defaults to 4096. Should match the configuration of the
                pretrained checkpoint. See [Toto docs](https://github.com/DataDog/toto#
                toto-model) for details.
            batch_size (int, optional): Batch size to use for inference. Defaults to 16.
                Adjust based on available memory and model size.
            num_samples (int, optional): Number of samples for probabilistic
                forecasting. Controls the number of forecast samples drawn for
                uncertainty estimation. Defaults to 128.
            samples_per_batch (int, optional): Number of samples processed per batch
                during inference. Controls memory usage. Defaults to 8.
            alias (str, optional): Name to use for the model in output DataFrames and
                logs. Defaults to "Toto".

        Notes:
            **Academic Reference:**

            - Paper: [Building a Foundation Model for Time Series](https://arxiv.org/abs/2505.14766)

            **Resources:**

            - GitHub: [DataDog/toto](https://github.com/DataDog/toto)
            - HuggingFace: [Datadog Models](https://huggingface.co/Datadog)

            **Technical Details:**

            - The model is loaded onto the best available device (GPU if available,
              otherwise CPU).
            - For best performance, a CUDA-capable GPU is recommended.
        """
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        # Number of samples for probabilistic forecasting
        self.num_samples = num_samples
        # Control memory usage during inference
        self.samples_per_batch = samples_per_batch
        self.alias = alias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @contextmanager
    def _get_model(self) -> TotoForecaster:
        model = TotoModel.from_pretrained(self.repo_id).to(self.device)
        try:
            yield TotoForecaster(model.model)
        finally:
            del model
            torch.cuda.empty_cache()

    def _to_masked_timeseries(self, batch: list[torch.Tensor]) -> MaskedTimeseries:
        batch_size = len(batch)
        # using toch.float as stated in the docs
        # https://github.com/DataDog/toto/blob/main/toto/notebooks/inference_tutorial.ipynb
        padded_tensor = torch.zeros(
            batch_size,
            self.context_length,
            dtype=torch.float,
            device=self.device,
        )
        padding_mask = torch.zeros(
            batch_size,
            self.context_length,
            dtype=torch.float,
            device=self.device,
        )
        for idx, ts in enumerate(batch):
            series_length = len(ts)
            if series_length > self.context_length:
                ts = ts[-self.context_length :]
            padded_tensor[idx, -series_length:] = ts
            padding_mask[idx, -series_length:] = 1.0
        masked_ts = MaskedTimeseries(
            series=padded_tensor,
            padding_mask=padding_mask,
            id_mask=torch.zeros_like(padded_tensor),
            # Prepare timestamp information (optional, but expected by API;
            # not used by the current model release)
            timestamp_seconds=torch.zeros_like(padded_tensor),
            time_interval_seconds=torch.full(
                (batch_size,),
                1,
                device=self.device,
            ),
        )
        return masked_ts

    def _forecast(
        self,
        model: TotoForecaster,
        dataset: TimeSeriesDataset,
        h: int,
        quantiles: list[float] | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """handles distinction between quantiles and no quantiles"""
        fcsts = [
            model.forecast(
                self._to_masked_timeseries(batch),
                prediction_length=h,
                num_samples=self.num_samples,
                samples_per_batch=self.samples_per_batch,
                use_kv_cache=True,
            )
            for batch in tqdm(dataset)
        ]  # list of fcsts objects

        fcsts_mean = [fcst.mean.cpu().numpy() for fcst in fcsts]
        fcsts_mean_np = np.concatenate(fcsts_mean, axis=1)
        if fcsts_mean_np.shape[0] != 1:
            raise ValueError(
                f"fcsts_mean_np.shape[0] != 1: {fcsts_mean_np.shape[0]} != 1, "
                "this is not expected, please open an issue on github"
            )
        fcsts_mean_np = fcsts_mean_np.squeeze(axis=0)
        if quantiles is not None:
            quantiles_torch = torch.tensor(
                quantiles,
                device=self.device,
                dtype=torch.float,
            )
            fcsts_quantiles = [
                fcst.quantile(quantiles_torch).cpu().numpy() for fcst in fcsts
            ]
            fcsts_quantiles_np = np.concatenate(fcsts_quantiles, axis=2)
            if fcsts_quantiles_np.shape[1] != 1:
                raise ValueError(
                    "fcsts_quantiles_np.shape[1] != 1: "
                    f"{fcsts_quantiles_np.shape[1]} != 1, "
                    "this is not expected, please open an issue on github"
                )
            fcsts_quantiles_np = np.moveaxis(fcsts_quantiles_np, 0, -1).squeeze(axis=0)
        else:
            fcsts_quantiles_np = None
        return fcsts_mean_np, fcsts_quantiles_np

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
        df = to_pandas(df)
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        dataset = TimeSeriesDataset.from_df(df, batch_size=self.batch_size)
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        with self._get_model() as model:
            fcsts_mean_np, fcsts_quantiles_np = self._forecast(
                model,
                dataset,
                h,
                quantiles=qc.quantiles,
            )
        fcst_df[self.alias] = fcsts_mean_np.reshape(-1, 1)
        if qc.quantiles is not None and fcsts_quantiles_np is not None:
            for i, q in enumerate(qc.quantiles):
                fcst_df[f"{self.alias}-q-{int(q * 100)}"] = fcsts_quantiles_np[
                    ..., i
                ].reshape(-1, 1)
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        return fcst_df
