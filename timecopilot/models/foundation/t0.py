import sys
from contextlib import contextmanager

if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
    raise ImportError("T0 requires Python >= 3.11 and < 3.14")

import numpy as np
import pandas as pd
import torch
from t0 import T0Forecaster
from tqdm import tqdm

from ..utils.forecaster import Forecaster, QuantileConverter
from .utils import TimeSeriesDataset


class T0(Forecaster):
    """
    T0 is an open-weights time series foundation model from
    [The Forecasting Company](https://theforecastingcompany.com/). It is a
    decoder-style patch transformer that alternates time and covariate
    attention layers, producing probabilistic multi-horizon quantile
    forecasts. It decodes up to 1,024 timesteps in a single forward pass and
    falls back on autoregressive rollout for longer horizons. T0 natively
    handles numerical covariates, both historical (known over the past) and
    future (known over the forecast horizon). See the
    [model card](https://huggingface.co/theforecastingcompany/t0-alpha)
    for more details.
    """

    def __init__(
        self,
        repo_id: str = "theforecastingcompany/t0-alpha",
        context_length: int = 4096,
        batch_size: int = 16,
        alias: str = "t0-alpha",
    ):
        # ruff: noqa: E501
        """
        Args:
            repo_id (str, optional): The Hugging Face Hub model ID or local path to
                load the T0 model from. Defaults to "theforecastingcompany/t0-alpha".
                See the full list of models at
                [Hugging Face](https://huggingface.co/theforecastingcompany).
            context_length (int, optional): Maximum context length (input window
                size) for the model. Series longer than this are truncated to the
                most recent `context_length` observations. Defaults to 4096.
            batch_size (int, optional): Batch size to use for inference. Defaults
                to 16. Adjust based on available memory.
            alias (str, optional): Name to use for the model in output DataFrames
                and logs. Defaults to "t0-alpha".

        Notes:
            **Requirements:**

            - T0 requires Python 3.11 to 3.13 (via the
              [`tfc-t0`](https://pypi.org/project/tfc-t0/) package).

            **Available models:**

            | Model ID                                                                                          | Parameters |
            | ------------------------------------------------------------------------------------------------- | ---------- |
            | [`theforecastingcompany/t0-alpha`](https://huggingface.co/theforecastingcompany/t0-alpha)         | ~102M      |

            **Resources:**

            - HuggingFace: [theforecastingcompany/t0-alpha](https://huggingface.co/theforecastingcompany/t0-alpha)
            - Platform: [Retrocast](https://app.retrocast.com/)

            **Technical Details:**

            - The model is loaded onto the best available device (GPU if
              available, otherwise CPU).
            - T0 predicts 5 quantile knots (0.1, 0.25, 0.5, 0.75, 0.9); the
              median (0.5) is used as the point forecast and other requested
              quantiles are obtained by linear interpolation across the knots.
            - NaN values in the context are treated as missing observations.
            - T0 natively supports past and known-future covariates through its
              `predict` API; this integration currently exposes the univariate
              path only.
        """
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @contextmanager
    def _get_model(self) -> T0Forecaster:
        model = T0Forecaster.from_pretrained(self.repo_id).to(self.device).eval()
        try:
            yield model
        finally:
            del model
            torch.cuda.empty_cache()

    def _to_context(self, batch: list[torch.Tensor]) -> torch.Tensor:
        """Left-pad a ragged batch with NaN (treated as missing by T0)."""
        max_len = min(
            max(len(ts) for ts in batch),
            self.context_length,
        )
        context = torch.full(
            (len(batch), max_len),
            float("nan"),
            dtype=torch.float32,
        )
        for idx, ts in enumerate(batch):
            ts = ts[-max_len:]
            context[idx, -len(ts) :] = ts.to(dtype=torch.float32)
        return context

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
                = 100 × quantile value. Quantiles the model wasn't trained on
                are linearly interpolated across its fixed knots.

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
        qc = QuantileConverter(level=level, quantiles=quantiles)
        dataset = TimeSeriesDataset.from_df(df, batch_size=self.batch_size)
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        # T0 interpolates arbitrary quantile levels from its trained knots,
        # so the median and any user-requested quantiles come from one pass.
        pred_quantiles = sorted(set(qc.quantiles or []) | {0.5})
        median_idx = pred_quantiles.index(0.5)
        fcsts: list[np.ndarray] = []
        with self._get_model() as model:
            for batch in tqdm(dataset):
                out = model.predict(
                    self._to_context(batch),
                    horizon=h,
                    quantiles=pred_quantiles,
                )
                # shape: (batch, h, n_quantiles)
                fcsts.append(out.quantiles.cpu().numpy())
        fcsts_np = np.concatenate(fcsts, axis=0)
        fcst_df[self.alias] = fcsts_np[..., median_idx].reshape(-1, 1)
        if qc.quantiles is not None:
            for q in qc.quantiles:
                fcst_df[f"{self.alias}-q-{int(q * 100)}"] = fcsts_np[
                    ..., pred_quantiles.index(q)
                ].reshape(-1, 1)
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        return fcst_df
