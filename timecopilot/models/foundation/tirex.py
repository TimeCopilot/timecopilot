import os
import sys
from contextlib import contextmanager

if sys.version_info < (3, 11):
    raise ImportError("TiRex requires Python >= 3.11")

import numpy as np
import pandas as pd
import torch
from tirex import load_model
from tirex.base import PretrainedModel
from tqdm import tqdm

from ..utils.forecaster import Forecaster, QuantileConverter
from ...utils.df_utils import to_pandas
from .utils import TimeSeriesDataset


class TiRex(Forecaster):
    """
    TiRex is a zero-shot time series forecasting model based on xLSTM,
    supporting both point and quantile predictions for long and short horizons.
    See the [official repo](https://github.com/NX-AI/tirex) for more details.
    """

    def __init__(
        self,
        repo_id: str = "NX-AI/TiRex",
        batch_size: int = 16,
        alias: str = "TiRex",
    ):
        """
        Args:
            repo_id (str, optional): The Hugging Face Hub model ID or local path to load
                the TiRex model from. Examples include "NX-AI/TiRex". Defaults to
                "NX-AI/TiRex". See the full list of models at
                [Hugging Face](https://huggingface.co/NX-AI).
            batch_size (int, optional): Batch size to use for inference. Defaults to 16.
                Adjust based on available memory and model size.
            alias (str, optional): Name to use for the model in output DataFrames
                and logs. Defaults to "TiRex".

        Notes:
            **Academic Reference:**

            - Paper: [TiRex: Zero-shot Time Series Forecasting with xLSTM](https://arxiv.org/abs/2505.23719)

            **Resources:**

            - GitHub: [NX-AI/tirex](https://github.com/NX-AI/tirex)
            - HuggingFace: [NX-AI Models](https://huggingface.co/NX-AI)

            **Technical Details:**

            - The model is loaded onto the best available device (GPU if available,
              otherwise CPU).
            - On CPU, CUDA kernels are disabled automatically. See the
              [CUDA kernels section](https://github.com/NX-AI/tirex#cuda-kernels)
              for details.
            - For best performance, a CUDA-capable GPU with compute capability >= 8.0
              is recommended.
            - The model is only available for Python >= 3.11.
        """
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.alias = alias

    @contextmanager
    def _get_model(self) -> PretrainedModel:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            # see https://github.com/NX-AI/tirex/tree/main?tab=readme-ov-file#cuda-kernels
            os.environ["TIREX_NO_CUDA"] = "1"
        model = load_model(self.repo_id, device=device)
        try:
            yield model
        finally:
            del model
            torch.cuda.empty_cache()

    def _forecast(
        self,
        model: PretrainedModel,
        dataset: TimeSeriesDataset,
        h: int,
        quantiles: list[float] | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """handles distinction between quantiles and no quantiles"""
        if quantiles is not None:
            fcsts = [
                model.forecast(
                    batch,
                    prediction_length=h,
                    quantile_levels=quantiles,
                    output_type="numpy",
                )
                for batch in tqdm(dataset)
            ]  # list of tuples
            fcsts_quantiles, fcsts_mean = zip(*fcsts, strict=False)
            fcsts_quantiles_np = np.concatenate(fcsts_quantiles)
            fcsts_mean_np = np.concatenate(fcsts_mean)
        else:
            fcsts = [
                model.forecast(
                    batch,
                    prediction_length=h,
                    output_type="numpy",
                )
                for batch in tqdm(dataset)
            ]
            fcsts_quantiles, fcsts_mean = zip(*fcsts, strict=False)
            fcsts_mean_np = np.concatenate(fcsts_mean)
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
