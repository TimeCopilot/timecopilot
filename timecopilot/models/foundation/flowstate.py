from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from gluonts.transform import LastValueImputation
from tqdm import tqdm
from tsfm_public import FlowStateForPrediction
from tsfm_public.models.flowstate.utils.utils import get_fixed_factor

from ..utils.forecaster import Forecaster, QuantileConverter
from .utils import TimeSeriesDataset


class FlowState(Forecaster):
    """
    FlowState is the first time-scale adjustable Time Series Foundation Model (TSFM),
    open-sourced by IBM Research. Combining a State Space Model (SSM) Encoder with a
    Functional Basis Decoder allows FlowState to transition into a timescale invariant
    coefficient space and make a continuous forecast from this space. This allows
    FlowState to seamlessly adjust to all possible sampling rates.

    See the [official repo](https://github.com/ibm-granite/granite-tsfm) and
    [paper](https://arxiv.org/abs/2508.05287) for more details.
    """

    def __init__(
        self,
        repo_id: str = "ibm-research/flowstate",
        scale_factor: float | None = None,
        context_length: int = 2_048,
        batch_size: int = 1_024,
        alias: str = "FlowState",
    ):
        """
        Initialize FlowState time series foundation model.

        Args:
            repo_id (str, optional): The Hugging Face Hub model ID or local path to
                load the FlowState model from. Supported models:

                - `ibm-research/flowstate` (default)
                - `ibm-granite/granite-timeseries-flowstate-r1`.

            scale_factor (float, optional): Scale factor for temporal adaptation.
                If None, will be automatically determined based on the time series
                frequency. The scale factor adjusts the model to different sampling
                rates. For example, if your data has seasonality every N=96 time steps
                (quarter hourly with daily cycle), scale_factor = 24/96 = 0.25.
            context_length (int, optional): Maximum context length (input window size)
                for the model. Controls how much history is used for each forecast.
                Defaults to 2,048. The model supports flexible context lengths.
            batch_size (int, optional): Batch size for inference. Defaults to 1,024.
                Adjust based on available memory and model size. Larger batch sizes
                can improve throughput but require more GPU memory.
            alias (str, optional): Name to use for the model in output DataFrames and
                logs. Defaults to "FlowState".

        Notes:
            **Academic Reference:**

            - Paper: [FlowState: Sampling Rate Invariant Time Series Forecasting](https://arxiv.org/abs/2508.05287)

            **Resources:**

            - GitHub: [ibm-granite/granite-tsfm](https://github.com/ibm-granite/granite-tsfm)
            - HuggingFace Models: [ibm-granite/granite-timeseries-flowstate-r1](
                https://huggingface.co/ibm-granite/granite-timeseries-flowstate-r1
              ), [ibm-research/flowstate](https://huggingface.co/ibm-research/flowstate).

            **Technical Details:**

            - The model is loaded onto the best available device (GPU if
              available, otherwise CPU).
            - FlowState uses State Space Model (SSM) encoder with Functional
              Basis Decoder (FBD) for time-scale invariant forecasting.
            - Recommended forecasting horizon: no more than 30 seasons.

            **Supported Models:**

            - `ibm-research/flowstate` (default)
            - `ibm-granite/granite-timeseries-flowstate-r1`.
        """
        self.repo_id = repo_id
        self.scale_factor = scale_factor
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

    @contextmanager
    def _get_model(self) -> FlowStateForPrediction:
        model = FlowStateForPrediction.from_pretrained(self.repo_id).to(self.device)
        try:
            model.eval()
            yield model
        finally:
            del model
            torch.cuda.empty_cache()

    def _left_pad_and_stack_1D(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        max_len = max(len(c) for c in tensors)
        padded = []
        for c in tensors:
            assert isinstance(c, torch.Tensor)
            assert c.ndim == 1
            padding = torch.full(
                size=(max_len - len(c),),
                fill_value=torch.nan,
                device=c.device,
                dtype=c.dtype,
            )
            padded.append(torch.concat((padding, c), dim=-1))
        return torch.stack(padded)

    def _prepare_and_validate_context(
        self,
        context: list[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(context, list):
            context = self._left_pad_and_stack_1D(context)
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2
        return context

    def _maybe_impute_missing(self, batch: torch.Tensor) -> torch.Tensor:
        if torch.isnan(batch).any():
            batch = batch.float().numpy()
            imputed_rows = []
            for i in range(batch.shape[0]):
                row = batch[i]
                imputed_row = LastValueImputation()(row)
                imputed_rows.append(imputed_row)
            batch = np.vstack(imputed_rows)
            batch = torch.tensor(
                batch,
                dtype=self.dtype,
                device=self.device,
            )
        return batch

    def _predict_batch(
        self,
        model: FlowStateForPrediction,
        batch: list[torch.Tensor],
        h: int,
        quantiles: list[float] | None,
        supported_quantiles: list[float],
        scale_factor: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        context = self._prepare_and_validate_context(batch)
        if context.shape[1] > self.context_length:
            context = context[..., -self.context_length :]
        context = self._maybe_impute_missing(context)
        # context is (batch, context_length)
        # then we convert it to (context_length, batch, 1)
        context = context.unsqueeze(-1).transpose(0, 1)
        context = context.to(self.device)
        # (batch, quantiles, h, n_ch)
        fcst = model(
            context,
            prediction_length=h,
            scale_factor=scale_factor,
            batch_first=False,
        ).prediction_outputs
        fcst = fcst.squeeze(-1).transpose(-1, -2)  # now shape is (batch, h, quantiles)
        fcst_mean = fcst[..., supported_quantiles.index(0.5)].squeeze()
        fcst_mean_np = fcst_mean.detach().numpy(force=True)
        fcst_quantiles_np = (
            fcst.detach().numpy(force=True) if quantiles is not None else None
        )
        return fcst_mean_np, fcst_quantiles_np

    def _predict(
        self,
        model: FlowStateForPrediction,
        dataset: TimeSeriesDataset,
        h: int,
        quantiles: list[float] | None,
        supported_quantiles: list[float],
        scale_factor: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        fcsts = [
            self._predict_batch(
                model,
                batch,
                h,
                quantiles,
                supported_quantiles,
                scale_factor,
            )
            for batch in tqdm(dataset)
        ]  # list of tuples
        fcsts_mean_tp, fcsts_quantiles_tp = zip(*fcsts, strict=False)
        fcsts_mean_np = np.concatenate(fcsts_mean_tp)
        if quantiles is not None:
            fcsts_quantiles_np = np.concatenate(fcsts_quantiles_tp)
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
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        dataset = TimeSeriesDataset.from_df(
            df,
            batch_size=self.batch_size,
            dtype=self.dtype,
        )
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        scale_factor = self.scale_factor or get_fixed_factor(freq)
        with self._get_model() as model:
            cfg = model.config
            supported_quantiles = cfg.quantiles
            if qc.quantiles is not None and not np.allclose(
                qc.quantiles,
                supported_quantiles,
            ):
                raise ValueError(
                    "FlowState only supports the default quantiles, "
                    f"supported quantiles are {supported_quantiles}, "
                    "please use the default quantiles or default level, "
                )
            fcsts_mean_np, fcsts_quantiles_np = self._predict(
                model,
                dataset,
                h,
                quantiles=qc.quantiles,
                supported_quantiles=supported_quantiles,
                scale_factor=scale_factor,
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
