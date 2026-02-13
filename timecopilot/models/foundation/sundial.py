import sys
from contextlib import contextmanager

if sys.version_info >= (3, 13):
    raise ImportError("Sundial requires Python < 3.13")

import numpy as np
import pandas as pd
import torch
from gluonts.transform import LastValueImputation
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from ..utils.forecaster import Forecaster, QuantileConverter
from ...utils.df_utils import to_pandas
from .utils import TimeSeriesDataset


class Sundial(Forecaster):
    """
    Sundial is a family of generative time series foundation models,
    pre-trained on TimeBench (10^12 time points). It uses the TimeFlow Loss to
    predict next-patch distributions, allowing Transformers to be trained without
    discrete tokenization and make non-deterministic predictions. The model supports
    both point and probabilistic zero-shot forecasting. See the
    [official repo](https://github.com/thuml/Sundial) for more details.
    """

    def __init__(
        self,
        repo_id: str = "thuml/sundial-base-128m",
        num_samples: int = 100,
        context_length: int = 2_880,
        batch_size: int = 1_024,
        alias: str = "Sundial",
    ):
        """
        Args:
            repo_id (str, optional): The Hugging Face Hub model ID or local path to
                load the Sundial model from. Examples include
                "thuml/sundial-base-128m". Defaults to "thuml/sundial-base-128m".
                See the full list of models at [Hugging Face](https://huggingface.co/
                thuml/sundial-base-128m).
            num_samples (int, optional): Number of samples to generate for
                probabilistic forecasting. More samples provide better distribution
                estimates but increase computation time. Defaults to 100.
            context_length (int, optional): Maximum context length (input window size)
                for the model. Controls how much history is used for each forecast.
                Defaults to 2,880. The model supports different lookback lengths.
            batch_size (int, optional): Batch size for inference. Defaults to 1,024.
                Adjust based on available memory and model size. Larger batch sizes
                can improve throughput but require more GPU memory.
            alias (str, optional): Name to use for the model in output DataFrames and
                logs. Defaults to "Sundial".

        Notes:
            **Academic Reference:**

            - Paper: [Sundial: A Family of Highly Capable Time Series Foundation Models](https://arxiv.org/abs/2502.00816)

            **Resources:**

            - GitHub: [thuml/Sundial](https://github.com/thuml/Sundial)
            - HuggingFace: [thuml/sundial-base-128m](https://huggingface.co/thuml/sundial-base-128m)

            **Technical Details:**

            - The model is loaded onto the best available device (GPU if
              available, otherwise CPU).
            - The model weights are loaded with torch_dtype=torch.bfloat16 for
              efficiency on supported hardware.
            - The model is only available for Python < 3.13.
        """
        self.repo_id = repo_id
        self.num_samples = num_samples
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16

    @contextmanager
    def _get_model(self) -> AutoModelForCausalLM:
        model = AutoModelForCausalLM.from_pretrained(
            self.repo_id,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        try:
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
        model: AutoModelForCausalLM,
        batch: list[torch.Tensor],
        h: int,
        quantiles: list[float] | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        context = self._prepare_and_validate_context(batch)
        if context.shape[1] > self.context_length:
            context = context[..., -self.context_length :]
        context = self._maybe_impute_missing(context)
        context = context.to(self.device)
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            # (batch_size, num_samples, h)
            samples = model.generate(
                context,
                max_new_tokens=h,
                revin=True,
                num_samples=self.num_samples,
            )
        q_median = torch.tensor(
            0.5,
            device=self.device,
            dtype=samples.dtype,
        )
        fcst_mean = torch.quantile(
            samples,
            q=q_median,
            dim=1,
        )
        fcst_mean_np = fcst_mean.cpu().numpy()
        if quantiles is not None:
            quantiles_torch = torch.tensor(
                quantiles,
                device=self.device,
                dtype=samples.dtype,
            )
            # (num_quantiles, batch_size, h)
            fcst_quantiles = torch.quantile(
                samples,
                q=quantiles_torch,
                dim=1,
            )
            fcst_quantiles_np = fcst_quantiles.cpu().numpy()
            # (batch_size, h, num_quantiles)
            fcst_quantiles_np = np.moveaxis(fcst_quantiles_np, 0, -1)
        else:
            fcst_quantiles_np = None
        return fcst_mean_np, fcst_quantiles_np

    def _predict(
        self,
        model: AutoModelForCausalLM,
        dataset: TimeSeriesDataset,
        h: int,
        quantiles: list[float] | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        fcsts = [
            self._predict_batch(model, batch, h, quantiles) for batch in tqdm(dataset)
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
        df = to_pandas(df)
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        dataset = TimeSeriesDataset.from_df(df, batch_size=self.batch_size)
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        with self._get_model() as model:
            fcsts_mean_np, fcsts_quantiles_np = self._predict(
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
