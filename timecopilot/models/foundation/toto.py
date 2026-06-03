import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto as TotoModel
from toto2 import Toto2Model
from tqdm import tqdm

from ..utils.forecaster import Forecaster, QuantileConverter
from .utils import TimeSeriesDataset

# Config key that only appears in Toto 2.0 checkpoints (a Toto2ModelConfig field).
# Used to dispatch between the Toto 1.0 and Toto 2.0 backends.
_TOTO2_CONFIG_KEY = "num_variate_layers_per_group"


class Toto(Forecaster):
    """
    Toto is a family of foundation models for multivariate time series
    forecasting, optimized for observability and high-dimensional data. This
    class transparently supports both Toto 1.0 and Toto 2.0 checkpoints,
    dispatching to the appropriate backend based on the loaded model. See the
    [official repo](https://github.com/DataDog/toto) for more details.
    """

    def __init__(
        self,
        repo_id: str = "Datadog/Toto-Open-Base-1.0",
        context_length: int = 4096,
        batch_size: int = 16,
        num_samples: int = 128,
        samples_per_batch: int = 8,
        decode_block_size: int | None = None,
        alias: str = "Toto",
    ):
        # ruff: noqa: E501
        """
        Args:
            repo_id (str, optional): The Hugging Face Hub model ID or local path to
                load the Toto model from. This can be either a Toto 1.0 checkpoint
                (e.g. "Datadog/Toto-Open-Base-1.0") or a Toto 2.0 checkpoint (e.g.
                "Datadog/Toto-2.0-4m"). The model family is detected automatically
                from the checkpoint configuration. Defaults to
                "Datadog/Toto-Open-Base-1.0". See the full list of models at
                [Hugging Face](https://huggingface.co/Datadog).
            context_length (int, optional): Maximum context length (input window size)
                for the model. Defaults to 4096. Should match the configuration of the
                pretrained checkpoint. See [Toto docs](https://github.com/DataDog/toto#
                toto-model) for details.
            batch_size (int, optional): Batch size to use for inference. Defaults to 16.
                Adjust based on available memory and model size.
            num_samples (int, optional): Number of samples for probabilistic
                forecasting. Controls the number of forecast samples drawn for
                uncertainty estimation. Defaults to 128. Only used by Toto 1.0
                checkpoints; ignored by Toto 2.0, which predicts fixed quantile knots
                directly.
            samples_per_batch (int, optional): Number of samples processed per batch
                during inference. Controls memory usage. Defaults to 8. Only used by
                Toto 1.0 checkpoints; ignored by Toto 2.0.
            decode_block_size (int | None, optional): Block size for Toto 2.0 block
                decoding, expressed in time steps and divisible by the model patch
                size. When None (default), Toto 2.0 forecasts in a single forward
                pass, which is faster and better for short horizons. Larger values
                (e.g. 768) improve long-term stability for very long horizons. Only
                used by Toto 2.0 checkpoints; ignored by Toto 1.0.
            alias (str, optional): Name to use for the model in output DataFrames and
                logs. Defaults to "Toto".

        Notes:
            **Available models:**

            *Toto 1.0 (sample-based):*

            | Model ID                                                                            | Parameters |
            | ----------------------------------------------------------------------------------- | ---------- |
            | [`Datadog/Toto-Open-Base-1.0`](https://huggingface.co/Datadog/Toto-Open-Base-1.0)   | 151M       |

            *Toto 2.0 (quantile-knot based):*

            | Model ID                                                              | Parameters |
            | --------------------------------------------------------------------- | ---------- |
            | [`Datadog/Toto-2.0-4m`](https://huggingface.co/Datadog/Toto-2.0-4m)     | 4M         |
            | [`Datadog/Toto-2.0-22m`](https://huggingface.co/Datadog/Toto-2.0-22m)   | 22M        |
            | [`Datadog/Toto-2.0-313m`](https://huggingface.co/Datadog/Toto-2.0-313m) | 313M       |
            | [`Datadog/Toto-2.0-1B`](https://huggingface.co/Datadog/Toto-2.0-1B)     | 1B         |
            | [`Datadog/Toto-2.0-2.5B`](https://huggingface.co/Datadog/Toto-2.0-2.5B) | 2.5B       |

            **Academic Reference:**

            - Paper (Toto 1.0): [Building a Foundation Model for Time Series](https://arxiv.org/abs/2505.14766)
            - Paper (Toto 2.0): [Toto 2.0: Time Series Forecasting Enters the Scaling Era](https://arxiv.org/abs/2605.20119)

            **Resources:**

            - GitHub: [DataDog/toto](https://github.com/DataDog/toto)
            - HuggingFace: [Datadog Models](https://huggingface.co/Datadog)

            **Technical Details:**

            - The model is loaded onto the best available device (GPU if available,
              otherwise CPU).
            - For best performance, a CUDA-capable GPU is recommended.
            - Toto 1.0 draws probabilistic samples and reports the sample mean as the
              point forecast, with exact sample quantiles. Toto 2.0 predicts a fixed
              set of quantile knots (0.1, 0.2, ..., 0.9); the median (0.5) is used as
              the point forecast and requested quantiles are obtained by linear
              interpolation across the knots.
        """
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        # Number of samples for probabilistic forecasting (Toto 1.0 only)
        self.num_samples = num_samples
        # Control memory usage during inference (Toto 1.0 only)
        self.samples_per_batch = samples_per_batch
        # Block decoding size (Toto 2.0 only)
        self.decode_block_size = decode_block_size
        self.alias = alias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_toto2_cache: bool | None = None

    def _is_toto2(self) -> bool:
        """Detect whether ``repo_id`` points to a Toto 2.0 checkpoint.

        Detection is based on the model ``config.json``: Toto 2.0
        configurations contain Toto2-specific fields that are absent from
        Toto 1.0 checkpoints. The result is cached on the instance.
        """
        if self._is_toto2_cache is not None:
            return self._is_toto2_cache
        repo_path = Path(self.repo_id)
        if repo_path.is_dir():
            config_path = repo_path / "config.json"
        else:
            config_path = Path(
                hf_hub_download(repo_id=self.repo_id, filename="config.json")
            )
        config = json.loads(config_path.read_text())
        self._is_toto2_cache = _TOTO2_CONFIG_KEY in config
        return self._is_toto2_cache

    @contextmanager
    def _get_model(self) -> TotoForecaster | Toto2Model:
        if self._is_toto2():
            model = Toto2Model.from_pretrained(self.repo_id).to(self.device).eval()
            try:
                yield model
            finally:
                del model
                torch.cuda.empty_cache()
        else:
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
                series_length = self.context_length
            padded_tensor[idx, -series_length:] = ts.to(device=self.device, dtype=torch.float)
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

    def _to_toto2_inputs(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Build the Toto 2.0 ``forecast`` inputs dict for a batch.

        Produces left-padded ``target``/``target_mask`` tensors of shape
        ``(batch, n_var=1, context_length)`` together with zero
        ``series_ids`` of shape ``(batch, n_var=1)``.
        """
        batch_size = len(batch)
        padded_tensor = torch.zeros(
            batch_size,
            self.context_length,
            dtype=torch.float,
            device=self.device,
        )
        padding_mask = torch.zeros(
            batch_size,
            self.context_length,
            dtype=torch.bool,
            device=self.device,
        )
        for idx, ts in enumerate(batch):
            series_length = len(ts)
            if series_length > self.context_length:
                ts = ts[-self.context_length :]
                series_length = self.context_length
            padded_tensor[idx, -series_length:] = ts.to(device=self.device, dtype=torch.float)
            padding_mask[idx, -series_length:] = True
        # add the variate dimension (n_var=1)
        target = padded_tensor.unsqueeze(1)
        target_mask = padding_mask.unsqueeze(1)
        series_ids = torch.zeros(
            batch_size,
            1,
            dtype=torch.long,
            device=self.device,
        )
        return {
            "target": target,
            "target_mask": target_mask,
            "series_ids": series_ids,
        }

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

        fcsts_mean = [fcst.median.cpu().numpy() for fcst in fcsts]
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

    def _forecast_toto2(
        self,
        model: Toto2Model,
        dataset: TimeSeriesDataset,
        h: int,
        quantiles: list[float] | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Forecast with a Toto 2.0 model.

        Toto 2.0 predicts a fixed set of quantile knots (0.1, ..., 0.9). The
        median knot is used as the point forecast, and requested quantiles are
        obtained by linear interpolation across the knots.
        """
        knots = list(model.output_head.knots)
        median_idx = knots.index(0.5)
        # collect per-batch quantile knots of shape (n_knots, batch, h)
        knot_fcsts: list[np.ndarray] = []
        for batch in tqdm(dataset):
            inputs = self._to_toto2_inputs(batch)
            with torch.no_grad():
                # shape: (n_knots, batch, n_var=1, h)
                q = model.forecast(
                    inputs,
                    horizon=h,
                    decode_block_size=self.decode_block_size,
                    has_missing_values=True,
                )
            q_np = q.float().cpu().numpy()
            if q_np.shape[2] != 1:
                raise ValueError(
                    f"toto2 forecast n_var != 1: {q_np.shape[2]} != 1, "
                    "this is not expected, please open an issue on github"
                )
            knot_fcsts.append(q_np.squeeze(axis=2))
        # shape: (n_knots, n_series, h)
        knots_np = np.concatenate(knot_fcsts, axis=1)
        fcsts_mean_np = knots_np[median_idx]
        if quantiles is not None:
            # interpolate requested quantiles across the fixed knots, per
            # (series, horizon) position. np.interp clamps at the edge knots.
            knots_arr = np.asarray(knots)
            fcsts_quantiles_np = np.stack(
                [
                    np.apply_along_axis(
                        lambda col, _q=q: np.interp(_q, knots_arr, col),
                        axis=0,
                        arr=knots_np,
                    )
                    for q in quantiles
                ],
                axis=-1,
            )
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
                = 100 × quantile value. For Toto 2.0 checkpoints, quantiles are
                linearly interpolated across the model's fixed knots.

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
        forecast_fn = self._forecast_toto2 if self._is_toto2() else self._forecast
        with self._get_model() as model:
            fcsts_mean_np, fcsts_quantiles_np = forecast_fn(
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
