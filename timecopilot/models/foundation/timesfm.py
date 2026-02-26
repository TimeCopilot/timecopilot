import os
from contextlib import contextmanager

import numpy as np
import pandas as pd
import timesfm
import timesfm_v1
import torch
from huggingface_hub import repo_exists
from timesfm import TimesFM_2p5_200M_torch
from timesfm_v1.timesfm_base import DEFAULT_QUANTILES as DEFAULT_QUANTILES_TFM
from tqdm import tqdm

from ..utils.forecaster import Forecaster, QuantileConverter
from ...utils.df_utils import to_pandas
from .utils import TimeSeriesDataset


class _TimesFMV1(Forecaster):
    def __init__(
        self,
        repo_id: str,
        context_length: int,
        batch_size: int,
        alias: str,
    ):
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias

    @contextmanager
    def _get_predictor(
        self,
        prediction_length: int,
        quantiles: list[float] | None = None,
    ) -> timesfm_v1.TimesFm:
        backend = "gpu" if torch.cuda.is_available() else "cpu"
        # these values are based on
        # https://github.com/google-research/timesfm/blob/ba034ae71c2fc88eaf59f80b4a778cc2c0dca7d6/experiments/extended_benchmarks/run_timesfm.py#L91
        v2_version = "2.0" in self.repo_id
        context_len = (
            min(self.context_length, 512) if not v2_version else self.context_length
        )
        num_layers = 50 if v2_version else 20
        use_positional_embedding = not v2_version

        tfm_hparams = timesfm_v1.TimesFmHparams(
            backend=backend,
            horizon_len=prediction_length,
            quantiles=quantiles,
            context_len=context_len,
            num_layers=num_layers,
            use_positional_embedding=use_positional_embedding,
            per_core_batch_size=self.batch_size,
        )
        if os.path.exists(self.repo_id):
            path = os.path.join(self.repo_id, "torch_model.ckpt")
            tfm_checkpoint = timesfm_v1.TimesFmCheckpoint(path=path)
            tfm = timesfm_v1.TimesFm(
                hparams=tfm_hparams,
                checkpoint=tfm_checkpoint,
            )
        elif repo_exists(self.repo_id):
            tfm_checkpoint = timesfm_v1.TimesFmCheckpoint(
                huggingface_repo_id=self.repo_id
            )
            tfm = timesfm_v1.TimesFm(
                hparams=tfm_hparams,
                checkpoint=tfm_checkpoint,
            )
        else:
            raise OSError(
                f"Failed to load model. Searched for '{self.repo_id}' "
                "as a local path to model directory and as a Hugging Face repo_id."
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
        df = to_pandas(df)
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        if qc.quantiles is not None and len(qc.quantiles) != len(DEFAULT_QUANTILES_TFM):
            raise ValueError(
                "TimesFM only supports the default quantiles, "
                "please use the default quantiles or default level, "
                "see https://github.com/google-research/timesfm/issues/286"
            )
        with self._get_predictor(
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
            renamer = {
                f"{self.alias}-q-{q}": f"{self.alias}-q-{int(q * 100)}"
                for q in qc.quantiles
            }
            fcst_df = fcst_df.rename(columns=renamer)
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        else:
            fcst_df = fcst_df[["unique_id", "ds", self.alias]]
        return fcst_df


class _TimesFMV2_p5(Forecaster):
    def __init__(
        self,
        repo_id: str,
        context_length: int,
        batch_size: int,
        alias: str,
        **kwargs: dict,
    ):
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias
        self.kwargs = kwargs

    @contextmanager
    def _get_predictor(
        self,
        prediction_length: int,
    ) -> TimesFM_2p5_200M_torch:
        # automatically detect the best device
        # https://github.com/AzulGarza/timesfm/blob/b810bbdf9f8a1e66396e7bd5cdb3b005e9116d86/src/timesfm/timesfm_2p5/timesfm_2p5_torch.py#L71
        if os.path.exists(self.repo_id):
            path = os.path.join(self.repo_id, "model.safetensors")
            tfm = TimesFM_2p5_200M_torch().model.load_checkpoint(path)
        elif repo_exists(self.repo_id):
            tfm = TimesFM_2p5_200M_torch.from_pretrained(self.repo_id)
        else:
            raise OSError(
                f"Failed to load model. Searched for '{self.repo_id}' "
                "as a local path to model directory and as a Hugging Face repo_id."
            )
        default_kwargs = {
            "max_context": self.context_length,
            "max_horizon": prediction_length,
            "normalize_inputs": True,
            "use_continuous_quantile_head": True,
            "fix_quantile_crossing": True,
        }
        passed_kwargs = self.kwargs or {}
        kwargs = {**default_kwargs, **passed_kwargs}
        config = timesfm.ForecastConfig(**kwargs)
        tfm.compile(config)
        try:
            yield tfm
        finally:
            del tfm
            torch.cuda.empty_cache()

    def _predict(
        self,
        model: TimesFM_2p5_200M_torch,
        dataset: TimeSeriesDataset,
        h: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        fcsts = [
            model.forecast(
                inputs=batch,
                horizon=h,
            )
            for batch in tqdm(dataset)
        ]
        fcsts_mean, fcsts_quantiles = zip(*fcsts, strict=False)
        fcsts_mean_np = np.concatenate(fcsts_mean)
        fcsts_quantiles_np = np.concatenate(fcsts_quantiles)
        return fcsts_mean_np, fcsts_quantiles_np

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        df = to_pandas(df)
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        if qc.quantiles is not None and len(qc.quantiles) != len(DEFAULT_QUANTILES_TFM):
            raise ValueError(
                "TimesFM only supports the default quantiles, "
                "please use the default quantiles or default level, "
                "see https://github.com/google-research/timesfm/issues/286"
            )
        dataset = TimeSeriesDataset.from_df(
            df,
            batch_size=self.batch_size,
            dtype=torch.float32,
        )
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        with self._get_predictor(prediction_length=h) as model:
            fcsts_mean_np, fcsts_quantiles_np = self._predict(
                model,
                dataset,
                h,
            )
        fcst_df[self.alias] = fcsts_mean_np.reshape(-1, 1)
        if qc.quantiles is not None:
            for i, q in enumerate(qc.quantiles):
                fcst_df[f"{self.alias}-q-{int(q * 100)}"] = fcsts_quantiles_np[
                    ..., i + 1  # skip the first quantile (mean)
                ].reshape(-1, 1)
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        return fcst_df


class TimesFM(Forecaster):
    """
    TimesFM is a large time series model for time series forecasting, supporting both
    probabilistic and point forecasts. See the [official repo](https://github.com/
    google-research/timesfm) for more details.
    """

    def __new__(
        cls,
        repo_id: str = "google/timesfm-2.0-500m-pytorch",
        context_length: int = 2048,
        batch_size: int = 64,
        alias: str = "TimesFM",
        **kwargs: dict,
    ):
        if "pytorch" not in repo_id:
            raise ValueError(
                "TimesFM only supports pytorch models, "
                "if you'd like to use jax, please open an issue"
            )
        if "1.0" in repo_id or "2.0" in repo_id:
            return _TimesFMV1(
                repo_id=repo_id,
                context_length=context_length,
                batch_size=batch_size,
                alias=alias,
            )
        elif "2.5" in repo_id:
            return _TimesFMV2_p5(
                repo_id=repo_id,
                context_length=context_length,
                batch_size=batch_size,
                alias=alias,
                **kwargs,
            )
        else:
            raise ValueError(
                "TimesFM only supports 1.0, 2.0 and 2.5 models, please use a "
                "valid model id"
            )

    def __init__(
        self,
        repo_id: str = "google/timesfm-2.0-500m-pytorch",
        context_length: int = 2048,
        batch_size: int = 64,
        alias: str = "TimesFM",
        kwargs: dict | None = None,
    ):
        """
        Args:
            repo_id (str, optional): The Hugging Face Hub model ID or local path to
                load the TimesFM model from. Examples include
                `google/timesfm-2.0-500m-pytorch`. Defaults to
                `google/timesfm-2.0-500m-pytorch`. See the full list of models at
                [Hugging Face](https://huggingface.co/collections/google/timesfm-release-
                66e4be5fdb56e960c1e482a6). Supported models:

                - `google/timesfm-1.0-200m-pytorch`
                - `google/timesfm-2.0-500m-pytorch`
                - `google/timesfm-2.5-200m-pytorch`
            context_length (int, optional): Maximum context length (input window size)
                for the model. Defaults to 2048. For TimesFM 2.0 models, max is 2048
                (must be a multiple of 32). For TimesFM 1.0 models, max is 512. See
                [TimesFM docs](https://github.com/google-research/timesfm#loading-the-
                model) for details.
            batch_size (int, optional): Batch size for inference. Defaults to 64.
                Adjust based on available memory and model size.
            alias (str, optional): Name to use for the model in output DataFrames and
                logs. Defaults to `TimesFM`.
            kwargs (dict, optional): Additional keyword arguments to pass to the model.
                Defaults to None. Only used for TimesFM 2.5 models.

        Notes:
            **Academic Reference:**

            - Paper: [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688)

            **Resources:**

            - GitHub: [google-research/timesfm](https://github.com/google-research/timesfm)
            - HuggingFace: [google/timesfm-release](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6)

            **Technical Details:**

            - Only PyTorch checkpoints are currently supported. JAX is not supported.
            - The model is loaded onto the best available device (GPU if available,
              otherwise CPU).

            **Supported Models:**

            - `google/timesfm-1.0-200m-pytorch`
            - `google/timesfm-2.0-500m-pytorch`
            - `google/timesfm-2.5-200m-pytorch`
        """
        pass
