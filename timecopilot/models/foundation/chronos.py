from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from chronos import (
    BaseChronosPipeline,
    Chronos2Pipeline,
    ChronosBoltPipeline,
    ChronosPipeline,
)
from tqdm import tqdm

from ..utils.forecaster import Forecaster, QuantileConverter
from .utils import TimeSeriesDataset


@dataclass
class ChronosFinetuningConfig:
    """Configuration for finetuning a Chronos pipeline before forecasting.

    Pass an instance to the ``Chronos`` constructor; when you call
    ``forecast()``, the model is finetuned on the context data before
    predicting. The forecast horizon ``h`` from ``forecast(df, h, ...)`` is
    used as ``prediction_length`` for the internal ``fit()``. Parameters are
    passed through to the chronos pipeline's ``fit()``, with ``finetune_steps``
    mapped to the library's ``num_steps``.

    Attributes:
        finetune_steps: Number of training steps. Passed to the pipeline as
            ``num_steps``. Defaults to 1000.
        learning_rate: Optimizer learning rate. Defaults to None (chronos
            uses 1e-6; for LoRA, 1e-5 is recommended).
        batch_size: Training batch size for finetuning. Defaults to None
            (chronos uses 256). The ``batch_size`` on ``Chronos`` is for
            inference only.
        finetune_mode: ``"full"`` (full parameter update) or ``"lora"``
            (low-rank adaptation). Defaults to None (chronos uses ``"full"``).
        lora_config: LoRA configuration when ``finetune_mode="lora"``. Defaults
            to None. See the Chronos-2 quickstart for details.
        save_path: If set, the finetuned model is saved to this directory (path
            or str). Use this same path as ``repo_id`` when creating
            ``Chronos(repo_id=save_path, finetuning_config=None)`` for
            subsequent forecasting without finetuning.

    Notes:
        - Based on the [Chronos-2 quickstart](https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb).
    """

    finetune_steps: int = 1000
    learning_rate: float | None = None
    batch_size: int | None = None
    finetune_mode: Literal["full", "lora"] | None = None
    lora_config: Any = None
    save_path: str | Path | None = None


class Chronos(Forecaster):
    """
    Chronos models are large pre-trained models for time series forecasting,
    supporting both probabilistic and point forecasts. See the
    [official repo](https://github.com/amazon-science/chronos-forecasting)
    for more details.
    """

    def __init__(
        self,
        repo_id: str = "amazon/chronos-t5-large",
        batch_size: int = 16,
        alias: str = "Chronos",
        dtype: torch.dtype = torch.float32,
        finetuning_config: ChronosFinetuningConfig | None = None,
    ):
        # ruff: noqa: E501
        """
        Args:
            repo_id (str, optional): The Hugging Face Hub model ID or local
                path to load the Chronos model from. Examples include
                "amazon/chronos-t5-tiny", "amazon/chronos-t5-large", or a
                local directory. You can also pass a path where a finetuned
                model was saved (see ``finetuning_config.save_path``); use
                that path as ``repo_id`` with ``finetuning_config=None`` to
                reuse the saved model. Defaults to "amazon/chronos-t5-large".
                See the full list of available models at
                [Hugging Face](https://huggingface.co/collections/
                amazon/chronos-models-65f1791d630a8d57cb718444)
            batch_size (int, optional): Batch size to use for inference only.
                Larger models may require smaller batch sizes due to GPU
                memory constraints. Defaults to 16. For Chronos-Bolt models,
                higher batch sizes (e.g., 256) are possible. When finetuning,
                use ``finetuning_config.batch_size`` to set the training
                batch size (optional; library default when not set).
            alias (str, optional): Name to use for the model in output
                DataFrames and logs. Defaults to "Chronos".
            dtype (torch.dtype, optional): Data type for model weights and
                input tensors. Defaults to torch.float32 for numerical
                precision. Use torch.bfloat16 for reduced memory usage on
                supported hardware.
            finetuning_config (ChronosFinetuningConfig | None, optional): If
                provided, the model is finetuned on the forecast context
                data before predicting. Set ``save_path`` on the config to
                save the finetuned model; then use that path as ``repo_id``
                with ``finetuning_config=None`` for later forecasts. See
                ChronosFinetuningConfig and the
                [Chronos-2 quickstart](https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb)
                for parameter details.

        Notes:
            **Available models:**

            | Model ID                                                               | Parameters |
            | ---------------------------------------------------------------------- | ---------- |
            | [`amazon/chronos-2`](https://huggingface.co/amazon/chronos-2)   | 120M         |
            | [`autogluon/chronos-2-synth`](https://huggingface.co/autogluon/chronos-2-synth)   | 120M         |
            | [`autogluon/chronos-2-small`](https://huggingface.co/autogluon/chronos-2-small)   | 28M         |
            | [`amazon/chronos-bolt-tiny`](https://huggingface.co/amazon/chronos-bolt-tiny)   | 9M         |
            | [`amazon/chronos-bolt-mini`](https://huggingface.co/amazon/chronos-bolt-mini)   | 21M        |
            | [`amazon/chronos-bolt-small`](https://huggingface.co/amazon/chronos-bolt-small) | 48M        |
            | [`amazon/chronos-bolt-base`](https://huggingface.co/amazon/chronos-bolt-base)   | 205M       |
            | [`amazon/chronos-t5-tiny`](https://huggingface.co/amazon/chronos-t5-tiny)   | 8M         |
            | [`amazon/chronos-t5-mini`](https://huggingface.co/amazon/chronos-t5-mini)   | 20M        |
            | [`amazon/chronos-t5-small`](https://huggingface.co/amazon/chronos-t5-small) | 46M        |
            | [`amazon/chronos-t5-base`](https://huggingface.co/amazon/chronos-t5-base)   | 200M       |
            | [`amazon/chronos-t5-large`](https://huggingface.co/amazon/chronos-t5-large) | 710M       |

            **Academic Reference:**

            - Paper: [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)

            **Resources:**

            - GitHub: [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
            - HuggingFace: [amazon/chronos-models](https://huggingface.co/collections/amazon/chronos-models-65f1791d630a8d57cb718444)

            **Technical Details:**

            - The model is loaded onto the best available device (GPU if
              available, otherwise CPU).
            - For best performance with large models (e.g., "chronos-t5-large"),
              a CUDA-compatible GPU is recommended.
            - Model weights and input tensors use dtype (default: torch.float32)
              for numerical precision. Can be overridden via the dtype parameter.

        """
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.alias = alias
        self.dtype = dtype
        self.finetuning_config = finetuning_config

    @staticmethod
    def _build_fit_inputs_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
        """Build list of fit inputs from a DataFrame (unique_id, ds, y)."""
        df_sorted = df.sort_values(by=["unique_id", "ds"])
        return [
            {"target": group["y"].values} for _, group in df_sorted.groupby("unique_id")
        ]

    def _maybe_finetune(
        self,
        model: BaseChronosPipeline,
        df: pd.DataFrame,
        h: int,
    ) -> BaseChronosPipeline:
        """If finetuning_config is set, finetune the model on df and return it."""
        if self.finetuning_config is None:
            return model
        if not hasattr(model, "fit"):
            raise ValueError(
                f"Finetuning is not supported for model {self.repo_id}; "
                "the loaded pipeline has no fit method."
            )
        train_inputs = self._build_fit_inputs_from_df(df)
        fit_kwargs: dict[str, Any] = {
            "inputs": train_inputs,
            "prediction_length": h,
            "num_steps": self.finetuning_config.finetune_steps,
        }
        if self.finetuning_config.learning_rate is not None:
            fit_kwargs["learning_rate"] = self.finetuning_config.learning_rate
        if self.finetuning_config.batch_size is not None:
            fit_kwargs["batch_size"] = self.finetuning_config.batch_size
        if self.finetuning_config.finetune_mode is not None:
            fit_kwargs["finetune_mode"] = self.finetuning_config.finetune_mode
        if self.finetuning_config.lora_config is not None:
            fit_kwargs["lora_config"] = self.finetuning_config.lora_config
        if self.finetuning_config.save_path is not None:
            sp = Path(self.finetuning_config.save_path)
            fit_kwargs["output_dir"] = str(sp.parent)
            fit_kwargs["finetuned_ckpt_name"] = sp.name
        return model.fit(**fit_kwargs)

    @contextmanager
    def _get_model(self) -> BaseChronosPipeline:
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        repo_path = Path(self.repo_id)
        # LoRA checkpoints save adapter_config.json; BaseChronosPipeline.from_pretrained
        # uses AutoConfig and fails. Chronos2Pipeline.from_pretrained handles LoRA via PEFT.
        if repo_path.is_dir() and (repo_path / "adapter_config.json").exists():
            cls = Chronos2Pipeline
        else:
            cls = BaseChronosPipeline
        model = cls.from_pretrained(
            self.repo_id,
            device_map=device_map,
            torch_dtype=self.dtype,
        )
        try:
            yield model
        finally:
            del model
            torch.cuda.empty_cache()

    def _predict(
        self,
        model: BaseChronosPipeline,
        dataset: TimeSeriesDataset,
        h: int,
        quantiles: list[float] | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """handles distinction between predict and predict_quantiles"""
        if quantiles is not None:
            fcsts = [
                model.predict_quantiles(
                    batch,
                    prediction_length=h,
                    quantile_levels=quantiles,
                )
                for batch in tqdm(dataset)
            ]  # list of tuples
            fcsts_quantiles, fcsts_mean = zip(*fcsts, strict=False)
            if isinstance(model, Chronos2Pipeline):
                fcsts_mean = [f_mean for fcst in fcsts_mean for f_mean in fcst]  # type: ignore
                fcsts_quantiles = [
                    f_quantile
                    for fcst in fcsts_quantiles
                    for f_quantile in fcst  # type: ignore
                ]
            fcsts_mean_np = torch.cat(fcsts_mean).numpy()
            fcsts_quantiles_np = torch.cat(fcsts_quantiles).numpy()
        else:
            fcsts = [
                model.predict(
                    batch,
                    prediction_length=h,
                )
                for batch in tqdm(dataset)
            ]
            if isinstance(model, Chronos2Pipeline):
                fcsts = [f_fcst for fcst in fcsts for f_fcst in fcst]  # type: ignore
            fcsts = torch.cat(fcsts)
            if isinstance(model, ChronosPipeline):
                # for t5 models, `predict` returns a tensor of shape
                # (batch_size, num_samples, prediction_length).
                # notice that the method return samples.
                # see https://github.com/amazon-science/chronos-forecasting/blob/6a9c8dadac04eb85befc935043e3e2cce914267f/src/chronos/chronos.py#L450-L537
                # also for these models, the following is how the mean is computed
                # in the `predict_quantiles` method
                # see https://github.com/amazon-science/chronos-forecasting/blob/6a9c8dadac04eb85befc935043e3e2cce914267f/src/chronos/chronos.py#L554
                fcsts_mean = fcsts.mean(dim=1)  # type: ignore
            elif isinstance(model, ChronosBoltPipeline | Chronos2Pipeline):
                # for bolt models, `predict` returns a tensor of shape
                # (batch_size, num_quantiles, prediction_length)
                # notice that in this case, the method returns the default quantiles
                # instead of samples
                # see https://github.com/amazon-science/chronos-forecasting/blob/6a9c8dadac04eb85befc935043e3e2cce914267f/src/chronos/chronos_bolt.py#L479-L563
                # for these models, the median is prefered as mean forecasts
                # as it can be seen in
                # https://github.com/amazon-science/chronos-forecasting/blob/6a9c8dadac04eb85befc935043e3e2cce914267f/src/chronos/chronos_bolt.py#L615-L616
                fcsts_mean = fcsts[:, model.quantiles.index(0.5), :]  # type: ignore
            else:
                raise ValueError(f"Unsupported model: {self.repo_id}")
            fcsts_mean_np = fcsts_mean.numpy()  # type: ignore
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

            When ``finetuning_config`` was set at construction, the model is
            finetuned on ``df`` before predicting.
        """
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        dataset = TimeSeriesDataset.from_df(
            df, batch_size=self.batch_size, dtype=self.dtype
        )
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        with self._get_model() as model:
            model = self._maybe_finetune(model, df, h)
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
