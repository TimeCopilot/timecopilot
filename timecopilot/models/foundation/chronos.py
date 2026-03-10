import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

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

logger = logging.getLogger(__name__)


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
    ):
        # ruff: noqa: E501
        """
        Args:
            repo_id (str, optional): The Hugging Face Hub model ID or local
                path to load the Chronos model from. Examples include
                "amazon/chronos-t5-tiny", "amazon/chronos-t5-large", or a
                local directory. Defaults to "amazon/chronos-t5-large". See
                the full list of available models at
                [Hugging Face](https://huggingface.co/collections/
                amazon/chronos-models-65f1791d630a8d57cb718444)
            batch_size (int, optional): Batch size to use for inference.
                Larger models may require smaller batch sizes due to GPU
                memory constraints. Defaults to 16. For Chronos-Bolt models,
                higher batch sizes (e.g., 256) are possible.
            alias (str, optional): Name to use for the model in output
                DataFrames and logs. Defaults to "Chronos".
            dtype (torch.dtype, optional): Data type for model weights and
                input tensors. Defaults to torch.float32 for numerical
                precision. Use torch.bfloat16 for reduced memory usage on
                supported hardware.

        Notes:
            **Available models:**

            | Model ID                                                               | Parameters |
            | ---------------------------------------------------------------------- | ---------- |
            | [`amazon/chronos-2`](https://huggingface.co/amazon/chronos-2)   | 120M         |
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

    @contextmanager
    def _get_model(self) -> BaseChronosPipeline:
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = BaseChronosPipeline.from_pretrained(
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
        """
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        dataset = TimeSeriesDataset.from_df(
            df, batch_size=self.batch_size, dtype=self.dtype
        )
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


class ChronosFinetuner:
    def __init__(
        self,
        repo_id: str,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            repo_id (str): The Hugging Face Hub model ID or local path to the
                Chronos model to fine-tune. Examples: "amazon/chronos-2",
                "amazon/chronos-t5-small", or a local directory.
            device (str, optional): Device to use for fine-tuning (e.g. "cuda:0",
                "cpu"). If not provided, defaults to "cuda:0" when a GPU is
                available, otherwise "cpu".
            dtype (torch.dtype, optional): Data type for model weights and tensors.
                Defaults to torch.float32. Use torch.bfloat16 for reduced memory
                usage on supported hardware.
        """
        self.repo_id = repo_id
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._infer_model_family()

    def _infer_model_family(self) -> None:
        repo_lower = self.repo_id.lower()
        if "chronos-2" in repo_lower or "chronos2" in repo_lower:
            self.model_family = "chronos-2"
        elif "chronos-bolt" in repo_lower:
            self.model_family = "chronos-bolt"
        elif "chronos-t5" in repo_lower:
            self.model_family = "chronos-t5"
        else:
            self.model_family = "unknown"
            logger.warning(
                "Could not infer model family from %s. Will attempt at runtime.",
                self.repo_id,
            )

    @staticmethod
    def prepare_arrow_dataset(
        time_series: list[np.ndarray],
        output_path: str | Path,
        start_date: str = "2000-01-01",
        compression: str = "lz4",
    ) -> Path:
        try:
            from gluonts.dataset.arrow import ArrowWriter
        except ImportError as e:
            raise ImportError(
                f"prepare_arrow_dataset requires gluonts: {e}\n"
                "Install with: pip install gluonts"
            ) from e

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        start = np.datetime64(start_date, "s")
        dataset = [{"start": start, "target": ts} for ts in time_series]

        logger.info("Writing %d series to Arrow: %s", len(dataset), output_path)
        ArrowWriter(compression=compression).write_to_file(dataset, path=output_path)
        return output_path

    def finetune(
        self,
        *args,
        **kwargs,
    ):
        """Dispatch to the appropriate fine-tuning method based on the model family.

        All positional and keyword arguments are forwarded to the underlying
        method (``_finetune_chronos2``, ``_finetune_chronos_t5``, or
        ``_finetune_chronos_bolt``). See each method's docstring for the
        accepted parameters.

        Returns:
            The result of the dispatched fine-tuning method.

        Raises:
            ValueError: If the model family could not be inferred from
                ``self.repo_id``.
        """
        if self.model_family == "chronos-2":
            return self._finetune_chronos2(*args, **kwargs)
        if self.model_family == "chronos-t5":
            return self._finetune_chronos_t5(*args, **kwargs)
        if self.model_family == "chronos-bolt":
            return self._finetune_chronos_bolt(*args, **kwargs)
        raise ValueError(
            f"Cannot determine fine-tuning method for repo_id '{self.repo_id}' "
            f"(model_family='{self.model_family}'). "
            "Ensure the repo_id contains 'chronos-2', 'chronos-bolt', or 'chronos-t5'."
        )

    def _finetune_chronos2(
        self,
        inputs: torch.Tensor | np.ndarray | list | pd.DataFrame,
        prediction_length: int,
        validation_inputs: torch.Tensor | np.ndarray | list | None = None,
        finetune_mode: str = "full",
        lora_config: dict | None = None,
        context_length: int | None = None,
        learning_rate: float = 1e-6,
        num_steps: int = 1000,
        batch_size: int = 256,
        output_dir: str | Path | None = None,
        min_past: int | None = None,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        **extra_trainer_kwargs,
    ) -> Chronos2Pipeline:
        if self.model_family not in ("chronos-2", "unknown"):
            raise ValueError(
                f"_finetune_chronos2 requires Chronos-2, got {self.repo_id} "
                f"(family: {self.model_family})"
            )

        pipeline = Chronos2Pipeline.from_pretrained(
            self.repo_id,
            device_map=self.device,
            torch_dtype=self.dtype,
        )

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = Path("chronos-2-finetuned") / timestamp
        else:
            output_dir = Path(output_dir)

        logger.info(
            "Chronos-2 finetune: mode=%s lr=%s steps=%s bs=%s out=%s",
            finetune_mode,
            learning_rate,
            num_steps,
            batch_size,
            output_dir,
        )

        return pipeline.fit(
            inputs=inputs,
            prediction_length=prediction_length,
            validation_inputs=validation_inputs,
            finetune_mode=finetune_mode,
            lora_config=lora_config,
            context_length=context_length,
            learning_rate=learning_rate,
            num_steps=num_steps,
            batch_size=batch_size,
            output_dir=output_dir,
            min_past=min_past,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            **extra_trainer_kwargs,
        )

    def _finetune_chronos_t5(
        self,
        data_path: str | Path,
        context_length: int = 512,
        prediction_length: int = 64,
        freq: str = "D",
        learning_rate: float = 1e-3,
        max_steps: int = 10_000,
        per_device_batch_size: int = 32,
        save_steps: int = 500,
        log_steps: int = 100,
        output_dir: str | Path | None = None,
        gradient_accumulation_steps: int = 1,
        warmup_ratio: float = 0.0,
        max_missing_prop: float = 0.1,
        min_past: int | None = None,
        seed: int = 42,
        torch_compile: bool = False,
        tf32: bool = True,
        dataloader_num_workers: int = 1,
        **extra_trainer_kwargs,
    ) -> Path:
        if self.model_family not in ("chronos-t5", "unknown"):
            raise ValueError(
                f"_finetune_chronos_t5 requires Chronos-T5, got {self.repo_id} "
                f"(family: {self.model_family})"
            )

        try:
            from gluonts.dataset.common import FileDataset
            from gluonts.transform import (
                ExpectedNumInstanceSampler,
                FilterTransformation,
                InstanceSplitter,
            )
            from transformers import (
                AutoConfig,
                AutoModelForSeq2SeqLM,
                Trainer,
                TrainingArguments,
            )
        except ImportError as e:
            raise ImportError(
                f"Chronos-T5 finetuning requires extra deps: {e}\n"
                "Install with: pip install transformers gluonts"
            ) from e

        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if min_past is None:
            min_past = prediction_length

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = Path("chronos-t5-finetuned") / timestamp
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config = AutoConfig.from_pretrained(self.repo_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.repo_id, torch_dtype=self.dtype
        )
        model = model.to(self.device)

        gts_dataset = FileDataset(path=data_path, freq=freq)

        instance_splitter = InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=min_past,
                min_future=prediction_length,
            ),
            past_length=context_length,
            future_length=prediction_length,
            dummy_value=np.nan,
        )

        transformed_data = gts_dataset.transform(instance_splitter)
        transformed_data = transformed_data.transform(
            FilterTransformation(
                lambda x: (
                    len(x["target"]) >= min_past + prediction_length
                    and np.isnan(x["target"]).mean() <= max_missing_prop
                )
            )
        )

        from chronos import ChronosConfig

        chronos_config = ChronosConfig(**config.chronos_config)
        tokenizer = chronos_config.create_tokenizer()

        class T5TrainingDataset:
            def __iter__(self):
                for entry in transformed_data:
                    target = np.array(entry["target"], dtype=np.float32)
                    ctx = target[:-prediction_length]
                    label = target[-prediction_length:]

                    if len(ctx) < min_past:
                        continue
                    if np.isnan(label).mean() > max_missing_prop:
                        continue

                    ctx_token, ctx_mask, scale = tokenizer.context_input_transform(
                        torch.tensor(ctx).unsqueeze(0)
                    )
                    label_token, _, _ = tokenizer.label_input_transform(
                        torch.tensor(label).unsqueeze(0), scale
                    )

                    yield {
                        "input_ids": ctx_token[0],
                        "attention_mask": ctx_mask[0],
                        "labels": label_token[0],
                    }

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            warmup_ratio=warmup_ratio,
            max_steps=max_steps,
            save_steps=save_steps,
            logging_steps=log_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=dataloader_num_workers,
            seed=seed,
            tf32=tf32,
            torch_compile=torch_compile,
            save_total_limit=1,
            save_strategy="steps",
            report_to="none",
            remove_unused_columns=False,
            **extra_trainer_kwargs,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=T5TrainingDataset(),
        )

        logger.info("Starting Chronos-T5 training...")
        trainer.train()

        checkpoints = list(output_dir.glob("checkpoint-*"))
        if checkpoints:
            return sorted(checkpoints, key=lambda p: int(p.name.split("-")[-1]))[-1]
        return output_dir

    def _finetune_chronos_bolt(self, *args, **kwargs):
        raise NotImplementedError("Chronos-Bolt fine-tuning not implemented here.")

    def load_finetuned(
        self, repo_id: str | None = None
    ) -> ChronosPipeline | Chronos2Pipeline:
        """Load a fine-tuned Chronos pipeline.

        Args:
            repo_id (str, optional): Path or Hugging Face Hub ID of the
                fine-tuned checkpoint. If not provided, ``self.repo_id`` is used.

        Returns:
            A ``Chronos2Pipeline`` or ``ChronosPipeline`` loaded from the
            checkpoint.

        Raises:
            NotImplementedError: For Chronos-Bolt (not yet supported).
            ValueError: For unknown model families.
        """
        effective_repo_id = repo_id if repo_id is not None else self.repo_id
        if self.model_family == "chronos-2":
            return Chronos2Pipeline.from_pretrained(
                effective_repo_id,
                device_map=self.device,
                torch_dtype=self.dtype,
            )
        if self.model_family == "chronos-t5":
            return ChronosPipeline.from_pretrained(
                effective_repo_id,
                device_map=self.device,
                torch_dtype=self.dtype,
            )
        if self.model_family == "chronos-bolt":
            raise NotImplementedError("load_finetuned not supported for Chronos-Bolt.")
        raise ValueError(f"Unknown model family: {self.model_family}")
