import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import timesfm
import timesfm_v1
import torch
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import repo_exists
from timesfm import TimesFM_2p5_200M_torch
from timesfm_v1.timesfm_base import DEFAULT_QUANTILES as DEFAULT_QUANTILES_TFM
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..utils.forecaster import Forecaster, QuantileConverter
from .utils import TimeSeriesDataset

__all__ = [
    "TimesFM",
    "TimesFMFineTuner",
    "TimesFMFineTuningConfig",
    "TimeSeriesFineTuningDataset",
]


@dataclass
class TimesFMFineTuningConfig:
    """Configuration for TimesFM model fine-tuning."""

    # Training parameters
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # Fine-tuning strategy
    use_lora: bool = False
    use_dora: bool = False
    use_linear_probing: bool = False

    # LoRA/DoRA parameters
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1

    # Learning rate schedule
    use_cosine_schedule: bool = True
    warmup_steps: int = 0  # reserved

    # Early stopping
    early_stopping_patience: int = 5

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_every_n_steps: int = 50

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

        self.scaling = self.alpha / self.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_out = self.lora_b(self.lora_a(self.lora_dropout(x)))
        return lora_out * self.scaling


class DoRALayer(nn.Module):
    """Magnitude-scaled LoRA adapter layer.

    .. note::
        This is a pragmatic approximation of the DoRA paper
        (Liu et al., 2024, "DoRA: Weight-Decomposed Low-Rank Adaptation").
        Strict DoRA decomposes the *combined* weight matrix
        (base + LoRA delta) into a direction (column-normalized) and a
        learnable per-output magnitude vector.  The implementation here
        instead applies the magnitude vector directly to the LoRA output
        activation, which is simpler but not equivalent to the paper's
        formulation.  For exact DoRA semantics, use a PEFT library such as
        ``peft`` (HuggingFace).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)

        self.magnitude = nn.Parameter(torch.ones(out_features))

        nn.init.kaiming_uniform_(self.lora_a.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

        self.scaling = self.alpha / self.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        direction = self.lora_b(self.lora_a(self.lora_dropout(x)))
        direction = direction * self.scaling
        return direction * self.magnitude


class _LoRALinearWrapper(nn.Module):
    """Wrap an existing nn.Linear and add a LoRA/DoRA residual."""

    def __init__(self, base: nn.Linear, adapter: nn.Module):
        super().__init__()
        self.base = base
        self.adapter = adapter

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.adapter(x)


def _set_module_by_qualname(
    root: nn.Module, qualname: str, new_module: nn.Module
) -> None:
    """
    Replace a submodule inside `root` by its qualified name
    (as produced by named_modules()).
    """
    parts = qualname.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


class FineTuningStrategy(ABC):
    """Base class for fine-tuning strategies."""

    def __init__(self, model: nn.Module, config: TimesFMFineTuningConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def prepare_model(self):
        """Prepare model for fine-tuning."""
        pass

    @abstractmethod
    def get_trainable_params(self) -> list[nn.Parameter]:
        """Get parameters that should be trained."""
        pass


class LoRAFineTuning(FineTuningStrategy):
    """LoRA fine-tuning strategy."""

    def prepare_model(self):
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.lora_layers: list[tuple[str, LoRALayer]] = []

        # Inject wrappers into Linear layers that look like transformer blocks.
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and (
                "transformer" in name or "stacked_transformer" in name
            ):
                lora_layer = LoRALayer(
                    module.in_features,
                    module.out_features,
                    rank=self.config.lora_rank,
                    alpha=self.config.lora_alpha,
                    dropout=self.config.lora_dropout,
                )
                wrapped = _LoRALinearWrapper(module, lora_layer)
                _set_module_by_qualname(self.model, name, wrapped)
                self.lora_layers.append((name, lora_layer))

    def get_trainable_params(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for _, lora_layer in self.lora_layers:
            params.extend(list(lora_layer.parameters()))
        return params


class DoRAFineTuning(FineTuningStrategy):
    """DoRA fine-tuning strategy."""

    def prepare_model(self):
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.dora_layers: list[tuple[str, DoRALayer]] = []

        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and (
                "transformer" in name or "stacked_transformer" in name
            ):
                dora_layer = DoRALayer(
                    module.in_features,
                    module.out_features,
                    rank=self.config.lora_rank,
                    alpha=self.config.lora_alpha,
                    dropout=self.config.lora_dropout,
                )
                wrapped = _LoRALinearWrapper(module, dora_layer)
                _set_module_by_qualname(self.model, name, wrapped)
                self.dora_layers.append((name, dora_layer))

    def get_trainable_params(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for _, dora_layer in self.dora_layers:
            params.extend(list(dora_layer.parameters()))
        return params


class LinearProbingFineTuning(FineTuningStrategy):
    """Linear Probing fine-tuning strategy."""

    def prepare_model(self):
        for name, param in self.model.named_parameters():
            if "transformer" in name or "stacked_transformer" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def get_trainable_params(self) -> list[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]


class FullFineTuning(FineTuningStrategy):
    """Full fine-tuning strategy - train all parameters."""

    def prepare_model(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> list[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]


class TimeSeriesFineTuningDataset(Dataset):
    """Dataset wrapper for time series fine-tuning."""

    def __init__(
        self,
        df: pd.DataFrame,
        context_length: int,
        prediction_length: int,
        value_col: str = "y",
    ):
        self.df = df
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.value_col = value_col

        self.data = df[value_col].values.astype(np.float32)
        self.indices = self._create_indices()

    def _create_indices(self) -> list[tuple[int, int]]:
        indices: list[tuple[int, int]] = []
        max_start = len(self.data) - self.context_length - self.prediction_length
        for i in range(max_start + 1):
            indices.append((i, i + self.context_length))
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        context_start, target_start = self.indices[idx]

        context = torch.tensor(
            self.data[context_start:target_start],
            dtype=torch.float32,
        )

        target_end = target_start + self.prediction_length
        target = torch.tensor(
            self.data[target_start:target_end],
            dtype=torch.float32,
        )

        if len(target) < self.prediction_length:
            target = torch.cat(
                [
                    target,
                    torch.zeros(
                        self.prediction_length - len(target), dtype=torch.float32
                    ),
                ]
            )

        return context, target


class TimesFMFineTuner:
    """Fine-tuner for TimesFM torch models."""

    def __init__(self, model: nn.Module, config: TimesFMFineTuningConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.logger = self._setup_logging()

        self.strategy = self._create_strategy()
        self.strategy.prepare_model()

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def _create_strategy(self) -> FineTuningStrategy:
        if self.config.use_lora and self.config.use_dora:
            self.logger.info("Using DoRA fine-tuning strategy")
            return DoRAFineTuning(self.model, self.config)
        elif self.config.use_lora:
            self.logger.info("Using LoRA fine-tuning strategy")
            return LoRAFineTuning(self.model, self.config)
        elif self.config.use_linear_probing:
            self.logger.info("Using Linear Probing fine-tuning strategy")
            return LinearProbingFineTuning(self.model, self.config)
        else:
            self.logger.info("Using Full fine-tuning strategy")
            return FullFineTuning(self.model, self.config)

    def finetune(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
    ) -> dict[str, Any]:
        self.model = self.model.to(self.device)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

        trainable_params = self.strategy.get_trainable_params()
        optimizer = optim.Adam(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler = (
            self._create_cosine_scheduler(optimizer, train_loader)
            if self.config.use_cosine_schedule
            else None
        )

        self.logger.info("Starting fine-tuning for %d epochs", self.config.num_epochs)
        self.logger.info("Training samples: %d", len(train_dataset))
        if val_dataset is not None:
            self.logger.info("Validation samples: %d", len(val_dataset))

        for epoch in range(self.config.num_epochs):
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            self.training_history["train_loss"].append(train_loss)
            self.training_history["learning_rate"].append(
                optimizer.param_groups[0]["lr"]
            )

            self.logger.info(
                "Epoch %d/%d - Train Loss: %.6f",
                epoch + 1,
                self.config.num_epochs,
                train_loss,
            )

            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.training_history["val_loss"].append(val_loss)
                self.logger.info("Val Loss: %.6f", val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(epoch)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.early_stopping_patience:
                        self.logger.info("Early stopping triggered")
                        break

        self.logger.info("Fine-tuning completed")
        return {"history": self.training_history, "best_val_loss": self.best_val_loss}

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Any | None = None,
    ) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_idx, (context, target) in enumerate(
            tqdm(train_loader, desc="Training")
        ):
            context = context.to(self.device)
            target = target.to(self.device)

            optimizer.zero_grad()

            output = self.model(context)
            if isinstance(output, tuple):
                output = output[0]

            loss = nn.functional.mse_loss(output, target)

            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += float(loss.item())

            if (batch_idx + 1) % self.config.log_every_n_steps == 0:
                self.logger.debug("Batch %d - Loss: %.6f", batch_idx + 1, loss.item())

        return total_loss / max(1, len(train_loader))

    def _validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for context, target in tqdm(val_loader, desc="Validating"):
                context = context.to(self.device)
                target = target.to(self.device)

                output = self.model(context)
                if isinstance(output, tuple):
                    output = output[0]

                loss = nn.functional.mse_loss(output, target)
                total_loss += float(loss.item())

        return total_loss / max(1, len(val_loader))

    def _create_cosine_scheduler(
        self, optimizer: optim.Optimizer, train_loader: DataLoader
    ) -> Any:
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

        total_steps = len(train_loader) * self.config.num_epochs
        t0 = max(1, total_steps // 10)
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t0,
            T_mult=1,
            eta_min=1e-6,
        )

    def _save_checkpoint(self, epoch: int) -> None:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "best_val_loss": self.best_val_loss,
                "config": self.config.__dict__,
            },
            checkpoint_path,
        )
        self.logger.info("Checkpoint saved: %s", checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.logger.info("Checkpoint loaded: %s", checkpoint_path)


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

    def create_finetuner(self, *args: Any, **kwargs: Any) -> TimesFMFineTuner:
        raise NotImplementedError(
            "Fine-tuning is only supported for torch-native TimesFM models in this "
            "integration. Use a TimesFM 2.5 pytorch checkpoint "
            "(google/timesfm-2.5-200m-pytorch)."
        )


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

    def _load_torch_backbone(self) -> nn.Module:
        """
        Load the underlying torch model for fine-tuning.
        Returns an nn.Module that can be trained.
        """
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

        # Prefer a torch nn.Module if present
        if isinstance(tfm, nn.Module):
            return tfm
        if hasattr(tfm, "model") and isinstance(tfm.model, nn.Module):
            return tfm.model

        raise TypeError(
            "Loaded TimesFM 2.5 object does not expose a torch.nn.Module"
            " for fine-tuning."
        )

    def create_finetuner(self, config: TimesFMFineTuningConfig) -> TimesFMFineTuner:
        backbone = self._load_torch_backbone()
        return TimesFMFineTuner(backbone, config)

    def finetune_from_df(
        self,
        df: pd.DataFrame,
        prediction_length: int,
        config: TimesFMFineTuningConfig,
        value_col: str = "y",
        train_split: float = 0.8,
    ) -> dict[str, Any]:
        """
        Convenience helper:
        - builds a sliding-window dataset from df[value_col]
        - fine-tunes the torch backbone
        """
        backbone = self._load_torch_backbone()
        tuner = TimesFMFineTuner(backbone, config)

        n = len(df)
        split_idx = max(1, int(n * train_split))
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        val_df = df.iloc[split_idx:].reset_index(drop=True) if split_idx < n else None

        train_ds = TimeSeriesFineTuningDataset(
            df=train_df,
            context_length=self.context_length,
            prediction_length=prediction_length,
            value_col=value_col,
        )
        val_ds = None
        if val_df is not None and len(val_df) >= (
            self.context_length + prediction_length + 1
        ):
            val_ds = TimeSeriesFineTuningDataset(
                df=val_df,
                context_length=self.context_length,
                prediction_length=prediction_length,
                value_col=value_col,
            )

        return tuner.finetune(train_dataset=train_ds, val_dataset=val_ds)


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
