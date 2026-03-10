"""
Test script for TimesFM fine-tuning capability.
Demonstrates usage of different fine-tuning strategies.
"""

import logging
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from timecopilot.models.foundation.timesfm import (
    FineTuningConfig,
    TimeSeriesFineTuningDataset,
    TimesFMFineTuner,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_dataset(
    num_samples: int = 1000,
    context_length: int = 96,
    prediction_length: int = 24,
    num_series: int = 5,
) -> pd.DataFrame:
    """Generate synthetic time series data for testing."""
    logger.info(f"Generating synthetic dataset with {num_samples} samples")

    data = []
    for series_id in range(num_series):
        # Generate time series with trend and seasonality
        t = np.arange(num_samples)
        trend = np.linspace(0, 10, num_samples)
        seasonal = 5 * np.sin(2 * np.pi * t / 24)
        noise = np.random.normal(0, 0.5, num_samples)
        values = trend + seasonal + noise

        for i in range(num_samples):
            data.append(
                {
                    "unique_id": f"series_{series_id}",
                    "ds": pd.Timestamp("2020-01-01") + pd.Timedelta(hours=i),
                    "y": values[i],
                }
            )

    df = pd.DataFrame(data)
    logger.info(f"Dataset shape: {df.shape}")
    return df


def create_mock_model() -> nn.Module:
    """Create a simple mock TimesFM model for testing."""
    logger.info("Creating mock TimesFM model")

    class MockTimesFM(nn.Module):
        def __init__(self, input_size: int = 96, output_size: int = 24):
            super().__init__()
            self.input_size = input_size
            self.output_size = output_size

            # Simple transformer-like architecture
            self.embedding = nn.Linear(1, 128)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True),
                num_layers=4,
            )
            self.output_projection = nn.Linear(128, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch_size, context_length)
            x = x.unsqueeze(-1)  # (batch_size, context_length, 1)
            x = self.embedding(x)  # (batch_size, context_length, 128)
            x = self.transformer(x)  # (batch_size, context_length, 128)
            x = x.mean(dim=1)  # (batch_size, 128)
            x = self.output_projection(x)  # (batch_size, output_size)
            return x

    return MockTimesFM()


def test_full_finetuning():
    """Test full fine-tuning strategy."""
    logger.info("=" * 80)
    logger.info("Testing Full Fine-Tuning Strategy")
    logger.info("=" * 80)

    # Generate data
    df = generate_synthetic_dataset(num_samples=500)

    # Split into train/val
    train_size = int(0.8 * len(df))
    train_df = df[:train_size].reset_index(drop=True)
    val_df = df[train_size:].reset_index(drop=True)

    # Create datasets
    context_length = 96
    prediction_length = 24

    train_dataset = TimeSeriesFineTuningDataset(
        train_df,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    val_dataset = TimeSeriesFineTuningDataset(
        val_df,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    # Create model
    model = create_mock_model()

    # Configure full fine-tuning
    config = FineTuningConfig(
        batch_size=16,
        num_epochs=3,
        learning_rate=1e-4,
        use_lora=False,
        use_dora=False,
        use_linear_probing=False,
        early_stopping_patience=2,
    )

    # Create fine-tuner and train
    with tempfile.TemporaryDirectory() as tmpdir:
        config.checkpoint_dir = tmpdir
        finetuner = TimesFMFineTuner(model, config)
        history = finetuner.finetune(train_dataset, val_dataset)

        logger.info(f"Training history: {history['history']}")
        logger.info(f"Best validation loss: {history['best_val_loss']:.4f}")

    logger.info("Full Fine-Tuning test PASSED ✓\n")


def test_lora_finetuning():
    """Test LoRA fine-tuning strategy."""
    logger.info("=" * 80)
    logger.info("Testing LoRA Fine-Tuning Strategy")
    logger.info("=" * 80)

    # Generate data
    df = generate_synthetic_dataset(num_samples=500)

    # Split into train/val
    train_size = int(0.8 * len(df))
    train_df = df[:train_size].reset_index(drop=True)
    val_df = df[train_size:].reset_index(drop=True)

    # Create datasets
    context_length = 96
    prediction_length = 24

    train_dataset = TimeSeriesFineTuningDataset(
        train_df,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    val_dataset = TimeSeriesFineTuningDataset(
        val_df,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    # Create model
    model = create_mock_model()

    # Configure LoRA fine-tuning
    config = FineTuningConfig(
        batch_size=16,
        num_epochs=3,
        learning_rate=1e-3,
        use_lora=True,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.1,
        early_stopping_patience=2,
    )

    # Create fine-tuner and train
    with tempfile.TemporaryDirectory() as tmpdir:
        config.checkpoint_dir = tmpdir
        finetuner = TimesFMFineTuner(model, config)
        history = finetuner.finetune(train_dataset, val_dataset)

        logger.info(f"Training history: {history['history']}")
        logger.info(f"Best validation loss: {history['best_val_loss']:.4f}")

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    logger.info("LoRA Fine-Tuning test PASSED ✓\n")


def test_dora_finetuning():
    """Test DoRA fine-tuning strategy."""
    logger.info("=" * 80)
    logger.info("Testing DoRA Fine-Tuning Strategy")
    logger.info("=" * 80)

    # Generate data
    df = generate_synthetic_dataset(num_samples=500)

    # Split into train/val
    train_size = int(0.8 * len(df))
    train_df = df[:train_size].reset_index(drop=True)
    val_df = df[train_size:].reset_index(drop=True)

    # Create datasets
    context_length = 96
    prediction_length = 24

    train_dataset = TimeSeriesFineTuningDataset(
        train_df,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    val_dataset = TimeSeriesFineTuningDataset(
        val_df,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    # Create model
    model = create_mock_model()

    # Configure DoRA fine-tuning
    config = FineTuningConfig(
        batch_size=16,
        num_epochs=3,
        learning_rate=1e-3,
        use_lora=True,
        use_dora=True,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.1,
        early_stopping_patience=2,
    )

    # Create fine-tuner and train
    with tempfile.TemporaryDirectory() as tmpdir:
        config.checkpoint_dir = tmpdir
        finetuner = TimesFMFineTuner(model, config)
        history = finetuner.finetune(train_dataset, val_dataset)

        logger.info(f"Training history: {history['history']}")
        logger.info(f"Best validation loss: {history['best_val_loss']:.4f}")

    logger.info("DoRA Fine-Tuning test PASSED ✓\n")


def test_linear_probing():
    """Test Linear Probing fine-tuning strategy."""
    logger.info("=" * 80)
    logger.info("Testing Linear Probing Fine-Tuning Strategy")
    logger.info("=" * 80)

    # Generate data
    df = generate_synthetic_dataset(num_samples=500)

    # Split into train/val
    train_size = int(0.8 * len(df))
    train_df = df[:train_size].reset_index(drop=True)
    val_df = df[train_size:].reset_index(drop=True)

    # Create datasets
    context_length = 96
    prediction_length = 24

    train_dataset = TimeSeriesFineTuningDataset(
        train_df,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    val_dataset = TimeSeriesFineTuningDataset(
        val_df,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    # Create model
    model = create_mock_model()

    # Configure Linear Probing
    config = FineTuningConfig(
        batch_size=16,
        num_epochs=3,
        learning_rate=1e-4,
        use_linear_probing=True,
        early_stopping_patience=2,
    )

    # Create fine-tuner and train
    with tempfile.TemporaryDirectory() as tmpdir:
        config.checkpoint_dir = tmpdir
        finetuner = TimesFMFineTuner(model, config)
        history = finetuner.finetune(train_dataset, val_dataset)

        logger.info(f"Training history: {history['history']}")
        logger.info(f"Best validation loss: {history['best_val_loss']:.4f}")

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    logger.info("Linear Probing test PASSED ✓\n")


def test_dataset():
    """Test TimeSeriesFineTuningDataset."""
    logger.info("=" * 80)
    logger.info("Testing TimeSeriesFineTuningDataset")
    logger.info("=" * 80)

    df = generate_synthetic_dataset(num_samples=200)
    context_length = 48
    prediction_length = 12

    dataset = TimeSeriesFineTuningDataset(
        df,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    logger.info(f"Dataset length: {len(dataset)}")
    context, target = dataset[0]
    logger.info(f"Context shape: {context.shape}, expected: ({context_length},)")
    logger.info(f"Target shape: {target.shape}, expected: ({prediction_length},)")

    assert context.shape == (context_length,), f"Invalid context shape: {context.shape}"
    assert target.shape == (prediction_length,), f"Invalid target shape: {target.shape}"

    logger.info("Dataset test PASSED ✓\n")


if __name__ == "__main__":
    logger.info("\n" + "=" * 80)
    logger.info("TimesFM Fine-Tuning Tests")
    logger.info("=" * 80 + "\n")

    try:
        # Test dataset
        test_dataset()

        # Test different fine-tuning strategies
        test_full_finetuning()
        test_lora_finetuning()
        test_dora_finetuning()
        test_linear_probing()

        logger.info("=" * 80)
        logger.info("All tests PASSED ✓")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        exit(1)
