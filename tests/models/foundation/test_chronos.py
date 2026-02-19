import torch

from timecopilot.models.foundation.utils import TimeSeriesDataset


def test_timeseries_dataset_default_dtype_is_float32():
    """Ensure TimeSeriesDataset defaults to float32 for numerical precision."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2020-01-01", periods=10),
            "y": range(10),
        }
    )
    dataset = TimeSeriesDataset.from_df(df, batch_size=10)
    assert dataset.data[0].dtype == torch.float32


def test_chronos_default_dtype_is_float32():
    """Ensure Chronos defaults to float32 dtype."""
    from timecopilot.models.foundation.chronos import Chronos

    model = Chronos(repo_id="amazon/chronos-t5-tiny")
    assert model.dtype == torch.float32


def test_chronos_model_uses_configured_dtype(mocker):
    """Ensure Chronos loads models with the configured dtype."""
    mock_pipeline = mocker.patch(
        "timecopilot.models.foundation.chronos.BaseChronosPipeline.from_pretrained"
    )
    mocker.patch("torch.cuda.is_available", return_value=False)

    from timecopilot.models.foundation.chronos import Chronos

    # Test default (float32)
    model = Chronos(repo_id="amazon/chronos-t5-tiny")
    with model._get_model():
        pass
    call_kwargs = mock_pipeline.call_args[1]
    assert call_kwargs["torch_dtype"] == torch.float32

    # Test custom dtype (bfloat16)
    mock_pipeline.reset_mock()
    model_bf16 = Chronos(repo_id="amazon/chronos-t5-tiny", dtype=torch.bfloat16)
    with model_bf16._get_model():
        pass
    call_kwargs = mock_pipeline.call_args[1]
    assert call_kwargs["torch_dtype"] == torch.bfloat16
