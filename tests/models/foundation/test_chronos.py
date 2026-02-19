import torch


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


def test_chronos_forecast_uses_configured_dtype(mocker):
    """Ensure Chronos.forecast uses the configured dtype for dataset creation."""
    import pandas as pd

    import pytest

    from timecopilot.models.foundation.chronos import Chronos

    # Patch dataset creation to capture dtype argument
    mock_from_df = mocker.patch(
        "timecopilot.models.foundation.chronos.TimeSeriesDataset.from_df"
    )

    # Avoid real model loading and CUDA branching
    mocker.patch(
        "timecopilot.models.foundation.chronos.BaseChronosPipeline.from_pretrained"
    )
    mocker.patch("torch.cuda.is_available", return_value=False)

    model_dtype = torch.bfloat16
    model = Chronos(repo_id="amazon/chronos-t5-tiny", dtype=model_dtype)

    df = pd.DataFrame(
        {
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2020-01-01", periods=10),
            "y": range(10),
        }
    )

    def _from_df_side_effect(*args, **kwargs):
        # Assert that Chronos.forecast passes the configured dtype through
        assert kwargs.get("dtype") == model_dtype
        # Short-circuit the rest of the forecast call
        raise RuntimeError("stop after dtype check")

    mock_from_df.side_effect = _from_df_side_effect

    with pytest.raises(RuntimeError, match="stop after dtype check"):
        model.forecast(df=df, h=2)
