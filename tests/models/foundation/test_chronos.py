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


def test_chronos_finetuning_save_and_reuse(tmp_path):
    """Finetune with save_path, run cross-validation,
    then forecast using the saved path."""
    from ..test_models import generate_series
    from timecopilot.models.foundation.chronos import Chronos, ChronosFinetuningConfig

    save_path = tmp_path / "chronos2-finetuned"
    config = ChronosFinetuningConfig(
        finetune_steps=2,
        save_path=save_path,
    )
    model = Chronos(
        repo_id="autogluon/chronos-2-small",
        finetuning_config=config,
        batch_size=2,
    )
    n_series = 2
    df = generate_series(n_series, freq="MS")

    cv_df = model.cross_validation(df, h=2, n_windows=1, freq="MS")
    assert not cv_df.empty
    assert "Chronos" in cv_df.columns

    assert save_path.is_dir(), f"Finetuned model should be saved to {save_path}"
    assert (save_path / "config.json").exists(), "Expected config.json "

    model_reuse = Chronos(
        repo_id=str(save_path),
        finetuning_config=None,
        batch_size=2,
    )
    fcst = model_reuse.forecast(df, h=2, freq="MS")
    assert not fcst.empty
    assert "Chronos" in fcst.columns
    assert len(fcst) == n_series * 2  # h=2 per series


def test_chronos_lora_finetuning_save_and_reuse(tmp_path):
    """Finetune Chronos-2 with LoRA and save_path, then load from path and forecast."""
    from ..test_models import generate_series
    from timecopilot.models.foundation.chronos import Chronos, ChronosFinetuningConfig

    save_path = tmp_path / "chronos2-lora-finetuned"
    config = ChronosFinetuningConfig(
        finetune_steps=2,
        finetune_mode="lora",
        learning_rate=1e-5,
        save_path=save_path,
    )
    model = Chronos(
        repo_id="autogluon/chronos-2-small",
        finetuning_config=config,
        batch_size=2,
    )
    n_series = 2
    df = generate_series(n_series, freq="MS")

    fcst = model.forecast(df, h=2, freq="MS")
    assert not fcst.empty
    assert "Chronos" in fcst.columns

    assert save_path.is_dir(), f"LoRA checkpoint should be saved to {save_path}"
    assert (save_path / "adapter_config.json").exists(), "Expected adapter_config.json "

    model_reuse = Chronos(
        repo_id=str(save_path),
        finetuning_config=None,
        batch_size=2,
    )
    fcst_reuse = model_reuse.forecast(df, h=2, freq="MS")
    assert not fcst_reuse.empty
    assert "Chronos" in fcst_reuse.columns
    assert len(fcst_reuse) == n_series * 2
