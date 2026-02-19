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


def test_chronos_model_uses_float32(mocker):
    """Ensure Chronos loads models with float32 dtype."""
    mock_pipeline = mocker.patch(
        "timecopilot.models.foundation.chronos.BaseChronosPipeline.from_pretrained"
    )
    mocker.patch("torch.cuda.is_available", return_value=False)

    from timecopilot.models.foundation.chronos import Chronos

    model = Chronos(repo_id="amazon/chronos-t5-tiny")

    with model._get_model():
        pass

    mock_pipeline.assert_called_once()
    call_kwargs = mock_pipeline.call_args[1]
    assert call_kwargs["torch_dtype"] == torch.float32
