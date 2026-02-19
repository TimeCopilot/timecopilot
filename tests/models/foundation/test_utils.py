import torch

from timecopilot.models.foundation.utils import TimeSeriesDataset


def test_timeseries_dataset_default_dtype_is_bfloat16():
    """Ensure TimeSeriesDataset defaults to bfloat16 for backward compatibility."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2020-01-01", periods=10),
            "y": range(10),
        }
    )
    dataset = TimeSeriesDataset.from_df(df, batch_size=10)
    assert dataset.data[0].dtype == torch.bfloat16


def test_timeseries_dataset_respects_custom_dtype():
    """Ensure TimeSeriesDataset respects custom dtype parameter."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2020-01-01", periods=10),
            "y": range(10),
        }
    )
    dataset = TimeSeriesDataset.from_df(df, batch_size=10, dtype=torch.float32)
    assert dataset.data[0].dtype == torch.float32
