import pytest
from utilsforecast.data import generate_series

from timecopilot.models.benchmarks.stats import ADIDA, AutoARIMA, SeasonalNaive
from timecopilot.models.foundational.timesfm import TimesFM


def test_max_length_parameter_exists():
    """Test that max_length parameter exists in model constructors."""
    # Test ADIDA
    model = ADIDA(max_length=50)
    assert model.max_length == 50
    
    # Test AutoARIMA  
    model = AutoARIMA(max_length=100)
    assert model.max_length == 100
    
    # Test SeasonalNaive
    model = SeasonalNaive(max_length=30)
    assert model.max_length == 30
    
    # Test TimesFM
    model = TimesFM(max_length=20)
    assert model.max_length == 20


def test_max_length_none_by_default():
    """Test that max_length is None by default."""
    model = ADIDA()
    assert model.max_length is None
    
    model = AutoARIMA()
    assert model.max_length is None
    
    model = SeasonalNaive()  
    assert model.max_length is None


def test_truncate_series_functionality():
    """Test the _maybe_truncate_series method."""
    # Generate test data
    df = generate_series(n_series=2, freq="D", min_length=50, max_length=50)
    
    # Test with max_length
    model = SeasonalNaive(max_length=20)
    truncated_df = model._maybe_truncate_series(df)
    
    # Check that each series has at most max_length rows
    for uid in df['unique_id'].unique():
        series_data = truncated_df[truncated_df['unique_id'] == uid]
        assert len(series_data) <= 20
        
    # Test without max_length (should not truncate)
    model_no_limit = SeasonalNaive(max_length=None)
    not_truncated_df = model_no_limit._maybe_truncate_series(df)
    assert len(not_truncated_df) == len(df)


def test_truncate_series_preserves_latest_data():
    """Test that truncation preserves the latest data points."""
    # Generate test data with known values
    df = generate_series(n_series=1, freq="D", min_length=30, max_length=30)
    
    # Sort by date to ensure proper ordering
    df = df.sort_values(['unique_id', 'ds'])
    
    # Test with max_length smaller than series length
    model = SeasonalNaive(max_length=10)
    truncated_df = model._maybe_truncate_series(df)
    
    # The truncated data should have the same latest date as the original
    original_latest = df['ds'].max()
    truncated_latest = truncated_df['ds'].max()
    assert original_latest == truncated_latest
    
    # And should have 10 rows
    assert len(truncated_df) == 10


def test_truncate_series_multiple_series():
    """Test truncation works correctly with multiple series."""
    # Generate multiple series of different lengths
    df1 = generate_series(n_series=1, freq="D", min_length=40, max_length=40)
    df1['unique_id'] = 'series_1'
    
    df2 = generate_series(n_series=1, freq="D", min_length=25, max_length=25)
    df2['unique_id'] = 'series_2'
    
    # Combine series
    import pandas as pd
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Test truncation
    model = SeasonalNaive(max_length=20)
    truncated_df = model._maybe_truncate_series(df)
    
    # Check each series separately
    series1_data = truncated_df[truncated_df['unique_id'] == 'series_1']
    series2_data = truncated_df[truncated_df['unique_id'] == 'series_2']
    
    # Series 1 should be truncated to 20 rows
    assert len(series1_data) == 20
    
    # Series 2 should remain at 25 rows (less than max_length)
    assert len(series2_data) == 25