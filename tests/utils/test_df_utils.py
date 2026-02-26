from importlib import util
from pathlib import Path

import narwhals as nw
import pandas as pd
import pytest


def load_to_pandas():
    module_path = (
        Path(__file__).resolve().parents[2] / "timecopilot" / "utils" / "df_utils.py"
    )
    spec = util.spec_from_file_location("df_utils", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load df_utils module.")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.to_pandas


def load_experiment_handler():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "timecopilot"
        / "utils"
        / "experiment_handler.py"
    )
    spec = util.spec_from_file_location("experiment_handler", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load experiment_handler module.")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_to_pandas_accepts_pandas_dataframe():
    df = pd.DataFrame({"unique_id": ["a"], "ds": ["2024-01-01"], "y": [1.0]})
    to_pandas = load_to_pandas()
    result = to_pandas(df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_to_pandas_accepts_narwhals_dataframe():
    nw_df = nw.from_dict(
        {"unique_id": ["a"], "ds": ["2024-01-01"], "y": [1.0]},
        backend="pandas",
    )
    to_pandas = load_to_pandas()
    result = to_pandas(nw_df)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["unique_id", "ds", "y"]


def test_to_pandas_accepts_polars_dataframe():
    pl = pytest.importorskip("polars")
    pytest.importorskip("pyarrow")

    pl_df = pl.DataFrame({"unique_id": ["a"], "ds": ["2024-01-01"], "y": [1.0]})
    to_pandas = load_to_pandas()
    result = to_pandas(pl_df)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["unique_id", "ds", "y"]


def test_to_pandas_fallback_for_unknown_dataframe():
    fallback = pd.DataFrame({"unique_id": ["fallback"], "ds": ["2024-01-01"], "y": [0.0]})
    to_pandas = load_to_pandas()
    result = to_pandas(object(), fallback=fallback)
    pd.testing.assert_frame_equal(result, fallback)


def test_to_pandas_raises_for_unknown_dataframe():
    to_pandas = load_to_pandas()
    with pytest.raises(TypeError, match="Unsupported dataframe type"):
        to_pandas(object())


def test_validate_df_accepts_polars_dataframe():
    pl = pytest.importorskip("polars")
    pytest.importorskip("pyarrow")

    df = pl.DataFrame({"unique_id": ["a"], "ds": ["2024-01-01"], "y": [1.0]})

    module = load_experiment_handler()
    parsed = module.ExperimentDatasetParser._validate_df(df)
    assert isinstance(parsed, pd.DataFrame)
