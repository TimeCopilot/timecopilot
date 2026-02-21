import pandas as pd
import pytest

from timecopilot.utils.dataframes import to_pandas


def _make_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {
            "unique_id": ["series_1"] * len(dates),
            "ds": dates,
            "y": [1.0, 2.0, 3.0],
        }
    )


def test_to_pandas_with_pandas():
    df = _make_frame()
    assert to_pandas(df) is df


def test_to_pandas_with_dask():
    dd = pytest.importorskip("dask.dataframe")
    df = _make_frame()
    dask_df = dd.from_pandas(df, npartitions=1)
    out = to_pandas(dask_df)
    pd.testing.assert_frame_equal(out, df)


def test_to_pandas_with_ray():
    ray = pytest.importorskip("ray")
    ray_data = pytest.importorskip("ray.data")
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True)
    try:
        df = _make_frame()
        ray_df = ray_data.from_pandas(df)
        out = to_pandas(ray_df)
        pd.testing.assert_frame_equal(out.sort_values("ds").reset_index(drop=True), df)
    finally:
        ray.shutdown()


def test_to_pandas_with_spark():
    pyspark_sql = pytest.importorskip("pyspark.sql")
    spark = pyspark_sql.SparkSession.builder.master("local[1]").appName(
        "timecopilot-tests"
    ).getOrCreate()
    try:
        df = _make_frame()
        spark_df = spark.createDataFrame(df)
        out = to_pandas(spark_df)
        pd.testing.assert_frame_equal(out.sort_values("ds").reset_index(drop=True), df)
    finally:
        spark.stop()


def test_to_pandas_unsupported_type():
    with pytest.raises(TypeError, match="Unsupported dataframe type"):
        to_pandas(["not", "a", "dataframe"])
