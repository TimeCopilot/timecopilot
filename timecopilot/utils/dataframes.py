from typing import Any

import pandas as pd


def to_pandas(df: Any) -> pd.DataFrame:
    """Convert supported dataframe inputs to pandas."""
    if isinstance(df, pd.DataFrame):
        return df

    spark_df = _maybe_spark_to_pandas(df)
    if spark_df is not None:
        return spark_df

    ray_df = _maybe_ray_to_pandas(df)
    if ray_df is not None:
        return ray_df

    dask_df = _maybe_dask_to_pandas(df)
    if dask_df is not None:
        return dask_df

    raise TypeError(
        "Unsupported dataframe type. Provide a pandas DataFrame or a compatible "
        "Spark, Ray, or Dask dataframe."
    )


def _maybe_spark_to_pandas(df: Any) -> pd.DataFrame | None:
    try:
        from pyspark.sql import DataFrame as SparkDataFrame
    except ImportError:
        return None
    if isinstance(df, SparkDataFrame):
        return df.toPandas()
    return None


def _maybe_ray_to_pandas(df: Any) -> pd.DataFrame | None:
    try:
        from ray.data import Dataset
    except ImportError:
        return None
    if isinstance(df, Dataset):
        return df.to_pandas()
    return None


def _maybe_dask_to_pandas(df: Any) -> pd.DataFrame | None:
    try:
        import dask.dataframe as dd
    except ImportError:
        return None
    if isinstance(df, dd.DataFrame):
        return df.compute()
    return None
