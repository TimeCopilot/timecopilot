from typing import TYPE_CHECKING, TypeAlias

import pandas as pd

if TYPE_CHECKING:
    import dask.dataframe as dd
    from pyspark.sql import DataFrame as SparkDataFrame
    from ray.data import Dataset

    DataFrameLike: TypeAlias = pd.DataFrame | SparkDataFrame | Dataset | dd.DataFrame
else:
    DataFrameLike: TypeAlias = pd.DataFrame


def to_pandas(df: DataFrameLike) -> pd.DataFrame:
    """Convert supported dataframe inputs (Spark, Ray, Dask) to pandas.

    Returns the input unchanged if it is already a pandas DataFrame. Conversion
    collects the data into memory and may be expensive. Raises a TypeError for
    unsupported dataframe types.
    """
    if isinstance(df, pd.DataFrame):
        return df

    module = type(df).__module__
    if module.startswith("pyspark"):
        spark_df = _maybe_spark_to_pandas(df)
        if spark_df is not None:
            return spark_df

    if module.startswith("ray"):
        ray_df = _maybe_ray_to_pandas(df)
        if ray_df is not None:
            return ray_df

    if module.startswith("dask"):
        dask_df = _maybe_dask_to_pandas(df)
        if dask_df is not None:
            return dask_df

    raise TypeError(
        "Unsupported dataframe type. Provide a pandas DataFrame or a compatible "
        "Spark, Ray, or Dask dataframe. Install optional dependencies with "
        "`pip install \"timecopilot[dataframes]\"` or `uv sync --group dataframes`."
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
