"""Distributed DataFrame utilities for TimeCopilot using Fugue."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ..models.utils.forecaster import Forecaster


def _register_fugue_backends() -> None:
    """Register Fugue backends for distributed DataFrames.

    This ensures that Fugue can recognize and handle Dask, Spark,
    and Ray DataFrames properly.
    """
    try:
        import fugue_dask  # noqa: F401
    except ImportError:
        pass

    try:
        import fugue_spark  # noqa: F401
    except ImportError:
        pass

    try:
        import fugue_ray  # noqa: F401
    except ImportError:
        pass


def _get_schema(
    df: Any,
    method: str,
    id_col: str,
    time_col: str,
    target_col: str,
    level: list[int | float] | None,
    quantiles: list[float] | None,
    models: list[Forecaster],
) -> "triad.Schema":
    """Build the output schema for distributed operations.

    Args:
        df: Input DataFrame (any distributed type).
        method: The method being called ("forecast", "cross_validation", "detect_anomalies").
        id_col: Name of the ID column.
        time_col: Name of the time column.
        target_col: Name of the target column.
        level: Confidence levels for prediction intervals.
        quantiles: Quantiles to forecast.
        models: List of forecaster models.

    Returns:
        Schema object for the output DataFrame.
    """
    _register_fugue_backends()
    import fugue.api as fa

    # Base columns depend on the method
    base_cols = [id_col, time_col]
    if method != "forecast":
        base_cols.append(target_col)

    # Extract base schema from input DataFrame
    schema = fa.get_schema(df).extract(base_cols).copy()

    # Add model columns
    for model in models:
        schema.append(f"{model.alias}:double")

    # Add method-specific columns
    if method == "detect_anomalies":
        # Add cutoff column with same type as time_col
        schema.append(("cutoff", schema[time_col].type))
        # Add anomaly detection columns for each model
        for model in models:
            schema.append(f"{model.alias}-lo-{int(level)}:double")
            schema.append(f"{model.alias}-hi-{int(level)}:double")
            schema.append(f"{model.alias}-anomaly:bool")
    elif method == "cross_validation":
        # Add cutoff column with same type as time_col
        schema.append(("cutoff", schema[time_col].type))

    # Add prediction interval columns if level is provided
    if level is not None and method != "detect_anomalies":
        if not isinstance(level, list):
            level = [level]
        level = sorted(level)
        for model in models:
            for lv in reversed(level):
                schema.append(f"{model.alias}-lo-{lv}:double")
            for lv in level:
                schema.append(f"{model.alias}-hi-{lv}:double")

    # Add quantile columns if quantiles are provided
    if quantiles is not None:
        quantiles = sorted(quantiles)
        for model in models:
            for q in quantiles:
                schema.append(f"{model.alias}-q-{int(q * 100)}:double")

    return schema


def _is_supported_distributed_df(df: Any) -> bool:
    """Check if the DataFrame is a supported distributed type.

    Args:
        df: DataFrame to check.

    Returns:
        True if supported (Spark, Dask, or Ray), False otherwise.
    """
    df_module = type(df).__module__
    df_name = type(df).__name__

    # Check for Dask DataFrame (both old and new dask-expr backend)
    if "dask.dataframe" in df_module or "dask_expr" in df_module:
        return True

    # Check for Spark DataFrame
    if "pyspark.sql" in df_module and "DataFrame" in df_name:
        return True

    # Check for Ray Dataset
    if "ray.data" in df_module:
        return True

    # Try Fugue's inference as fallback
    try:
        from fugue.execution import infer_execution_engine

        return infer_execution_engine([df]) is not None
    except Exception:
        return False


def _distributed_setup(
    df: Any,
    method: str,
    id_col: str,
    time_col: str,
    target_col: str,
    level: list[int | float] | None,
    quantiles: list[float] | None,
    num_partitions: int | None,
    models: list[Forecaster],
) -> tuple[Any, dict[str, Any]]:
    """Set up schema and partition configuration for distributed operations.

    Args:
        df: Input DataFrame (any distributed type).
        method: The method being called.
        id_col: Name of the ID column.
        time_col: Name of the time column.
        target_col: Name of the target column.
        level: Confidence levels for prediction intervals.
        quantiles: Quantiles to forecast.
        num_partitions: Number of partitions to use.
        models: List of forecaster models.

    Returns:
        Tuple of (schema, partition_config).

    Raises:
        ValueError: If execution engine cannot be inferred from DataFrame type.
    """
    if not _is_supported_distributed_df(df):
        raise ValueError(
            f"Could not infer execution engine for type {type(df).__name__}. "
            "Expected a Spark or Dask DataFrame or a Ray Dataset."
        )

    # Build output schema based on method
    schema = _get_schema(
        df=df,
        method=method,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        level=level,
        quantiles=quantiles,
        models=models,
    )

    # Configure partitioning: by id_col with coarse algorithm
    # "coarse" means series with the same unique_id are grouped together
    partition_config: dict[str, Any] = {"by": id_col, "algo": "coarse"}
    if num_partitions is not None:
        partition_config["num"] = num_partitions

    return schema, partition_config


def _forecast_wrapper(
    df: pd.DataFrame,
    forecaster: Any,  # TimeCopilotForecaster - using Any to avoid import issues
    h: int,
    freq: str | None,
    level: list[int | float] | None,
    quantiles: list[float] | None,
) -> pd.DataFrame:
    """Wrapper function for distributed forecast.

    This function is called by Fugue's transform for each partition.
    It receives a pandas DataFrame partition and calls the regular
    forecast method.

    Args:
        df: Pandas DataFrame partition (grouped by unique_id).
        forecaster: The TimeCopilotForecaster instance.
        h: Forecast horizon.
        freq: Frequency of the time series.
        level: Confidence levels for prediction intervals.
        quantiles: Quantiles to forecast.

    Returns:
        Pandas DataFrame with forecast results.
    """
    return forecaster._forecast_pandas(
        df=df,
        h=h,
        freq=freq,
        level=level,
        quantiles=quantiles,
    )


def _cross_validation_wrapper(
    df: pd.DataFrame,
    forecaster: Any,  # TimeCopilotForecaster - using Any to avoid import issues
    h: int,
    freq: str | None,
    n_windows: int,
    step_size: int | None,
    level: list[int | float] | None,
    quantiles: list[float] | None,
) -> pd.DataFrame:
    """Wrapper function for distributed cross-validation.

    This function is called by Fugue's transform for each partition.
    It receives a pandas DataFrame partition and calls the regular
    cross_validation method.

    Args:
        df: Pandas DataFrame partition (grouped by unique_id).
        forecaster: The TimeCopilotForecaster instance.
        h: Forecast horizon.
        freq: Frequency of the time series.
        n_windows: Number of cross-validation windows.
        step_size: Step size between windows.
        level: Confidence levels for prediction intervals.
        quantiles: Quantiles to forecast.

    Returns:
        Pandas DataFrame with cross-validation results.
    """
    return forecaster._cross_validation_pandas(
        df=df,
        h=h,
        freq=freq,
        n_windows=n_windows,
        step_size=step_size,
        level=level,
        quantiles=quantiles,
    )


def _detect_anomalies_wrapper(
    df: pd.DataFrame,
    forecaster: Any,  # TimeCopilotForecaster - using Any to avoid import issues
    h: int | None,
    freq: str | None,
    n_windows: int | None,
    level: int | float,
) -> pd.DataFrame:
    """Wrapper function for distributed anomaly detection.

    This function is called by Fugue's transform for each partition.
    It receives a pandas DataFrame partition and calls the regular
    detect_anomalies method.

    Args:
        df: Pandas DataFrame partition (grouped by unique_id).
        forecaster: The TimeCopilotForecaster instance.
        h: Forecast horizon.
        freq: Frequency of the time series.
        n_windows: Number of cross-validation windows.
        level: Confidence level for anomaly detection.

    Returns:
        Pandas DataFrame with anomaly detection results.
    """
    return forecaster._detect_anomalies_pandas(
        df=df,
        h=h,
        freq=freq,
        n_windows=n_windows,
        level=level,
    )
