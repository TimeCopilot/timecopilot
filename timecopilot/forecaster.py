from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import pandas as pd

from .models.utils.forecaster import Forecaster

if TYPE_CHECKING:
    from dask.dataframe import DataFrame as DaskDataFrame
    from pyspark.sql import DataFrame as SparkDataFrame
    from ray.data import Dataset as RayDataset

# Type variable for any supported DataFrame type
AnyDataFrame = TypeVar(
    "AnyDataFrame",
    pd.DataFrame,
    "DaskDataFrame",
    "SparkDataFrame",
    "RayDataset",
)

# Type variable for distributed DataFrame types only
DistributedDataFrame = TypeVar(
    "DistributedDataFrame",
    "DaskDataFrame",
    "SparkDataFrame",
    "RayDataset",
)


class TimeCopilotForecaster(Forecaster):
    """
    Unified forecaster for multiple time series models.

    This class enables forecasting and cross-validation across a list of models
    from different families (foundational, statistical, machine learning, neural, etc.)
    using a single, consistent interface. It is designed to handle panel (multi-series)
    data and to aggregate results from all models for easy comparison
    and ensemble workflows.

    The unified API ensures that users can call `forecast` or `cross_validation`
    once, passing a list of models, and receive merged results for all models.
    """

    def __init__(
        self,
        models: list[Forecaster],
        fallback_model: Forecaster | None = None,
    ):
        """
        Initialize the TimeCopilotForecaster with a list of models.

        Args:
            models (list[Forecaster]):
                List of instantiated model objects from any supported family
                (foundational, statistical, ML, neural, etc.). Each model must
                implement the `forecast` and `cross_validation` methods with
                compatible signatures.
            fallback_model (Forecaster, optional):
                Model to use as a fallback when a model fails.

        Raises:
            ValueError: If duplicate model aliases are found in the models list.
        """
        self._validate_unique_aliases(models)
        self.models = models
        self.fallback_model = fallback_model

    def _validate_unique_aliases(self, models: list[Forecaster]) -> None:
        """
        Validate that all models have unique aliases.

        Args:
            models (list[Forecaster]): List of model instances to validate.

        Raises:
            ValueError: If duplicate aliases are found.
        """
        aliases = [model.alias for model in models]
        duplicates = set([alias for alias in aliases if aliases.count(alias) > 1])

        if duplicates:
            raise ValueError(
                f"Duplicate model aliases found: {sorted(duplicates)}. "
                f"Each model must have a unique alias to avoid column name conflicts. "
                f"Please provide different aliases when instantiating models of the "
                f"same class."
            )

    @staticmethod
    def _is_distributed_df(df: AnyDataFrame) -> bool:
        """Check if a DataFrame is a distributed DataFrame type.

        Args:
            df: DataFrame to check.

        Returns:
            True if the DataFrame is Spark, Dask, or Ray; False if pandas.
        """
        return not isinstance(df, pd.DataFrame)

    def _call_models(
        self,
        attr: str,
        merge_on: list[str],
        df: pd.DataFrame,
        h: int,
        freq: str | None,
        level: list[int | float] | None,
        quantiles: list[float] | None,
        **kwargs,
    ) -> pd.DataFrame:
        # infer just once to avoid multiple calls to _maybe_infer_freq
        freq = self._maybe_infer_freq(df, freq)
        res_df: pd.DataFrame | None = None
        for model in self.models:
            known_kwargs = {
                "df": df,
                "h": h,
                "freq": freq,
                "level": level,
            }
            if attr != "detect_anomalies":
                known_kwargs["quantiles"] = quantiles
            fn = getattr(model, attr)
            try:
                res_df_model = fn(**known_kwargs, **kwargs)
            except (ValueError, RuntimeError) as e:
                if self.fallback_model is None:
                    raise e
                fn = getattr(self.fallback_model, attr)
                try:
                    res_df_model = fn(**known_kwargs, **kwargs)
                    res_df_model = res_df_model.rename(
                        columns={
                            col: (
                                col.replace(self.fallback_model.alias, model.alias)
                                if col.startswith(self.fallback_model.alias)
                                else col
                            )
                            for col in res_df_model.columns
                        }
                    )
                except (ValueError, RuntimeError) as e:
                    raise e
            if res_df is None:
                res_df = res_df_model
            else:
                if "y" in res_df_model:
                    # drop y to avoid duplicate columns
                    # y was added by the previous condition
                    # to cross validation
                    # (the initial model)
                    res_df_model = res_df_model.drop(columns=["y"])
                res_df = res_df.merge(res_df_model, on=merge_on, how="left")
        return res_df

    def _forecast_pandas(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        """Internal pandas-only forecast implementation.

        This method is called directly for pandas DataFrames or by the
        distributed wrapper for each partition.
        """
        return self._call_models(
            "forecast",
            merge_on=["unique_id", "ds"],
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
        )

    def _distributed_forecast(
        self,
        df: DistributedDataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        num_partitions: int | None = None,
    ) -> DistributedDataFrame:
        """Distributed forecast implementation using Fugue.

        This method handles Spark, Dask, and Ray DataFrames by partitioning
        the data by unique_id and running the pandas forecast on each partition.

        Args:
            df: Distributed DataFrame (Spark, Dask, or Ray).
            h: Forecast horizon.
            freq: Frequency of the time series.
            level: Confidence levels for prediction intervals.
            quantiles: Quantiles to forecast.
            num_partitions: Number of partitions to use.

        Returns:
            Distributed DataFrame with forecast results (same type as input).
        """
        import fugue.api as fa

        from .utils.distributed import _distributed_setup, _forecast_wrapper

        schema, partition_config = _distributed_setup(
            df=df,
            method="forecast",
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            level=level,
            quantiles=quantiles,
            num_partitions=num_partitions,
            models=self.models,
        )

        result_df = fa.transform(
            df,
            using=_forecast_wrapper,
            schema=schema,
            params={
                "forecaster": self,
                "h": h,
                "freq": freq,
                "level": level,
                "quantiles": quantiles,
            },
            partition=partition_config,
            as_fugue=True,
        )

        return fa.get_native_as_df(result_df)

    def forecast(
        self,
        df: AnyDataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        num_partitions: int | None = None,
    ) -> AnyDataFrame:
        """
        Generate forecasts for one or more time series using all models.

        This method produces point forecasts and, optionally, prediction
        intervals or quantile forecasts. The input DataFrame can contain one
        or multiple time series in stacked (long) format.

        Supports pandas, Spark, Dask, and Ray DataFrames. For distributed
        DataFrames, the data is partitioned by unique_id and processed in
        parallel using Fugue.

        Args:
            df (AnyDataFrame):
                DataFrame containing the time series to forecast. Supports
                pandas DataFrame, Spark DataFrame, Dask DataFrame, or Ray Dataset.
                It must include as columns:

                    - "unique_id": an ID column to distinguish multiple series.
                    - "ds": a time column indicating timestamps or periods.
                    - "y": a target column with the observed values.

            h (int):
                Forecast horizon specifying how many future steps to predict.
            freq (str, optional):
                Frequency of the time series (e.g. "D" for daily, "M" for
                monthly). See [Pandas frequency aliases](https://pandas.pydata.org/
                pandas-docs/stable/user_guide/timeseries.html#offset-aliases) for
                valid values. If not provided, the frequency will be inferred
                from the data.
            level (list[int | float], optional):
                Confidence levels for prediction intervals, expressed as
                percentages (e.g. [80, 95]). If provided, the returned
                DataFrame will include lower and upper interval columns for
                each specified level.
            quantiles (list[float], optional):
                List of quantiles to forecast, expressed as floats between 0
                and 1. Should not be used simultaneously with `level`. When
                provided, the output DataFrame will contain additional columns
                named in the format "model-q-{percentile}", where {percentile}
                = 100 × quantile value.
            num_partitions (int, optional):
                Number of partitions to use for distributed DataFrames. Only
                used when df is a Spark, Dask, or Ray DataFrame. If not provided,
                the default partitioning is used.

        Returns:
            AnyDataFrame:
                DataFrame containing forecast results (same type as input).
                Includes:

                    - point forecasts for each timestamp, series and model.
                    - prediction intervals if `level` is specified.
                    - quantile forecasts if `quantiles` is specified.

                For multi-series data, the output retains the same unique
                identifiers as the input DataFrame.
        """
        # Route to distributed implementation for non-pandas DataFrames
        if self._is_distributed_df(df):
            return self._distributed_forecast(
                df=df,
                h=h,
                freq=freq,
                level=level,
                quantiles=quantiles,
                num_partitions=num_partitions,
            )

        # Pandas DataFrame path
        return self._forecast_pandas(
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
        )

    def _cross_validation_pandas(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        n_windows: int = 1,
        step_size: int | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        """Internal pandas-only cross-validation implementation.

        This method is called directly for pandas DataFrames or by the
        distributed wrapper for each partition.
        """
        return self._call_models(
            "cross_validation",
            merge_on=["unique_id", "ds", "cutoff"],
            df=df,
            h=h,
            freq=freq,
            n_windows=n_windows,
            step_size=step_size,
            level=level,
            quantiles=quantiles,
        )

    def _distributed_cross_validation(
        self,
        df: DistributedDataFrame,
        h: int,
        freq: str | None = None,
        n_windows: int = 1,
        step_size: int | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        num_partitions: int | None = None,
    ) -> DistributedDataFrame:
        """Distributed cross-validation implementation using Fugue.

        This method handles Spark, Dask, and Ray DataFrames by partitioning
        the data by unique_id and running the pandas cross-validation
        on each partition.

        Args:
            df: Distributed DataFrame (Spark, Dask, or Ray).
            h: Forecast horizon.
            freq: Frequency of the time series.
            n_windows: Number of cross-validation windows.
            step_size: Step size between windows.
            level: Confidence levels for prediction intervals.
            quantiles: Quantiles to forecast.
            num_partitions: Number of partitions to use.

        Returns:
            Distributed DataFrame with cross-validation results (same type as input).
        """
        import fugue.api as fa

        from .utils.distributed import (
            _cross_validation_wrapper,
            _distributed_setup,
        )

        schema, partition_config = _distributed_setup(
            df=df,
            method="cross_validation",
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            level=level,
            quantiles=quantiles,
            num_partitions=num_partitions,
            models=self.models,
        )

        result_df = fa.transform(
            df,
            using=_cross_validation_wrapper,
            schema=schema,
            params={
                "forecaster": self,
                "h": h,
                "freq": freq,
                "n_windows": n_windows,
                "step_size": step_size,
                "level": level,
                "quantiles": quantiles,
            },
            partition=partition_config,
            as_fugue=True,
        )

        return fa.get_native_as_df(result_df)

    def cross_validation(
        self,
        df: AnyDataFrame,
        h: int,
        freq: str | None = None,
        n_windows: int = 1,
        step_size: int | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        num_partitions: int | None = None,
    ) -> AnyDataFrame:
        """
        This method splits the time series into multiple training and testing
        windows and generates forecasts for each window. It enables evaluating
        forecast accuracy over different historical periods. Supports point
        forecasts and, optionally, prediction intervals or quantile forecasts.

        Supports pandas, Spark, Dask, and Ray DataFrames. For distributed
        DataFrames, the data is partitioned by unique_id and processed in
        parallel using Fugue.

        Args:
            df (AnyDataFrame):
                DataFrame containing the time series to forecast. Supports
                pandas DataFrame, Spark DataFrame, Dask DataFrame, or Ray Dataset.
                It must include as columns:

                    - "unique_id": an ID column to distinguish multiple series.
                    - "ds": a time column indicating timestamps or periods.
                    - "y": a target column with the observed values.

            h (int):
                Forecast horizon specifying how many future steps to predict in
                each window.
            freq (str, optional):
                Frequency of the time series (e.g. "D" for daily, "M" for
                monthly). See [Pandas frequency aliases](https://pandas.pydata.
                org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
                for valid values. If not provided, the frequency will be inferred
                from the data.
            n_windows (int, optional):
                Number of cross-validation windows to generate. Defaults to 1.
            step_size (int, optional):
                Step size between the start of consecutive windows. If None, it
                defaults to `h`.
            level (list[int | float], optional):
                Confidence levels for prediction intervals, expressed as
                percentages (e.g. [80, 95]). When specified, the output
                DataFrame includes lower and upper interval columns for each
                level.
            quantiles (list[float], optional):
                Quantiles to forecast, expressed as floats between 0 and 1.
                Should not be used simultaneously with `level`. If provided,
                additional columns named "model-q-{percentile}" will appear in
                the output, where {percentile} is 100 × quantile value.
            num_partitions (int, optional):
                Number of partitions to use for distributed DataFrames. Only
                used when df is a Spark, Dask, or Ray DataFrame. If not provided,
                the default partitioning is used.

        Returns:
            AnyDataFrame:
                DataFrame containing the forecasts for each cross-validation
                window (same type as input). The output includes:

                    - "unique_id" column to indicate the series.
                    - "ds" column to indicate the timestamp.
                    - "y" column to indicate the target.
                    - "cutoff" column to indicate which window each forecast
                      belongs to.
                    - point forecasts for each timestamp, series and model.
                    - prediction intervals if `level` is specified.
                    - quantile forecasts if `quantiles` is specified.
        """
        # Route to distributed implementation for non-pandas DataFrames
        if self._is_distributed_df(df):
            return self._distributed_cross_validation(
                df=df,
                h=h,
                freq=freq,
                n_windows=n_windows,
                step_size=step_size,
                level=level,
                quantiles=quantiles,
                num_partitions=num_partitions,
            )

        # Pandas DataFrame path
        return self._cross_validation_pandas(
            df=df,
            h=h,
            freq=freq,
            n_windows=n_windows,
            step_size=step_size,
            level=level,
            quantiles=quantiles,
        )

    def _detect_anomalies_pandas(
        self,
        df: pd.DataFrame,
        h: int | None = None,
        freq: str | None = None,
        n_windows: int | None = None,
        level: int | float = 99,
    ) -> pd.DataFrame:
        """Internal pandas-only anomaly detection implementation.

        This method is called directly for pandas DataFrames or by the
        distributed wrapper for each partition.
        """
        return self._call_models(
            "detect_anomalies",
            merge_on=["unique_id", "ds", "cutoff"],
            df=df,
            h=h,  # type: ignore
            freq=freq,
            n_windows=n_windows,
            level=level,  # type: ignore
            quantiles=None,
        )

    def _distributed_detect_anomalies(
        self,
        df: DistributedDataFrame,
        h: int | None = None,
        freq: str | None = None,
        n_windows: int | None = None,
        level: int | float = 99,
        num_partitions: int | None = None,
    ) -> DistributedDataFrame:
        """Distributed anomaly detection implementation using Fugue.

        This method handles Spark, Dask, and Ray DataFrames by partitioning
        the data by unique_id and running the pandas anomaly detection
        on each partition.

        Args:
            df: Distributed DataFrame (Spark, Dask, or Ray).
            h: Forecast horizon.
            freq: Frequency of the time series.
            n_windows: Number of cross-validation windows.
            level: Confidence level for anomaly detection.
            num_partitions: Number of partitions to use.

        Returns:
            Distributed DataFrame with anomaly detection results (same type as input).
        """
        import fugue.api as fa

        from .utils.distributed import (
            _detect_anomalies_wrapper,
            _distributed_setup,
        )

        schema, partition_config = _distributed_setup(
            df=df,
            method="detect_anomalies",
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            level=level,
            quantiles=None,
            num_partitions=num_partitions,
            models=self.models,
        )

        result_df = fa.transform(
            df,
            using=_detect_anomalies_wrapper,
            schema=schema,
            params={
                "forecaster": self,
                "h": h,
                "freq": freq,
                "n_windows": n_windows,
                "level": level,
            },
            partition=partition_config,
            as_fugue=True,
        )

        return fa.get_native_as_df(result_df)

    def detect_anomalies(
        self,
        df: AnyDataFrame,
        h: int | None = None,
        freq: str | None = None,
        n_windows: int | None = None,
        level: int | float = 99,
        num_partitions: int | None = None,
    ) -> AnyDataFrame:
        """
        Detect anomalies in time-series using a cross-validated z-score test.

        This method uses rolling-origin cross-validation to (1) produce
        adjusted (out-of-sample) predictions and (2) estimate the
        standard deviation of forecast errors. It then computes a per-point z-score,
        flags values outside a two-sided prediction interval (with confidence `level`),
        and returns a DataFrame with results.

        Supports pandas, Spark, Dask, and Ray DataFrames. For distributed
        DataFrames, the data is partitioned by unique_id and processed in
        parallel using Fugue.

        Args:
            df (AnyDataFrame):
                DataFrame containing the time series to detect anomalies.
                Supports pandas DataFrame, Spark DataFrame, Dask DataFrame,
                or Ray Dataset.
            h (int, optional):
                Forecast horizon specifying how many future steps to predict.
                In each cross validation window. If not provided, the seasonality
                of the data (inferred from the frequency) is used.
            freq (str, optional):
                Frequency of the time series (e.g. "D" for daily, "M" for
                monthly). See [Pandas frequency aliases](https://pandas.pydata.
                org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
                for valid values. If not provided, the frequency will be inferred
                from the data.
            n_windows (int, optional):
                Number of cross-validation windows to generate.
                If not provided, the maximum number of windows
                (computed by the shortest time series) is used.
                If provided, the number of windows is the minimum
                between the maximum number of windows
                (computed by the shortest time series)
                and the number of windows provided.
            level (int | float):
                Confidence levels for z-score, expressed as
                percentages (e.g. 80, 95). Default is 99.
            num_partitions (int, optional):
                Number of partitions to use for distributed DataFrames. Only
                used when df is a Spark, Dask, or Ray DataFrame. If not provided,
                the default partitioning is used.

        Returns:
            AnyDataFrame:
                DataFrame containing the forecasts for each cross-validation
                window (same type as input). The output includes:

                    - "unique_id" column to indicate the series.
                    - "ds" column to indicate the timestamp.
                    - "y" column to indicate the target.
                    - model column to indicate the model.
                    - lower prediction interval.
                    - upper prediction interval.
                    - anomaly column to indicate if the value is an anomaly.
                        an anomaly is defined as a value that is outside of the
                        prediction interval (True or False).
        """
        # Route to distributed implementation for non-pandas DataFrames
        if self._is_distributed_df(df):
            return self._distributed_detect_anomalies(
                df=df,
                h=h,
                freq=freq,
                n_windows=n_windows,
                level=level,
                num_partitions=num_partitions,
            )

        # Pandas DataFrame path
        return self._detect_anomalies_pandas(
            df=df,
            h=h,
            freq=freq,
            n_windows=n_windows,
            level=level,
        )
