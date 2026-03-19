import pandas as pd

from .models.utils.forecaster import Forecaster


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
                            col: col.replace(self.fallback_model.alias, model.alias)
                            if col.startswith(self.fallback_model.alias)
                            else col
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

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Generate forecasts for one or more time series using all models.

        This method produces point forecasts and, optionally, prediction
        intervals or quantile forecasts. The input DataFrame can contain one
        or multiple time series in stacked (long) format.

        Args:
            df (pd.DataFrame):
                DataFrame containing the time series to forecast. It must
                include as columns:

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

        Returns:
            pd.DataFrame:
                DataFrame containing forecast results. Includes:

                    - point forecasts for each timestamp, series and model.
                    - prediction intervals if `level` is specified.
                    - quantile forecasts if `quantiles` is specified.

                For multi-series data, the output retains the same unique
                identifiers as the input DataFrame.
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

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        n_windows: int = 1,
        step_size: int | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        This method splits the time series into multiple training and testing
        windows and generates forecasts for each window. It enables evaluating
        forecast accuracy over different historical periods. Supports point
        forecasts and, optionally, prediction intervals or quantile forecasts.

        Args:
            df (pd.DataFrame):
                DataFrame containing the time series to forecast. It must
                include as columns:

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

        Returns:
            pd.DataFrame:
                DataFrame containing the forecasts for each cross-validation
                window. The output includes:

                    - "unique_id" column to indicate the series.
                    - "ds" column to indicate the timestamp.
                    - "y" column to indicate the target.
                    - "cutoff" column to indicate which window each forecast
                      belongs to.
                    - point forecasts for each timestamp, series and model.
                    - prediction intervals if `level` is specified.
                    - quantile forecasts if `quantiles` is specified.
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

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        h: int | None = None,
        freq: str | None = None,
        n_windows: int | None = None,
        level: int | float = 99,
    ) -> pd.DataFrame:
        """
        Detect anomalies in time-series using a cross-validated z-score test.

        This method uses rolling-origin cross-validation to (1) produce
        adjusted (out-of-sample) predictions and (2) estimate the
        standard deviation of forecast errors. It then computes a per-point z-score,
        flags values outside a two-sided prediction interval (with confidence `level`),
        and returns a DataFrame with results.

        Args:
            df (pd.DataFrame):
                DataFrame containing the time series to detect anomalies.
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

        Returns:
            pd.DataFrame:
                DataFrame containing the forecasts for each cross-validation
                window. The output includes:

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
