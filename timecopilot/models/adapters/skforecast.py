from copy import deepcopy
from typing import Any

import pandas as pd
from threadpoolctl import threadpool_limits

from ..utils.parallel_forecaster import ParallelForecaster

# TODO: exogenous data support
# NOTE: skforecaster baseforecaster class:
#           skforecast.base._forecaster_base.ForecasterBase


class SKForecastAdapter(ParallelForecaster):
    def __init__(
        self,
        model,
        alias: str | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Args:
            model (sktime.forecasting.base.BaseForecaster): sktime forecasting model
            alias (str, optional): Custom name for the model instance.
                By default alias is retrieved from the type name of model.
            *args: Additional positional arguments passed to SKTimeAdapter.
            **kwargs: Additional keyword arguments passed to SKTimeAdapter.
        """
        super().__init__(*args, **kwargs)
        self.alias = alias if alias is not None else type(model).__name__
        self.model = model

    def _local_forecast_impl(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        # qc = QuantileConverter(level=level, quantiles=quantiles)
        model = deepcopy(self.model)
        y_col = "y"
        time_col = "ds"
        series = df.loc[:, y_col]
        series.index = df[time_col]
        series = series.asfreq(freq)
        model.fit(series)
        fcst_series = model.predict(h)
        fcst_df = fcst_series.reset_index()
        pred_col = "pred"
        fcst_df.rename(
            {"index": time_col, pred_col: self.alias}, axis="columns", inplace=True
        )
        return fcst_df

    def _local_forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        with threadpool_limits(limits=1):
            return self._local_forecast_impl(
                df=df,
                h=h,
                freq=freq,
                level=level,
                quantiles=quantiles,
            )

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        # fmt: off
        """
        Generate forecasts for time series data using an sktime model.

        This method produces point forecasts and, optionally, prediction
        intervals or quantile forecasts. The input DataFrame can contain one
        or multiple time series in stacked (long) format.

        Prediction intervals and quantile forecasts are not currently supported
        with sktime based models

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

                    - point forecasts for each timestamp and series.
                    - prediction intervals if `level` is specified.
                    - quantile forecasts if `quantiles` is specified.

                For multi-series data, the output retains the same unique
                identifiers as the input DataFrame.

        Example:
            ```python
            from lightgbm import LGBMRegressor
            import pandas as pd
            from timecopilot import TimeCopilot
            from timecopilot.models.adapters.skforecast import SKForecastAdapter
            from skforecast.recursive import ForecasterRecursive
            from skforecast.preprocessing import RollingFeatures

            forecaster = ForecasterRecursive(
                 estimator       = LGBMRegressor(random_state=123, verbose=-1),
                 lags            = 10,
                 window_features = RollingFeatures(stats=['mean'], window_sizes=10)
             )

            df = pd.read_csv("https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv")
            adapted_skf_model = SKForecastAdapter(forecaster)
            tc = TimeCopilot(llm="openai:gpt-4o", forecasters=[adapted_skf_model])
            result = tc.forecast(df, h=12, freq="MS")
            print(result.output)
            ```
        """
        # fmt: on
        if level is not None or quantiles is not None:
            raise ValueError(
                "Level and quantiles are not supported for adapted skforecast"
                " models yet."
            )
        if self.model.get_tags()["forecasting_scope"].startswith("single-series"):
            return super().forecast(df, h, freq=freq, level=level, quantiles=quantiles)
        freq = self._maybe_infer_freq(df, freq)
        # importing in the function since skforecast isn't a required
        # dependency but should be present when using the skforecast adapter
        # there is also an exogenous data conversion
        from skforecast.preprocessing import reshape_series_long_to_dict

        id_col = "unique_id"
        date_col = "ds"
        y_col = "y"
        df_dict = reshape_series_long_to_dict(
            df,
            freq=freq,
            series_id=id_col,
            index=date_col,
            values=y_col,
        )
        #
        self.model.fit(df_dict)
        fcst_df: pd.DataFrame = self.model.predict(h)
        pred_col = "pred"
        fcst_df.reset_index(inplace=True)
        fcst_df.rename(
            columns={"level": id_col, "index": date_col, pred_col: self.alias},
            inplace=True,
        )
        fcst_df.sort_values([id_col, date_col], inplace=True)
        fcst_df.reindex(columns=[id_col, date_col, self.alias])
        return fcst_df
