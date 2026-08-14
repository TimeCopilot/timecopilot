from copy import deepcopy
from typing import Any

import pandas as pd
from prophet import Prophet as ProphetBase
from threadpoolctl import threadpool_limits

from .utils.forecaster import QuantileConverter
from .utils.parallel_forecaster import ParallelForecaster


class Prophet(ProphetBase, ParallelForecaster):
    """
    Wrapper for Facebook Prophet for time series forecasting.

    Prophet is a procedure for forecasting time series data based on an additive model
    where non-linear trends are fit with yearly, weekly, and daily seasonality, plus
    holiday effects. It works best with time series that have strong seasonal effects
    and several seasons of historical data. Prophet is robust to missing data and
    shifts in the trend, and typically handles outliers well.

    See the [official documentation](https://github.com/facebook/prophet)
    for more details.
    """

    def __init__(
        self,
        alias: str = "Prophet",
        *args: Any,
        **kwargs: Any,
    ):
        """
        Args:
            alias (str, optional): Custom name for the model instance.
                Default is "Prophet".
            *args: Additional positional arguments passed to ProphetBase.
            **kwargs: Additional keyword arguments passed to ProphetBase.

        Raises:
            ValueError: If 'interval_width' is provided in kwargs. Use 'level' or
                'quantiles' instead when using 'forecast' or 'cross_validation'.
        """
        super().__init__(*args, **kwargs)
        self.alias = alias
        if "interval_width" in kwargs:
            raise ValueError(
                "interval_width is not supported, "
                "use `level` or `quantiles` instead when using "
                "`forecast` or `cross_validation`"
            )

    def predict_uncertainty(
        self,
        df: pd.DataFrame,
        vectorized: bool,
        quantiles: list[float],
    ) -> pd.DataFrame:
        # adapted from https://github.com/facebook/prophet/blob/e64606036325bfb225333ef0991e41bdfb66f7c1/python/prophet/forecaster.py#L1431-L1455
        sim_values = self.sample_posterior_predictive(df, vectorized)
        series = {}
        for q in quantiles:
            series[f"yhat-q-{int(q * 100)}"] = self.percentile(
                sim_values["yhat"], 100 * q, axis=1
            )
        return pd.DataFrame(series)

    def predict(
        self,
        df: pd.DataFrame = None,
        vectorized: bool = True,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        # Predict using the prophet model.
        # adapted from https://github.com/facebook/prophet/blob/e64606036325bfb225333ef0991e41bdfb66f7c1/python/prophet/forecaster.py#L1249C2-L1295C1
        # to allow for quantiles
        if self.history is None:
            raise Exception("Model has not been fit.")
        if df is None:
            df = self.history.copy()
        else:
            if df.shape[0] == 0:
                raise ValueError("Dataframe has no rows.")
            df = self.setup_dataframe(df.copy())
        df["trend"] = self.predict_trend(df)
        seasonal_components = self.predict_seasonal_components(df)
        if self.uncertainty_samples and quantiles is not None:
            intervals = self.predict_uncertainty(df, vectorized, quantiles)
        else:
            intervals = None
        # Drop columns except ds, cap, floor, and trend
        cols = ["ds", "trend"]
        if "cap" in df:
            cols.append("cap")
        if self.logistic_floor:
            cols.append("floor")
        # Add in forecast components
        df2 = pd.concat((df[cols], intervals, seasonal_components), axis=1)
        df2["yhat"] = (
            df2["trend"] * (1 + df2["multiplicative_terms"]) + df2["additive_terms"]
        )
        df2.columns = [col.replace("yhat", self.alias) for col in df2.columns]
        cols = [col for col in df2.columns if self.alias in col or "ds" in col]
        return df2[cols]

    def _local_forecast_impl(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        qc = QuantileConverter(level=level, quantiles=quantiles)
        model = deepcopy(self)
        model.fit(df=df)
        future_df = model.make_future_dataframe(
            periods=h,
            include_history=False,
            freq=freq,
        )
        fcst_df = model.predict(future_df, quantiles=qc.quantiles)
        fcst_df = qc.maybe_convert_quantiles_to_level(fcst_df, models=[self.alias])
        return fcst_df

    def _anomaly_min_series_length(self, h: int) -> int:
        return h + 2

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
