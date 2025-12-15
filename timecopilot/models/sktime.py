from typing import Any

import numpy as np
import pandas as pd

from .utils.forecaster import Forecaster

# from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

# NOTE: SKTime notes
#       https://www.sktime.net/en/stable/examples/01_forecasting.html#
#       https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.base.BaseForecaster.html
# NOTE: forecaster setup args vary, that setup is currently being left to users
# TODO: some forecasters require horizon be provided in fit() call, account for that
# TODO: exogenous data support
# TODO: different alias for different sktime forecasters
#       should this be required?


class SKTimeAdapter(Forecaster):
    """
    Wrapper for SKTime Forecaster models for time series forecasting.


    See the [official documentation](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.base.BaseForecaster.html)
    for more details.
    """

    def __init__(
        self,
        model,
        # model: BaseForecaster,
        alias: str = "SKTimeAdapter",
        *args: Any,
        **kwargs: Any,
    ):
        """
        Args:
            model (sktime.forecasting.base.BaseForecaster): sktime forecasting model
            alias (str, optional): Custom name for the model instance.
                Default is "SKTimeAdapter".
            *args: Additional positional arguments passed to SKTimeAdapter.
            **kwargs: Additional keyword arguments passed to SKTimeAdapter.
        """
        super().__init__(*args, **kwargs)
        self.alias = alias
        self.model = model

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
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
        """
        # TODO: support for exogenous data
        # TODO: add support for level for sktime models that can support it
        # TODO: add support for quantiles for sktime models that can support it
        if level is not None:
            raise ValueError(
                "Level and quantiles are not supported for adapted sktime models yet."
            )
        # NOTE: may not be needed
        freq = self._maybe_infer_freq(df, freq)
        forecast_horizon = np.arange(1, 1 + h)
        id_col = "unique_id"
        datetime_col = "ds"
        y_col = "y"
        df = df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index([id_col, datetime_col])

        # some sktime models require fh be passed in fit()
        model = self.model
        model.fit(y=df, fh=forecast_horizon)
        # fh doesn't need to be passed in predict because it is being passed in fit
        # if quantiles is not None:
        #     print("sktime quantile pred")
        #     fcst_df = model.predict_quantiles(
        #         alpha=qc.quantiles
        #     )
        #     fcst_df = fcst_df.reset_index()
        #     return fcst_df
        fcst_df = model.predict()
        fcst_df = fcst_df.reset_index()
        # fcst_df = qc.maybe_convert_quantiles_to_level(fcst_df, models=[self.alias])
        fcst_df.rename(columns={y_col: self.alias}, inplace=True)
        return fcst_df
