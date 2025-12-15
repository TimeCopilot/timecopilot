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
