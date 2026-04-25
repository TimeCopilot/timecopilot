import os

import pandas as pd
from mlforecast.auto import AutoLightGBM, AutoMLForecast
from mlforecast.utils import PredictionIntervals

from .utils.forecaster import Forecaster, QuantileConverter, get_seasonality

os.environ["NIXTLA_ID_AS_COL"] = "true"


class AutoLGBM(Forecaster):
    """AutoLGBM forecaster using AutoMLForecast with LightGBM.

    Notes:
        - Level and quantiles are not supported for AutoLGBM yet. Please open
            an issue if you need this feature.
        - AutoLGBM requires a minimum length for some frequencies.
    """

    def __init__(
        self,
        alias: str = "AutoLGBM",
        num_samples: int = 10,
        cv_n_windows: int = 5,
    ):
        self.alias = alias
        self.num_samples = num_samples
        self.cv_n_windows = cv_n_windows

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts for time series data using the model.

        This method produces point forecasts and, optionally, quantile
        forecasts. The input DataFrame can contain one or multiple time series
        in stacked (long) format.

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
                Not supported for AutoLGBM. Use `quantiles` instead.
            quantiles (list[float], optional):
                List of quantiles to forecast, expressed as floats between 0
                and 1. Should not be used simultaneously with `level`. When
                provided, the output DataFrame will contain additional columns
                named in the format "model-q-{percentile}", where {percentile}
                = 100 × quantile value. Prediction intervals are computed via
                conformal prediction using cross-validation residuals.

        Returns:
            pd.DataFrame:
                DataFrame containing forecast results. Includes:

                    - point forecasts for each timestamp and series.
                    - quantile forecasts if `quantiles` is specified.

                For multi-series data, the output retains the same unique
                identifiers as the input DataFrame.
        """
        if level is not None and quantiles is not None:
            raise ValueError(
                "You must not provide both `level` and `quantiles` simultaneously."
            )
        if level is not None:
            raise ValueError(
                "Level is not supported for AutoLGBM. Please use `quantiles` instead."
            )

        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=None, quantiles=quantiles)
        mf = AutoMLForecast(
            models=[AutoLightGBM()],
            freq=freq,
            season_length=get_seasonality(freq),
            num_threads=-1,
        )
        prediction_intervals = (
            PredictionIntervals(n_windows=self.cv_n_windows)
            if qc.level is not None
            else None
        )
        mf.fit(
            df=df,
            n_windows=self.cv_n_windows,
            h=h,
            num_samples=self.num_samples,
            prediction_intervals=prediction_intervals,
        )
        fcst_df = mf.predict(h=h, level=qc.level)
        fcst_df.columns = [
            c.replace("AutoLightGBM", self.alias) for c in fcst_df.columns
        ]
        fcst_df = qc.maybe_convert_level_to_quantiles(fcst_df, [self.alias])
        return fcst_df
