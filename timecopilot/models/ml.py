import os

import pandas as pd
from mlforecast.auto import AutoLightGBM, AutoMLForecast

from .utils.forecaster import Forecaster, get_seasonality
from ..utils.df_utils import to_pandas

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

                    - point forecasts for each timestamp and series.
                    - prediction intervals if `level` is specified.
                    - quantile forecasts if `quantiles` is specified.

                For multi-series data, the output retains the same unique
                identifiers as the input DataFrame.
        """
        if level is not None or quantiles is not None:
            raise ValueError("Level and quantiles are not supported for AutoLGBM yet.")

        df = to_pandas(df)
        freq = self._maybe_infer_freq(df, freq)
        mf = AutoMLForecast(
            models=[AutoLightGBM()],
            freq=freq,
            season_length=get_seasonality(freq),
            num_threads=-1,
        )
        mf.fit(
            df=df,
            n_windows=self.cv_n_windows,
            h=h,
            num_samples=self.num_samples,
        )
        fcst_df = mf.predict(h=h)
        fcst_df = fcst_df.rename(columns={"AutoLightGBM": self.alias})
        return fcst_df
