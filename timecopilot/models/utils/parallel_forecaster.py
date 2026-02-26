import os
from collections.abc import Callable
from multiprocessing import Pool

import pandas as pd

from .forecaster import Forecaster
from ..utils.df_utils import to_pandas


class ParallelForecaster(Forecaster):
    def _process_group(
        self,
        df: pd.DataFrame,
        func: Callable,
        **kwargs,
    ) -> pd.DataFrame:
        uid = df["unique_id"].iloc[0]
        _df = df.drop("unique_id", axis=1)
        res_df = func(_df, **kwargs)
        res_df.insert(0, "unique_id", uid)
        return res_df

    def _apply_parallel(
        self,
        df_grouped: pd.DataFrame,
        func: Callable,
        **kwargs,
    ) -> pd.DataFrame:
        with Pool(max(1, (os.cpu_count() or 1) - 1)) as executor:
            futures = [
                executor.apply_async(
                    self._process_group,
                    args=(df, func),
                    kwds=kwargs,
                )
                for _, df in df_grouped
            ]
            results = [future.get() for future in futures]
        return pd.concat(results)

    def _local_forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

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
        df = to_pandas(df)
        freq = self._maybe_infer_freq(df, freq)
        fcst_df = self._apply_parallel(
            df.groupby("unique_id"),
            self._local_forecast,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df
