import pandas as pd
import utilsforecast.processing as ufp
from gluonts.time_feature.seasonality import (
    DEFAULT_SEASONALITIES,
)
from gluonts.time_feature.seasonality import (
    get_seasonality as _get_seasonality,
)
from tqdm import tqdm
from utilsforecast.processing import (
    backtest_splits,
    drop_index_if_pandas,
    join,
    maybe_compute_sort_indices,
    take_rows,
    vertical_concat,
)


def get_seasonality(freq: str) -> int:
    return _get_seasonality(freq, seasonalities=DEFAULT_SEASONALITIES | {"D": 7})


def maybe_infer_freq(df: pd.DataFrame, freq: str | None) -> str:
    """
    Infer the frequency of the time series data.

    Args:
        df (pd.DataFrame): The time series data.
        freq (str | None): The frequency of the time series data. If None,
            the frequency will be inferred from the data.

    Returns:
        str: The inferred frequency of the time series data.
    """
    # based on https://github.com/Nixtla/nixtla/blob/bf67c76fd473a61c72b1f54725ffbcb51a3048c5/nixtla/nixtla_client.py#L208C1-L235C25
    if freq is not None:
        return freq
    sizes = df["unique_id"].value_counts(sort=True)
    times = df.loc[df["unique_id"] == sizes.index[0], "ds"].sort_values()
    if times.dt.tz is not None:
        times = times.dt.tz_convert("UTC").dt.tz_localize(None)
    inferred_freq = pd.infer_freq(times.values)
    if inferred_freq is None:
        raise RuntimeError(
            "Could not infer the frequency of the time column. This could be due "
            "to inconsistent intervals. Please check your data for missing, "
            "duplicated or irregular timestamps"
        )
    return inferred_freq


def maybe_convert_col_to_datetime(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df[col_name]):
        df = df.copy()
        df[col_name] = pd.to_datetime(df[col_name])
    return df


class Forecaster:
    alias: str

    def __init__(self, max_length: int | None = None):
        """
        Args:
            max_length (int, optional): Maximum number of observations to use from the
                end of each time series for training and inference. If None, all
                observations are used. This can significantly improve inference times
                for long time series by reducing the amount of data processed.
        """
        self.max_length = max_length

    @staticmethod
    def _maybe_infer_freq(
        df: pd.DataFrame,
        freq: str | None,
    ) -> str:
        return maybe_infer_freq(df, freq)

    def _maybe_get_seasonality(self, freq: str) -> int:
        if hasattr(self, "season_length"):
            if self.season_length is not None:
                return self.season_length
            else:
                return get_seasonality(freq)
        else:
            return get_seasonality(freq)

    def _maybe_truncate_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Truncate each time series to the last max_length observations if max_length is set.
        
        Args:
            df: DataFrame with columns 'unique_id', 'ds', 'y'
            
        Returns:
            DataFrame with truncated series if max_length is set, otherwise original DataFrame
        """
        if self.max_length is None:
            return df
        
        # Sort by unique_id and ds to ensure proper ordering
        df = df.sort_values(['unique_id', 'ds'])
        
        # Take the last max_length rows for each series
        truncated_dfs = []
        for uid in df['unique_id'].unique():
            series_df = df[df['unique_id'] == uid]
            if len(series_df) > self.max_length:
                series_df = series_df.tail(self.max_length)
            truncated_dfs.append(series_df)
        
        return pd.concat(truncated_dfs, ignore_index=True)

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
        raise NotImplementedError("This method must be implemented in a subclass.")

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
        Perform cross-validation on time series data.

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
                    - point forecasts for each timestamp and series.
                    - prediction intervals if `level` is specified.
                    - quantile forecasts if `quantiles` is specified.
        """
        freq = self._maybe_infer_freq(df, freq)
        df = maybe_convert_col_to_datetime(df, "ds")
        df = self._maybe_truncate_series(df)
        # mlforecast cv code
        results = []
        sort_idxs = maybe_compute_sort_indices(df, "unique_id", "ds")
        if sort_idxs is not None:
            df = take_rows(df, sort_idxs)
        splits = backtest_splits(
            df,
            n_windows=n_windows,
            h=h,
            id_col="unique_id",
            time_col="ds",
            freq=pd.tseries.frequencies.to_offset(freq),
            step_size=h if step_size is None else step_size,
        )
        for _, (cutoffs, train, valid) in tqdm(enumerate(splits)):
            if len(valid.columns) > 3:
                raise NotImplementedError(
                    "Cross validation with exogenous variables is not yet supported."
                )
            y_pred = self.forecast(
                df=train,
                h=h,
                freq=freq,
                level=level,
                quantiles=quantiles,
            )
            y_pred = join(y_pred, cutoffs, on="unique_id", how="left")
            result = join(
                valid[["unique_id", "ds", "y"]],
                y_pred,
                on=["unique_id", "ds"],
            )
            if result.shape[0] < valid.shape[0]:
                raise ValueError(
                    "Cross validation result produced less results than expected. "
                    "Please verify that the frequency parameter (freq) "
                    "matches your series' "
                    "and that there aren't any missing periods."
                )
            results.append(result)
        out = vertical_concat(results)
        out = drop_index_if_pandas(out)
        first_out_cols = ["unique_id", "ds", "cutoff", "y"]
        remaining_cols = [c for c in out.columns if c not in first_out_cols]
        fcst_cv_df = out[first_out_cols + remaining_cols]
        return fcst_cv_df


class QuantileConverter:
    """Handles inputs and outputs for probabilistic forecasts."""

    def __init__(
        self,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ):
        level, quantiles, level_was_provided = self._prepare_level_and_quantiles(
            level, quantiles
        )
        self.level = level
        self.quantiles = quantiles
        # this is used to determine whether to return the level or the quantiles
        self.level_was_provided = level_was_provided

    @staticmethod
    def _prepare_level_and_quantiles(
        level: list[int | float] | None,
        quantiles: list[float] | None,
    ) -> tuple[list[int | float] | None, list[float] | None, bool]:
        # based on https://github.com/Nixtla/nixtla/blob/e74d98d9346a055153f84801cac94715c2342946/nixtla/nixtla_client.py#L444
        if level is not None and quantiles is not None:
            raise ValueError(
                "You must not provide both `level` and `quantiles` simultaneously."
            )
        if quantiles is None and level is not None:
            _quantiles = []
            for lv in level:
                q_lo, q_hi = QuantileConverter._level_to_quantiles(lv)
                _quantiles.append(q_lo)
                _quantiles.append(q_hi)
            quantiles = sorted(set(_quantiles))
            level_was_provided = True
            return level, quantiles, level_was_provided
        if level is None and quantiles is not None:
            # we recover level from quantiles
            if not all(0 < q < 1 for q in quantiles):
                raise ValueError("`quantiles` should be floats between 0 and 1.")
            level = [abs(int(100 - 200 * q)) for q in quantiles]
            level_was_provided = False
            return sorted(set(level)), quantiles, level_was_provided
        else:
            return None, None, False

    @staticmethod
    def _level_to_quantiles(level: int | float) -> tuple[float, float]:
        """
        Given a prediction interval level (e.g. 80) return the lower & upper
        quantiles that delimit the central interval (e.g. 0.10, 0.90).
        """
        alpha = 1 - level / 100
        q_lo = alpha / 2
        q_hi = 1 - q_lo
        return q_lo, q_hi

    def maybe_convert_level_to_quantiles(
        self,
        df: pd.DataFrame,
        models: list[str],
    ) -> pd.DataFrame:
        """
        Receives a DataFrame with levels and returns
        a DataFrame with quantiles if level was provided
        """
        if self.level_was_provided or self.level is None:
            return df
        if self.quantiles is None:
            raise ValueError("No quantiles were provided.")
        out_cols = [c for c in df.columns if "-lo-" not in c and "-hi-" not in c]
        df = ufp.copy_if_pandas(df, deep=False)
        for model in models:
            for q in sorted(self.quantiles):
                if q == 0.5:
                    col = model
                else:
                    lv = int(100 - 200 * q)
                    hi_or_lo = "lo" if lv > 0 else "hi"
                    lv = abs(lv)
                    col = f"{model}-{hi_or_lo}-{lv}"
                q_col = f"{model}-q-{int(q * 100)}"
                df = ufp.assign_columns(df, q_col, df[col])
                out_cols.append(q_col)
        return df[out_cols]

    def maybe_convert_quantiles_to_level(
        self,
        df: pd.DataFrame,
        models: list[str],
    ) -> pd.DataFrame:
        """
        Receives a DataFrame with quantiles and returns
        a DataFrame with levels if quantiles were provided
        """
        if not self.level_was_provided or self.quantiles is None:
            return df
        if self.level is None:
            raise ValueError("No levels were provided.")
        out_cols = [c for c in df.columns if "-q-" not in c]
        df = ufp.copy_if_pandas(df, deep=False)
        for model in models:
            if 0 in self.level:
                mid_col = f"{model}-q-50"
                if mid_col in df:
                    df = ufp.assign_columns(df, model, df[mid_col])
                    if model not in out_cols:
                        out_cols.append(model)
            for lv in self.level:
                if lv == 0:
                    continue
                q_lo, q_hi = self._level_to_quantiles(lv)
                lo_src = f"{model}-q-{int(q_lo * 100)}"
                hi_src = f"{model}-q-{int(q_hi * 100)}"
                lo_tgt = f"{model}-lo-{lv}"
                hi_tgt = f"{model}-hi-{lv}"
                if lo_src in df and hi_src in df:
                    df = ufp.assign_columns(df, lo_tgt, df[lo_src])
                    df = ufp.assign_columns(df, hi_tgt, df[hi_src])
                    out_cols.extend([lo_tgt, hi_tgt])
        return df[out_cols]
