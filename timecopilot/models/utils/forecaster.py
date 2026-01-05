import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects
import utilsforecast.processing as ufp
from gluonts.time_feature.seasonality import (
    DEFAULT_SEASONALITIES,
)
from gluonts.time_feature.seasonality import (
    get_seasonality as _get_seasonality,
)
from scipy import stats
from tqdm import tqdm
from utilsforecast.plotting import plot_series
from utilsforecast.processing import (
    backtest_splits,
    drop_index_if_pandas,
    join,
    maybe_compute_sort_indices,
    take_rows,
    vertical_concat,
)
from utilsforecast.validation import ensure_time_dtype


def get_seasonality(
    freq: str,
    custom_seasonalities: dict[str, int] | None = None,
) -> int:
    # fmt: off
    """
    Get the seasonality of a frequency.

    Args:
        freq (str): The frequency to get the seasonality of.
        custom_seasonalities (dict[str, int] | None): Custom seasonalities to use.
            If None, the default seasonalities are used.

    Returns:
        int: The seasonality of the frequency.

    Example:
        ```python
        from timecopilot.models.utils.forecaster import get_seasonality

        get_seasonality("D", custom_seasonalities={"D": 7})
        # 7
        get_seasonality("D") # default seasonalities are used
        # 1
        ```
    """
    # fmt: on
    if custom_seasonalities is None:
        custom_seasonalities = dict()
    return _get_seasonality(
        freq,
        seasonalities=DEFAULT_SEASONALITIES | custom_seasonalities,
    )


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

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        X_df: pd.DataFrame | None = None,  # known future covariates
        static_features: list[str] | None = None,  # past-only covariates (in df)
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts for time series data using the model.

        Notes on exogenous covariates:
            - `static_features` refers to past-only covariate columns that already
              exist in `df` (the historical/context DataFrame).
            - `X_df` refers to known-future covariates and MUST be provided as a separate
              `future_df` for models that support it (Chronos-2 only in this library).

        Args:
            df (pd.DataFrame):
                DataFrame containing the time series to forecast. It must
                include as columns:

                    - "unique_id": an ID column to distinguish multiple series.
                    - "ds": a time column indicating timestamps or periods.
                    - "y": a target column with the observed values.

                It may also include past-only covariates (listed in `static_features`).

            h (int):
                Forecast horizon specifying how many future steps to predict.
            freq (str, optional):
                Frequency of the time series (e.g. "D" for daily, "M" for
                monthly). If not provided, the frequency will be inferred
                from the data.
            X_df (pd.DataFrame, optional):
                DataFrame containing known future covariates. Must include
                ["unique_id", "ds"] plus covariate columns.
            static_features (list[str], optional):
                Column names of past-only covariates present in `df`.
            level (list[int | float], optional):
                Confidence levels for prediction intervals, expressed as
                percentages (e.g. [80, 95]).
            quantiles (list[float], optional):
                List of quantiles to forecast (0..1). Should not be used
                simultaneously with `level`.

        Returns:
            pd.DataFrame:
                Forecast DataFrame with ["unique_id", "ds"] and model outputs.
        """
        raise NotImplementedError("This method must be implemented in a subclass.")

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        n_windows: int = 1,
        step_size: int | None = None,
        X_df: pd.DataFrame | None = None,  # known future covariates (NOT supported in CV)
        static_features: list[str] | None = None,  # past-only covariates (NOT supported in CV)
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Perform cross-validation on time series data.

        IMPORTANT:
            Cross-validation with exogenous variables is explicitly NOT supported.
            We fail loudly to avoid silently doing the wrong thing (e.g., leaking
            future covariates across windows, or mis-aligning future_df per split).

            Chronos-2 exogenous support is wired ONLY for the simple `forecast(...)`
            call via Chronos2Pipeline.predict_df.

        (Other docstring text unchanged.)
        """
        # --- Explicitly block CV when exogenous variables are requested. ---
        # Why: rolling-origin splits require per-window construction of future_df.
        # This library intentionally does not implement that logic to avoid subtle
        # leakage/misalignment bugs. Only the direct Chronos-2 predict_df wiring is supported.
        requested_exog = (
            X_df is not None
            or static_features is not None
            or bool(getattr(self, "exog_cols", None))
        )
        if requested_exog:
            raise NotImplementedError(
                "Cross validation with exogenous variables is not supported. "
                "Use `forecast(...)` instead, and only Chronos-2 supports exogenous "
                "covariates in this library."
            )

        freq = self._maybe_infer_freq(df, freq)
        df = maybe_convert_col_to_datetime(df, "ds")
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
        (unchanged)
        """
        freq = self._maybe_infer_freq(df, freq)
        df = maybe_convert_col_to_datetime(df, "ds")
        if h is None:
            h = self._maybe_get_seasonality(freq)
        min_series_length = df.groupby("unique_id").size().min()
        # we require at least one observation before the first forecast
        max_possible_windows = (min_series_length - 1) // h
        if n_windows is None:
            _n_windows = max_possible_windows
        else:
            _n_windows = min(n_windows, max_possible_windows)
        if _n_windows < 1:
            raise ValueError(
                f"Cannot perform anomaly detection: series too short. "
                f"Minimum series length required: {h + 1}, "
                f"actual minimum length: {min_series_length}"
            )
        cv_results = self.cross_validation(
            df=df,
            h=h,
            freq=freq,
            n_windows=_n_windows,
            step_size=h,  # this is the default but who knows, anxiety
        )
        cv_results["residuals"] = cv_results["y"] - cv_results[self.alias]
        residual_stats = (
            cv_results.groupby("unique_id")["residuals"].std().reset_index()
        )
        residual_stats.columns = ["unique_id", "residual_std"]
        cv_results = cv_results.merge(residual_stats, on="unique_id", how="left")
        cv_results["z_score"] = cv_results["residuals"] / cv_results["residual_std"]
        alpha = 1 - level / 100
        critical_z = stats.norm.ppf(1 - alpha / 2)
        an_col = f"{self.alias}-anomaly"
        cv_results[an_col] = np.abs(cv_results["z_score"]) > critical_z
        lo_col = f"{self.alias}-lo-{int(level)}"
        hi_col = f"{self.alias}-hi-{int(level)}"
        margin = critical_z * cv_results["residual_std"]
        cv_results[lo_col] = cv_results[self.alias] - margin
        cv_results[hi_col] = cv_results[self.alias] + margin
        output_cols = [
            "unique_id",
            "ds",
            "cutoff",
            "y",
            self.alias,
            lo_col,
            hi_col,
            an_col,
        ]
        result = cv_results[output_cols].copy()
        result = drop_index_if_pandas(result)
        return result

    @staticmethod
    def plot(
        df: pd.DataFrame | None = None,
        forecasts_df: pd.DataFrame | None = None,
        ids: list[str] | None = None,
        plot_random: bool = True,
        max_ids: int | None = 8,
        models: list[str] | None = None,
        level: list[float] | None = None,
        max_insample_length: int | None = None,
        plot_anomalies: bool = False,
        engine: str = "matplotlib",
        palette: str | None = None,
        seed: int | None = None,
        resampler_kwargs: dict | None = None,
        ax: plt.Axes | np.ndarray | plotly.graph_objects.Figure | None = None,
    ):
        """Plot forecasts and insample values. (unchanged)"""
        df = ensure_time_dtype(df, time_col="ds")
        if forecasts_df is not None:
            forecasts_df = ensure_time_dtype(forecasts_df, time_col="ds")
            if any("anomaly" in col for col in forecasts_df.columns):
                df = None
                models = [
                    col.split("-")[0]
                    for col in forecasts_df.columns
                    if col.endswith("-anomaly")
                ]
                forecasts_df = ufp.drop_columns(
                    forecasts_df,
                    [f"{model}-anomaly" for model in models],
                )
                lv_cols = [
                    c.replace(f"{model}-lo-", "")
                    for model in models
                    for c in forecasts_df.columns
                    if f"{model}-lo-" in c
                ]
                level = [float(c) if "." in c else int(c) for c in lv_cols]
                level = list(set(level))
                plot_anomalies = True
        return plot_series(
            df=df,
            forecasts_df=forecasts_df,
            ids=ids,
            plot_random=plot_random,
            max_ids=max_ids,
            models=models,
            level=level,
            max_insample_length=max_insample_length,
            plot_anomalies=plot_anomalies,
            engine=engine,
            resampler_kwargs=resampler_kwargs,
            palette=palette,
            seed=seed,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            ax=ax,
        )


class QuantileConverter:
    """Handles inputs and outputs for probabilistic forecasts."""
    # (unchanged)
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
            if not all(0 < q < 1 for q in quantiles):
                raise ValueError("`quantiles` should be floats between 0 and 1.")
            level = [abs(int(100 - 200 * q)) for q in quantiles]
            level_was_provided = False
            return sorted(set(level)), quantiles, level_was_provided
        else:
            return None, None, False

    @staticmethod
    def _level_to_quantiles(level: int | float) -> tuple[float, float]:
        alpha = 1 - level / 100
        q_lo = alpha / 2
        q_hi = 1 - q_lo
        return q_lo, q_hi

    def maybe_convert_level_to_quantiles(
        self,
        df: pd.DataFrame,
        models: list[str],
    ) -> pd.DataFrame:
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