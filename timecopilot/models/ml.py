import os

import pandas as pd
from mlforecast.auto import (
    AutoCatboost as _AutoCatboost,
)
from mlforecast.auto import (
    AutoElasticNet as _AutoElasticNet,
)
from mlforecast.auto import (
    AutoLasso as _AutoLasso,
)
from mlforecast.auto import (
    AutoLightGBM as _AutoLightGBM,
)
from mlforecast.auto import (
    AutoLinearRegression as _AutoLinearRegression,
)
from mlforecast.auto import (
    AutoMLForecast,
)
from mlforecast.auto import (
    AutoRandomForest as _AutoRandomForest,
)
from mlforecast.auto import (
    AutoRidge as _AutoRidge,
)
from mlforecast.auto import (
    AutoXGBoost as _AutoXGBoost,
)
from mlforecast.utils import PredictionIntervals

from .utils.forecaster import Forecaster, QuantileConverter, get_seasonality

os.environ["NIXTLA_ID_AS_COL"] = "true"


def _make_safe_init_config(season_length: int):
    """Return an AutoMLForecast init_config that avoids transforms that fail on
    short series (Differences and log transforms).

    The standard search space includes Differences([1]) and log1p transforms
    which can produce empty feature matrices when the series is short. This
    safe config restricts target transforms to None and LocalStandardScaler only,
    while keeping the same frequency-aware lag candidates.
    """
    from mlforecast.target_transforms import LocalStandardScaler
    from window_ops.ewm import ExponentiallyWeightedMean
    from window_ops.rolling import RollingMean

    candidate_targ_tfms = [None, [LocalStandardScaler()]]

    candidate_lags = [None, [season_length]]
    seasonality2extra_lags = {
        7: [[7, 14], [7, 28]],
        12: [list(range(1, 13))],
        24: [list(range(1, 25)), list(range(24, 24 * 7 + 1, 24))],
        52: [list(range(4, 53, 4))],
    }
    if season_length in seasonality2extra_lags:
        candidate_lags.extend(seasonality2extra_lags[season_length])

    candidate_lag_tfms = [None, {1: [ExponentiallyWeightedMean(0.9)]}]
    if season_length > 1:
        candidate_lag_tfms.append(
            {
                1: [ExponentiallyWeightedMean(0.9)],
                season_length: [RollingMean(window_size=season_length, min_samples=1)],
            }
        )

    def init_config(trial) -> dict:
        tfm_idx = trial.suggest_categorical(
            "target_transforms_idx", list(range(len(candidate_targ_tfms)))
        )
        lags_idx = trial.suggest_categorical(
            "lags_idx", list(range(len(candidate_lags)))
        )
        lag_tfms_idx = trial.suggest_categorical(
            "lag_transforms_idx", list(range(len(candidate_lag_tfms)))
        )
        return {
            "lags": candidate_lags[lags_idx],
            "target_transforms": candidate_targ_tfms[tfm_idx],
            "lag_transforms": candidate_lag_tfms[lag_tfms_idx],
        }

    return init_config


def run_automlforecast_model(
    model,
    model_name: str,
    df: pd.DataFrame,
    h: int,
    freq: str,
    alias: str,
    num_samples: int,
    cv_n_windows: int,
    level: list[int | float] | None,
    quantiles: list[float] | None,
    init_config=None,
    safe_config: bool = False,
) -> pd.DataFrame:
    if level is not None and quantiles is not None:
        raise ValueError(
            "You must not provide both `level` and `quantiles` simultaneously."
        )
    if level is not None:
        raise ValueError(
            f"Level is not supported for {alias}. " "Please use `quantiles` instead."
        )
    qc = QuantileConverter(level=None, quantiles=quantiles)
    season_length = get_seasonality(freq)
    if init_config is None and safe_config:
        init_config = _make_safe_init_config(season_length)
    mf = AutoMLForecast(
        models=[model],
        freq=freq,
        season_length=season_length if init_config is None else None,
        init_config=init_config,
        num_threads=-1,
    )
    prediction_intervals = (
        PredictionIntervals(n_windows=cv_n_windows) if qc.level is not None else None
    )
    mf.fit(
        df=df,
        n_windows=cv_n_windows,
        h=h,
        num_samples=num_samples,
        prediction_intervals=prediction_intervals,
    )
    fcst_df = mf.predict(h=h, level=qc.level)
    fcst_df.columns = [c.replace(model_name, alias) for c in fcst_df.columns]
    fcst_df = qc.maybe_convert_level_to_quantiles(fcst_df, [alias])
    return fcst_df


class AutoLGBM(Forecaster):
    """AutoLGBM forecaster using AutoMLForecast with LightGBM.

    Notes:
        - Level is not supported. Use `quantiles` for probabilistic forecasts.
        - AutoLGBM requires a minimum length for some frequencies.
    """

    def __init__(
        self,
        alias: str = "AutoLGBM",
        num_samples: int = 10,
        cv_n_windows: int = 5,
        init_config=None,
        safe_config: bool = False,
    ):
        self.alias = alias
        self.num_samples = num_samples
        self.cv_n_windows = cv_n_windows
        self.init_config = init_config
        self.safe_config = safe_config

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
        freq = self._maybe_infer_freq(df, freq)
        return run_automlforecast_model(
            model=_AutoLightGBM(),
            model_name="AutoLightGBM",
            df=df,
            h=h,
            freq=freq,
            alias=self.alias,
            num_samples=self.num_samples,
            cv_n_windows=self.cv_n_windows,
            level=level,
            quantiles=quantiles,
            init_config=self.init_config,
            safe_config=self.safe_config,
        )


class AutoLinearRegression(Forecaster):
    """AutoLinearRegression forecaster using AutoMLForecast with LinearRegression.

    Notes:
        - Level is not supported. Use `quantiles` for probabilistic forecasts.
        - AutoLinearRegression requires a minimum length for some frequencies.
    """

    def __init__(
        self,
        alias: str = "AutoLinearRegression",
        num_samples: int = 10,
        cv_n_windows: int = 5,
        init_config=None,
        safe_config: bool = False,
    ):
        self.alias = alias
        self.num_samples = num_samples
        self.cv_n_windows = cv_n_windows
        self.init_config = init_config
        self.safe_config = safe_config

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
                Not supported for AutoLinearRegression. Use `quantiles` instead.
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
        freq = self._maybe_infer_freq(df, freq)
        return run_automlforecast_model(
            model=_AutoLinearRegression(),
            model_name="AutoLinearRegression",
            df=df,
            h=h,
            freq=freq,
            alias=self.alias,
            num_samples=self.num_samples,
            cv_n_windows=self.cv_n_windows,
            level=level,
            quantiles=quantiles,
            init_config=self.init_config,
            safe_config=self.safe_config,
        )


class AutoXGBoost(Forecaster):
    """AutoXGBoost forecaster using AutoMLForecast with XGBoost.

    Notes:
        - Level is not supported. Use `quantiles` for probabilistic forecasts.
        - AutoXGBoost requires a minimum length for some frequencies.
        - Requires the `xgboost` package to be installed.
    """

    def __init__(
        self,
        alias: str = "AutoXGBoost",
        num_samples: int = 10,
        cv_n_windows: int = 5,
        init_config=None,
        safe_config: bool = False,
    ):
        self.alias = alias
        self.num_samples = num_samples
        self.cv_n_windows = cv_n_windows
        self.init_config = init_config
        self.safe_config = safe_config

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
                Not supported for AutoXGBoost. Use `quantiles` instead.
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
        freq = self._maybe_infer_freq(df, freq)
        return run_automlforecast_model(
            model=_AutoXGBoost(),
            model_name="AutoXGBoost",
            df=df,
            h=h,
            freq=freq,
            alias=self.alias,
            num_samples=self.num_samples,
            cv_n_windows=self.cv_n_windows,
            level=level,
            quantiles=quantiles,
            init_config=self.init_config,
            safe_config=self.safe_config,
        )


class AutoRidge(Forecaster):
    """AutoRidge forecaster using AutoMLForecast with Ridge regression.

    Notes:
        - Level is not supported. Use `quantiles` for probabilistic forecasts.
        - AutoRidge requires a minimum length for some frequencies.
    """

    def __init__(
        self,
        alias: str = "AutoRidge",
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
                Not supported for AutoRidge. Use `quantiles` instead.
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
        freq = self._maybe_infer_freq(df, freq)
        return run_automlforecast_model(
            model=_AutoRidge(),
            model_name="AutoRidge",
            df=df,
            h=h,
            freq=freq,
            alias=self.alias,
            num_samples=self.num_samples,
            cv_n_windows=self.cv_n_windows,
            level=level,
            quantiles=quantiles,
        )


class AutoLasso(Forecaster):
    """AutoLasso forecaster using AutoMLForecast with Lasso regression.

    Notes:
        - Level is not supported. Use `quantiles` for probabilistic forecasts.
        - AutoLasso requires a minimum length for some frequencies.
    """

    def __init__(
        self,
        alias: str = "AutoLasso",
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
                Not supported for AutoLasso. Use `quantiles` instead.
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
        freq = self._maybe_infer_freq(df, freq)
        return run_automlforecast_model(
            model=_AutoLasso(),
            model_name="AutoLasso",
            df=df,
            h=h,
            freq=freq,
            alias=self.alias,
            num_samples=self.num_samples,
            cv_n_windows=self.cv_n_windows,
            level=level,
            quantiles=quantiles,
        )


class AutoElasticNet(Forecaster):
    """AutoElasticNet forecaster using AutoMLForecast with ElasticNet.

    Notes:
        - Level is not supported. Use `quantiles` for probabilistic forecasts.
        - AutoElasticNet requires a minimum length for some frequencies.
    """

    def __init__(
        self,
        alias: str = "AutoElasticNet",
        num_samples: int = 10,
        cv_n_windows: int = 5,
        init_config=None,
        safe_config: bool = False,
    ):
        self.alias = alias
        self.num_samples = num_samples
        self.cv_n_windows = cv_n_windows
        self.init_config = init_config
        self.safe_config = safe_config

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
                Not supported for AutoElasticNet. Use `quantiles` instead.
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
        freq = self._maybe_infer_freq(df, freq)
        return run_automlforecast_model(
            model=_AutoElasticNet(),
            model_name="AutoElasticNet",
            df=df,
            h=h,
            freq=freq,
            alias=self.alias,
            num_samples=self.num_samples,
            cv_n_windows=self.cv_n_windows,
            level=level,
            quantiles=quantiles,
            init_config=self.init_config,
            safe_config=self.safe_config,
        )


class AutoRandomForest(Forecaster):
    """AutoRandomForest forecaster using AutoMLForecast with RandomForest.

    Notes:
        - Level is not supported. Use `quantiles` for probabilistic forecasts.
        - AutoRandomForest requires a minimum length for some frequencies.
    """

    def __init__(
        self,
        alias: str = "AutoRandomForest",
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
                Not supported for AutoRandomForest. Use `quantiles` instead.
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
        freq = self._maybe_infer_freq(df, freq)
        return run_automlforecast_model(
            model=_AutoRandomForest(),
            model_name="AutoRandomForest",
            df=df,
            h=h,
            freq=freq,
            alias=self.alias,
            num_samples=self.num_samples,
            cv_n_windows=self.cv_n_windows,
            level=level,
            quantiles=quantiles,
        )


class AutoCatboost(Forecaster):
    """AutoCatboost forecaster using AutoMLForecast with CatBoost.

    Notes:
        - Level is not supported. Use `quantiles` for probabilistic forecasts.
        - AutoCatboost requires a minimum length for some frequencies.
        - Requires the `catboost` package to be installed.
    """

    def __init__(
        self,
        alias: str = "AutoCatboost",
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
                Not supported for AutoCatboost. Use `quantiles` instead.
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
        freq = self._maybe_infer_freq(df, freq)
        return run_automlforecast_model(
            model=_AutoCatboost(),
            model_name="AutoCatboost",
            df=df,
            h=h,
            freq=freq,
            alias=self.alias,
            num_samples=self.num_samples,
            cv_n_windows=self.cv_n_windows,
            level=level,
            quantiles=quantiles,
        )
