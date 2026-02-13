import os

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    _TS as StatsForecastModel,
)
from statsforecast.models import (
    ADIDA as _ADIDA,
)
from statsforecast.models import (
    IMAPA as _IMAPA,
)
from statsforecast.models import (
    AutoARIMA as _AutoARIMA,
)
from statsforecast.models import (
    AutoCES as _AutoCES,
)
from statsforecast.models import (
    AutoETS as _AutoETS,
)
from statsforecast.models import (
    CrostonClassic as _CrostonClassic,
)
from statsforecast.models import (
    DynamicOptimizedTheta as _DynamicOptimizedTheta,
)
from statsforecast.models import (
    HistoricAverage as _HistoricAverage,
)
from statsforecast.models import (
    SeasonalNaive as _SeasonalNaive,
)
from statsforecast.models import (
    Theta as _Theta,
)
from statsforecast.models import (
    ZeroModel as _ZeroModel,
)
from statsforecast.utils import ConformalIntervals

from .utils.forecaster import Forecaster, QuantileConverter, get_seasonality
from ..utils.df_utils import to_pandas

os.environ["NIXTLA_ID_AS_COL"] = "true"


def run_statsforecast_model(
    model: StatsForecastModel,
    df: pd.DataFrame,
    h: int,
    freq: str,
    level: list[int | float] | None,
    quantiles: list[float] | None,
) -> pd.DataFrame:
    sf = StatsForecast(
        models=[model],
        freq=freq,
        n_jobs=-1,
        fallback_model=_SeasonalNaive(
            season_length=get_seasonality(freq),
        ),
    )
    qc = QuantileConverter(level=level, quantiles=quantiles)
    fcst_df = sf.forecast(df=df, h=h, level=qc.level)
    fcst_df = qc.maybe_convert_level_to_quantiles(
        df=fcst_df,
        models=[model.alias],
    )
    return fcst_df


class ADIDA(Forecaster):
    """
    ADIDA (Aggregate-Disaggregate Intermittent Demand Approach) model for
    intermittent demand forecasting. Useful for series with many zero values.
    """

    def __init__(
        self,
        alias: str = "ADIDA",
        prediction_intervals: ConformalIntervals | None = None,
    ):
        """
        Args:
            alias (str): Custom name of the model.
            prediction_intervals (ConformalIntervals, optional): Information to
                compute conformal prediction intervals.
        """
        self.alias = alias
        self.prediction_intervals = prediction_intervals

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
        inferred_freq = self._maybe_infer_freq(df, freq)
        fcst_df = run_statsforecast_model(
            model=_ADIDA(alias=self.alias),
            df=df,
            h=h,
            freq=inferred_freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class AutoARIMA(Forecaster):
    """
    AutoARIMA automatically selects the best ARIMA (AutoRegressive Integrated
    Moving Average) model using an information criterion (default: AICc).
    Suitable for univariate time series with trend and seasonality.
    """

    def __init__(
        self,
        d: int | None = None,
        D: int | None = None,
        max_p: int = 5,
        max_q: int = 5,
        max_P: int = 2,
        max_Q: int = 2,
        max_order: int = 5,
        max_d: int = 2,
        max_D: int = 1,
        start_p: int = 2,
        start_q: int = 2,
        start_P: int = 1,
        start_Q: int = 1,
        stationary: bool = False,
        seasonal: bool = True,
        ic: str = "aicc",
        stepwise: bool = True,
        nmodels: int = 94,
        trace: bool = False,
        approximation: bool | None = False,
        method: str | None = None,
        truncate: bool | None = None,
        test: str = "kpss",
        test_kwargs: str | None = None,
        seasonal_test: str = "seas",
        seasonal_test_kwargs: dict | None = None,
        allowdrift: bool = True,
        allowmean: bool = True,
        blambda: float | None = None,
        biasadj: bool = False,
        season_length: int | None = None,
        alias: str = "AutoARIMA",
        prediction_intervals: ConformalIntervals | None = None,
    ):
        """
        Args:
            d (int, optional): Order of first-differencing.
            D (int, optional): Order of seasonal-differencing.
            max_p (int): Max autoregressives p.
            max_q (int): Max moving averages q.
            max_P (int): Max seasonal autoregressives P.
            max_Q (int): Max seasonal moving averages Q.
            max_order (int): Max p+q+P+Q value if not stepwise selection.
            max_d (int): Max non-seasonal differences.
            max_D (int): Max seasonal differences.
            start_p (int): Starting value of p in stepwise procedure.
            start_q (int): Starting value of q in stepwise procedure.
            start_P (int): Starting value of P in stepwise procedure.
            start_Q (int): Starting value of Q in stepwise procedure.
            stationary (bool): If True, restricts search to stationary models.
            seasonal (bool): If False, restricts search to non-seasonal models.
            ic (str): Information criterion to be used in model selection.
            stepwise (bool): If True, will do stepwise selection (faster).
            nmodels (int): Number of models considered in stepwise search.
            trace (bool): If True, the searched ARIMA models is reported.
            approximation (bool, optional): If True, conditional sums-of-squares
                estimation, final MLE.
            method (str, optional): Fitting method between maximum likelihood or
                sums-of-squares.
            truncate (bool, optional): Observations truncated series used in model
                selection.
            test (str): Unit root test to use. See `ndiffs` for details.
            test_kwargs (str, optional): Unit root test additional arguments.
            seasonal_test (str): Selection method for seasonal differences.
            seasonal_test_kwargs (dict, optional): Seasonal unit root test arguments.
            allowdrift (bool): If True, drift models terms considered.
            allowmean (bool): If True, non-zero mean models considered.
            blambda (float, optional): Box-Cox transformation parameter.
            biasadj (bool): Use adjusted back-transformed mean Box-Cox.
            season_length (int, optional): Number of observations per unit of time.
                If None, it will be inferred automatically using
                [`get_seasonality`][timecopilot.models.utils.forecaster.get_seasonality].
            alias (str): Custom name of the model.
            prediction_intervals (ConformalIntervals, optional): Information to
                compute conformal prediction intervals.
        """
        self.d = d
        self.D = D
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.max_order = max_order
        self.max_d = max_d
        self.max_D = max_D
        self.start_p = start_p
        self.start_q = start_q
        self.start_P = start_P
        self.start_Q = start_Q
        self.stationary = stationary
        self.seasonal = seasonal
        self.ic = ic
        self.stepwise = stepwise
        self.nmodels = nmodels
        self.trace = trace
        self.approximation = approximation
        self.method = method
        self.truncate = truncate
        self.test = test
        self.test_kwargs = test_kwargs
        self.seasonal_test = seasonal_test
        self.seasonal_test_kwargs = seasonal_test_kwargs
        self.allowdrift = allowdrift
        self.allowmean = allowmean
        self.blambda = blambda
        self.biasadj = biasadj
        self.season_length = season_length
        self.alias = alias
        self.prediction_intervals = prediction_intervals

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
        inferred_freq = self._maybe_infer_freq(df, freq)
        season_length = self._maybe_get_seasonality(inferred_freq)
        fcst_df = run_statsforecast_model(
            model=_AutoARIMA(
                d=self.d,
                D=self.D,
                max_p=self.max_p,
                max_q=self.max_q,
                max_P=self.max_P,
                max_Q=self.max_Q,
                max_order=self.max_order,
                max_d=self.max_d,
                max_D=self.max_D,
                start_p=self.start_p,
                start_q=self.start_q,
                start_P=self.start_P,
                start_Q=self.start_Q,
                stationary=self.stationary,
                seasonal=self.seasonal,
                ic=self.ic,
                stepwise=self.stepwise,
                nmodels=self.nmodels,
                trace=self.trace,
                approximation=self.approximation,
                method=self.method,
                truncate=self.truncate,
                test=self.test,
                test_kwargs=self.test_kwargs,
                seasonal_test=self.seasonal_test,
                seasonal_test_kwargs=self.seasonal_test_kwargs,
                allowdrift=self.allowdrift,
                allowmean=self.allowmean,
                blambda=self.blambda,
                biasadj=self.biasadj,
                season_length=season_length,
                alias=self.alias,
            ),
            df=df,
            h=h,
            freq=inferred_freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class AutoCES(Forecaster):
    """
    AutoCES automatically selects the best Complex Exponential Smoothing (CES)
    model using an information criterion (default: AICc). Suitable for
    univariate time series with trend and seasonality.
    """

    def __init__(
        self,
        season_length: int | None = None,
        model: str = "Z",
        alias: str = "AutoCES",
        prediction_intervals: ConformalIntervals | None = None,
    ):
        """
        Args:
            season_length (int, optional): Number of observations per unit of time.
                If None, it will be inferred automatically using
                [`get_seasonality`][timecopilot.models.utils.forecaster.get_seasonality].
            model (str): CES model string (e.g., "Z").
            alias (str): Custom name of the model.
            prediction_intervals (ConformalIntervals, optional): Information to
                compute conformal prediction intervals.
        """
        self.season_length = season_length
        self.model = model
        self.alias = alias
        self.prediction_intervals = prediction_intervals

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
        inferred_freq = self._maybe_infer_freq(df, freq)
        season_length = self._maybe_get_seasonality(inferred_freq)
        fcst_df = run_statsforecast_model(
            model=_AutoCES(
                season_length=season_length,
                model=self.model,
                alias=self.alias,
            ),
            df=df,
            h=h,
            freq=inferred_freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class AutoETS(Forecaster):
    """
    AutoETS automatically selects the best Error, Trend, Seasonality (ETS)
    model using an information criterion (default: AICc). Suitable for
    univariate time series with trend and seasonality.
    """

    def __init__(
        self,
        season_length: int | None = None,
        model: str = "ZZZ",
        damped: bool | None = None,
        phi: float | None = None,
        alias: str = "AutoETS",
        prediction_intervals: ConformalIntervals | None = None,
    ):
        """
        Args:
            season_length (int, optional): Number of observations per unit of time.
                If None, it will be inferred automatically using
                [`get_seasonality`][timecopilot.models.utils.forecaster.get_seasonality].
            model (str): ETS model string (e.g., "ZZZ").
            damped (bool, optional): Whether to use a damped trend.
            phi (float, optional): Damping parameter.
            alias (str): Custom name of the model.
            prediction_intervals (ConformalIntervals, optional): Information to
                compute conformal prediction intervals.
        """
        self.season_length = season_length
        self.model = model
        self.damped = damped
        self.phi = phi
        self.alias = alias
        self.prediction_intervals = prediction_intervals

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
        inferred_freq = self._maybe_infer_freq(df, freq)
        season_length = self._maybe_get_seasonality(inferred_freq)
        fcst_df = run_statsforecast_model(
            model=_AutoETS(
                season_length=season_length,
                model=self.model,
                damped=self.damped,
                phi=self.phi,
                alias=self.alias,
            ),
            df=df,
            h=h,
            freq=inferred_freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class CrostonClassic(Forecaster):
    """
    CrostonClassic model for intermittent demand forecasting.
    """

    def __init__(
        self,
        alias: str = "CrostonClassic",
        prediction_intervals: ConformalIntervals | None = None,
    ):
        """
        Args:
            alias (str): Custom name of the model.
            prediction_intervals (ConformalIntervals, optional): Information to
                compute conformal prediction intervals.
        """
        self.alias = alias
        self.prediction_intervals = prediction_intervals

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
        inferred_freq = self._maybe_infer_freq(df, freq)
        fcst_df = run_statsforecast_model(
            model=_CrostonClassic(
                alias=self.alias,
            ),
            df=df,
            h=h,
            freq=inferred_freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class DynamicOptimizedTheta(Forecaster):
    """
    Dynamic Optimized Theta model for univariate time series forecasting.
    """

    def __init__(
        self,
        season_length: int | None = None,
        alias: str = "DynamicOptimizedTheta",
    ):
        """
        Args:
            season_length (int, optional): Number of observations per unit of time.
                If None, it will be inferred automatically using
                [`get_seasonality`][timecopilot.models.utils.forecaster.get_seasonality].
            alias (str): Custom name of the model.
        """
        self.season_length = season_length
        self.alias = alias

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
        inferred_freq = self._maybe_infer_freq(df, freq)
        season_length = self._maybe_get_seasonality(inferred_freq)
        fcst_df = run_statsforecast_model(
            model=_DynamicOptimizedTheta(
                season_length=season_length,
                alias=self.alias,
            ),
            df=df,
            h=h,
            freq=inferred_freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class HistoricAverage(Forecaster):
    """
    HistoricAverage model for univariate time series forecasting.
    """

    def __init__(
        self,
        alias: str = "HistoricAverage",
        prediction_intervals: ConformalIntervals | None = None,
    ):
        """
        Args:
            alias (str): Custom name of the model.
            prediction_intervals (ConformalIntervals, optional): Information to
                compute conformal prediction intervals.
        """
        self.alias = alias
        self.prediction_intervals = prediction_intervals

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
        inferred_freq = self._maybe_infer_freq(df, freq)
        fcst_df = run_statsforecast_model(
            model=_HistoricAverage(
                alias=self.alias,
            ),
            df=df,
            h=h,
            freq=inferred_freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class IMAPA(Forecaster):
    """
    IMAPA (Intermittent Demand Aggregated Moving Average) model for
    intermittent demand forecasting. Useful for series with many zero values.
    """

    def __init__(
        self,
        alias: str = "IMAPA",
        prediction_intervals: ConformalIntervals | None = None,
    ):
        """
        Args:
            alias (str): Custom name of the model.
            prediction_intervals (ConformalIntervals, optional): Information to
                compute conformal prediction intervals.
        """
        self.alias = alias
        self.prediction_intervals = prediction_intervals

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
        inferred_freq = self._maybe_infer_freq(df, freq)
        fcst_df = run_statsforecast_model(
            model=_IMAPA(
                alias=self.alias,
            ),
            df=df,
            h=h,
            freq=inferred_freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class SeasonalNaive(Forecaster):
    """
    SeasonalNaive model for univariate time series forecasting.
    """

    def __init__(
        self,
        season_length: int | None = None,
        alias: str = "SeasonalNaive",
    ):
        """
        Args:
            season_length (int, optional): Number of observations per unit of time.
                If None, it will be inferred automatically using
                [`get_seasonality`][timecopilot.models.utils.forecaster.get_seasonality].
            alias (str): Custom name of the model.
        """
        self.season_length = season_length
        self.alias = alias

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
        inferred_freq = self._maybe_infer_freq(df, freq)
        season_length = self._maybe_get_seasonality(inferred_freq)
        fcst_df = run_statsforecast_model(
            model=_SeasonalNaive(
                season_length=season_length,
                alias=self.alias,
            ),
            df=df,
            h=h,
            freq=inferred_freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class Theta(Forecaster):
    """
    Theta model for univariate time series forecasting.
    """

    def __init__(
        self,
        season_length: int | None = None,
        alias: str = "Theta",
    ):
        """
        Args:
            season_length (int, optional): Number of observations per unit of time.
                If None, it will be inferred automatically using
                [`get_seasonality`][timecopilot.models.utils.forecaster.get_seasonality].
            alias (str): Custom name of the model.
        """
        self.season_length = season_length
        self.alias = alias

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
        inferred_freq = self._maybe_infer_freq(df, freq)
        season_length = self._maybe_get_seasonality(inferred_freq)
        fcst_df = run_statsforecast_model(
            model=_Theta(
                season_length=season_length,
                alias=self.alias,
            ),
            df=df,
            h=h,
            freq=inferred_freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df


class ZeroModel(Forecaster):
    """
    ZeroModel model for univariate time series forecasting.
    """

    def __init__(
        self,
        alias: str = "ZeroModel",
        prediction_intervals: ConformalIntervals | None = None,
    ):
        """
        Args:
            alias (str): Custom name of the model.
            prediction_intervals (ConformalIntervals, optional): Information to
                compute conformal prediction intervals.
        """
        self.alias = alias
        self.prediction_intervals = prediction_intervals

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
        inferred_freq = self._maybe_infer_freq(df, freq)
        fcst_df = run_statsforecast_model(
            model=_ZeroModel(
                alias=self.alias,
            ),
            df=df,
            h=h,
            freq=inferred_freq,
            level=level,
            quantiles=quantiles,
        )
        return fcst_df
