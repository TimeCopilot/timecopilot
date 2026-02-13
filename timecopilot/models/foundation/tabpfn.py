import sys
from contextlib import contextmanager

if sys.version_info >= (3, 13):
    raise ImportError("TabPFN requires Python < 3.13")

import numpy as np
import pandas as pd
import torch
from tabpfn_client import set_access_token
from tabpfn_time_series import (
    TABPFN_TS_DEFAULT_QUANTILE_CONFIG,
    FeatureTransformer,
    TabPFNMode,
    TabPFNTimeSeriesPredictor,
    TimeSeriesDataFrame,
)
from tabpfn_time_series.data_preparation import generate_test_X
from tabpfn_time_series.features import (
    AutoSeasonalFeature,
    CalendarFeature,
    RunningIndexFeature,
)
from tabpfn_time_series.features.feature_generator_base import (
    FeatureGenerator,
)

from ..utils.forecaster import Forecaster, QuantileConverter
from ...utils.df_utils import to_pandas


class TabPFN(Forecaster):
    """
    TabPFN is a zero-shot time series forecasting model that frames univariate
    forecasting as a tabular regression problem using TabPFNv2. It supports both
    point and probabilistic forecasts, and can incorporate exogenous variables via
    feature engineering. See the
    [official repo](https://github.com/PriorLabs/tabpfn-time-series) for more details.
    """

    def __init__(
        self,
        features: list[FeatureGenerator] | None = None,
        context_length: int = 4096,
        mode: TabPFNMode | None = None,
        api_key: str | None = None,
        alias: str = "TabPFN",
    ):
        """
        Args:
            features (list[FeatureGenerator], optional): List of TabPFN-TS feature
                generators to use for feature engineering. If None, uses
                `[RunningIndexFeature(), CalendarFeature(), AutoSeasonalFeature()]`
                by default.
                See
                [TabPFN-TS features](https://github.com/PriorLabs/tabpfn-time-series/
                tree/main/tabpfn_time_series/features).
            context_length (int, optional): Maximum context length (input window size)
                for the model. Defaults to 4096. Controls how much history is used for
                each forecast.
            mode (TabPFNMode, optional): Inference mode for TabPFN. If None, uses LOCAL
                (`"tabpfn-local"`) if a GPU is available, otherwise CLIENT (cloud
                inference via `"tabpfn-client"`). See
                [TabPFN-TS docs](https://github.com/PriorLabs/tabpfn-time-series/
                blob/3cd61ad556466de837edd1c6036744176145c024/tabpfn_time_series/
                predictor.py#L11) for available modes.
            api_key (str, optional): API key for tabpfn-client cloud inference. Required
                if using CLIENT mode and not already set in the environment.
            alias (str, optional): Name to use for the model in output DataFrames and
                logs. Defaults to "TabPFN".

        Notes:
            **Academic Reference:**

            - Paper: [From Tables to Time: How TabPFN-v2 Outperforms
            Specialized Time Series Forecasting Models](https://arxiv.org/abs/2501.02945)

            **Resources:**

            - GitHub: [PriorLabs/tabpfn-time-series](https://github.com/PriorLabs/tabpfn-time-series)

            **Technical Details:**

            - For LOCAL mode, a CUDA-capable GPU is recommended for best performance.
            - The model is only available for Python < 3.13.
        """
        if features is None:
            features = [
                RunningIndexFeature(),
                CalendarFeature(),
                AutoSeasonalFeature(),
            ]
        self.feature_transformer = FeatureTransformer(features)
        self.context_length = context_length
        if mode is None:
            mode = TabPFNMode.LOCAL if torch.cuda.is_available() else TabPFNMode.CLIENT
        if mode == TabPFNMode.CLIENT and api_key is not None:
            set_access_token(api_key)
        self.mode = mode
        self.alias = alias

    @contextmanager
    def _get_model(self) -> TabPFNTimeSeriesPredictor:
        model = TabPFNTimeSeriesPredictor(tabpfn_mode=self.mode)
        try:
            yield model
        finally:
            del model
            torch.cuda.empty_cache()

    def _forecast(
        self,
        model: TabPFNTimeSeriesPredictor,
        df: pd.DataFrame,
        h: int,
        quantiles: list[float] | None,
    ) -> pd.DataFrame:
        """handles distinction between quantiles and no quantiles"""
        renamer = {
            "unique_id": "item_id",
            "ds": "timestamp",
            "y": "target",
        }
        tsdf = df.rename(columns=renamer)
        tsdf = TimeSeriesDataFrame(tsdf.set_index(["item_id", "timestamp"]))
        if self.context_length > 0:
            tsdf = tsdf.slice_by_timestep(-self.context_length, None)
        future_tsdf = generate_test_X(tsdf, h)
        tsdf, future_tsdf = self.feature_transformer.transform(tsdf, future_tsdf)
        fcst_df = model.predict(tsdf, future_tsdf)
        fcst_df = fcst_df.reset_index()
        re_renamer = {v: k for k, v in renamer.items()}
        re_renamer["target"] = self.alias
        fcst_df = fcst_df.rename(columns=re_renamer)
        if quantiles is None:
            fcst_df = fcst_df[["unique_id", "ds", self.alias]]
        else:
            q_renamer = {
                q_orig: f"{self.alias}-q-{int(100 * q_user)}"
                for q_orig, q_user in zip(
                    TABPFN_TS_DEFAULT_QUANTILE_CONFIG,
                    quantiles,
                    strict=True,
                )
            }
            fcst_df = fcst_df.rename(columns=q_renamer)
        return pd.DataFrame(fcst_df)

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
        qc = QuantileConverter(level=level, quantiles=quantiles)
        if qc.quantiles is not None and not np.allclose(
            qc.quantiles,
            TABPFN_TS_DEFAULT_QUANTILE_CONFIG,
        ):
            raise ValueError(
                "TabPFN only supports the default quantiles, "
                "please use the default quantiles or default level, "
            )
        with self._get_model() as model:
            fcst_df = self._forecast(
                model,
                df,
                h,
                quantiles=qc.quantiles,
            )
        if qc.quantiles is not None:
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        return fcst_df
