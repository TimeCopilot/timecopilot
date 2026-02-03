import os

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.auto import (
    AutoNHITS as _AutoNHITS,
)
from neuralforecast.auto import (
    AutoTFT as _AutoTFT,
)
from neuralforecast.common._base_model import BaseModel as NeuralForecastModel
from ray import tune

from .utils.forecaster import Forecaster
from ..utils.df_utils import to_pandas

os.environ["NIXTLA_ID_AS_COL"] = "true"


def run_neuralforecast_model(
    model: NeuralForecastModel,
    df: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    nf = NeuralForecast(
        models=[model],
        freq=freq,
    )
    nf.fit(df=df)
    fcst_df = nf.predict()
    return fcst_df


class AutoNHITS(Forecaster):
    """AutoNHITS forecaster using NeuralForecast.

    Notes:
        - Level and quantiles are not supported for AutoNHITS yet. Please open
            an issue if you need this feature.
        - AutoNHITS requires a minimum length for some frequencies.
    """

    def __init__(
        self,
        alias: str = "AutoNHITS",
        num_samples: int = 10,
        backend: str = "optuna",
        config: dict | None = None,
    ):
        self.alias = alias
        self.num_samples = num_samples
        self.backend = backend
        self.config = config

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
            raise ValueError("Level and quantiles are not supported for AutoNHITS yet.")

        df = to_pandas(df)
        inferred_freq = self._maybe_infer_freq(df, freq)
        if self.config is None:
            config = _AutoNHITS.get_default_config(h=h, backend="ray")
            config["scaler_type"] = tune.choice(["robust"])
        else:
            config = self.config

        if self.backend == "optuna":
            config = _AutoNHITS._ray_config_to_optuna(config)
        fcst_df = run_neuralforecast_model(
            model=_AutoNHITS(
                h=h,
                alias=self.alias,
                num_samples=self.num_samples,
                backend=self.backend,
                config=config,
            ),
            df=df,
            freq=inferred_freq,
        )
        return fcst_df


class AutoTFT(Forecaster):
    """AutoTFT forecaster using NeuralForecast.

    Notes:
        - Level and quantiles are not supported for AutoTFT yet. Please open
            an issue if you need this feature.
        - AutoTFT requires a minimum length for some frequencies.
    """

    def __init__(
        self,
        alias: str = "AutoTFT",
        num_samples: int = 10,
        backend: str = "optuna",
        config: dict | None = None,
    ):
        self.alias = alias
        self.num_samples = num_samples
        self.backend = backend
        self.config = config

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
            raise ValueError("Level and quantiles are not supported for AutoTFT yet.")

        df = to_pandas(df)
        inferred_freq = self._maybe_infer_freq(df, freq)
        if self.config is None:
            config = _AutoTFT.get_default_config(h=h, backend="ray")
            config["scaler_type"] = tune.choice(["robust"])
        else:
            config = self.config
        if self.backend == "optuna":
            config = _AutoTFT._ray_config_to_optuna(config)
        fcst_df = run_neuralforecast_model(
            model=_AutoTFT(
                h=h,
                alias=self.alias,
                num_samples=self.num_samples,
                backend=self.backend,
                config=config,
            ),
            df=df,
            freq=inferred_freq,
        )
        return fcst_df
