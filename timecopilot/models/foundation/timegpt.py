import os

import pandas as pd
from nixtla import NixtlaClient

from ..utils.forecaster import Forecaster
from ...utils.df_utils import to_pandas


class TimeGPT(Forecaster):
    """
    TimeGPT is a pre-trained foundation model for time series forecasting and anomaly
    detection, developed by Nixtla. It is based on a large encoder-decoder transformer
    architecture trained on over 100 billion data points from diverse domains.
    See the [official repo](https://github.com/nixtla/nixtla),
    [docs](https://www.nixtla.io/docs),
    and [arXiv:2310.03589](https://arxiv.org/abs/2310.03589) for more details.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 1,
        model: str = "timegpt-1",
        alias: str = "TimeGPT",
    ):
        """
        Args:
            api_key (str, optional): API key for authenticating with the Nixtla TimeGPT
                API. If not provided, will use the `NIXTLA_API_KEY`
                environment variable.
            base_url (str, optional): Base URL for the TimeGPT API. Defaults to the
                official Nixtla endpoint.
            max_retries (int, optional): Maximum number of retries for API requests.
                Defaults to 1.
            model (str, optional): Model name or version to use. Defaults to
                "timegpt-1". See the [Nixtla docs](https://www.nixtla.io/docs) for
                available models.
            alias (str, optional): Name to use for the model in output DataFrames and
                logs. Defaults to "TimeGPT".

        Notes:
            **Academic Reference:**

            - Paper: [TimeGPT-1](https://arxiv.org/abs/2310.03589)

            **Resources:**

            - GitHub: [Nixtla/nixtla](https://github.com/Nixtla/nixtla)

            **Technical Details:**

            - TimeGPT is a foundation model for time series forecasting designed for
              production-ready forecasting with minimal setup.
            - Provides zero-shot forecasting capabilities across various
              domains and frequencies.
            - Requires a valid API key from Nixtla to use.
            - For more information, see the
              [TimeGPT documentation](https://www.nixtla.io/docs).
        """
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.model = model
        self.alias = alias

    def _get_client(self) -> NixtlaClient:
        if self.api_key is None:  # noqa: SIM108
            api_key = os.environ["NIXTLA_API_KEY"]
        else:
            api_key = self.api_key
        return NixtlaClient(
            api_key=api_key,
            base_url=self.base_url,
            max_retries=self.max_retries,
        )

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
        client = self._get_client()
        fcst_df = client.forecast(
            df=df,
            h=h,
            freq=freq,
            model=self.model,
            level=level,
            quantiles=quantiles,
        )
        fcst_df["ds"] = pd.to_datetime(fcst_df["ds"])
        cols = [col.replace("TimeGPT", self.alias) for col in fcst_df.columns]
        fcst_df.columns = cols
        return fcst_df
