from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

import pandas as pd
import torch
import utilsforecast.processing as ufp
from gluonts.dataset.pandas import PandasDataset
from gluonts.model.forecast import Forecast
from gluonts.torch.model.predictor import PyTorchPredictor
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from .forecaster import Forecaster, QuantileConverter
from ..utils.df_utils import to_pandas


def fix_freq(freq: str) -> str:
    # see https://github.com/awslabs/gluonts/pull/2462/files
    replacer = {"MS": "M", "ME": "M"}
    return replacer.get(freq, freq)


def maybe_convert_col_to_float32(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if df[col_name].dtype != "float32":
        df = df.copy()
        df[col_name] = df[col_name].astype("float32")
    return df


class GluonTSForecaster(Forecaster):
    def __init__(
        self,
        repo_id: str,
        filename: str,
        alias: str,
        num_samples: int = 100,
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.alias = alias
        self.num_samples = num_samples

    @property
    def checkpoint_path(self) -> str:
        return hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
        )

    @property
    def map_location(self) -> str:
        map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
        return map_location

    def load(self) -> Any:
        return torch.load(
            self.checkpoint_path,
            map_location=self.map_location,
        )

    @contextmanager
    def get_predictor(self, prediction_length: int) -> PyTorchPredictor:
        raise NotImplementedError

    def gluonts_instance_fcst_to_df(
        self,
        fcst: Forecast,
        freq: str,
        model_name: str,
        quantiles: list[float] | None,
    ) -> pd.DataFrame:
        point_forecast = fcst.median
        h = len(point_forecast)
        dates = pd.date_range(
            fcst.start_date.to_timestamp(),
            freq=freq,
            periods=h,
        )
        fcst_df = pd.DataFrame(
            {
                "ds": dates,
                "unique_id": fcst.item_id,
                model_name: point_forecast,
            }
        )
        if quantiles is not None:
            for q in quantiles:
                fcst_df = ufp.assign_columns(
                    fcst_df,
                    f"{model_name}-q-{int(q * 100)}",
                    fcst.quantile(q),
                )
        return fcst_df

    def gluonts_fcsts_to_df(
        self,
        fcsts: Iterable[Forecast],
        freq: str,
        model_name: str,
        quantiles: list[float] | None,
    ) -> pd.DataFrame:
        df = []
        for fcst in tqdm(fcsts):
            fcst_df = self.gluonts_instance_fcst_to_df(
                fcst=fcst,
                freq=freq,
                model_name=model_name,
                quantiles=quantiles,
            )
            df.append(fcst_df)
        return pd.concat(df).reset_index(drop=True)

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
                valid values. If None, the frequency will be inferred from the data.
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
        df = maybe_convert_col_to_float32(df, "y")
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        gluonts_dataset = PandasDataset.from_long_dataframe(
            df.copy(deep=False),
            target="y",
            item_id="unique_id",
            timestamp="ds",
            freq=fix_freq(freq),
        )
        with self.get_predictor(prediction_length=h) as predictor:
            fcsts = predictor.predict(
                gluonts_dataset,
                num_samples=self.num_samples,
            )
        fcst_df = self.gluonts_fcsts_to_df(
            fcsts,
            freq=freq,
            model_name=self.alias,
            quantiles=qc.quantiles,
        )
        if qc.quantiles is not None:
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )

        return fcst_df
