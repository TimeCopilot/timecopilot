import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import _zero_to_nan, mae

from ..models.utils.forecaster import (
    get_seasonality,
    maybe_convert_col_to_datetime,
    maybe_infer_freq,
)

warnings.simplefilter(
    action="ignore",
    category=FutureWarning,
)


def mase(
    df: pd.DataFrame,
    models: list[str],
    seasonality: int,
    train_df: pd.DataFrame,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> pd.DataFrame:
    mean_abs_err = mae(df, models, id_col, target_col)
    mean_abs_err = mean_abs_err.set_index(id_col)
    # assume train_df is sorted
    lagged = train_df.groupby(id_col, observed=True)[target_col].shift(seasonality)
    scale = train_df[target_col].sub(lagged).abs()
    scale = scale.groupby(train_df[id_col], observed=True).mean()
    scale[scale < 1e-2] = 0.0
    res = mean_abs_err.div(_zero_to_nan(scale), axis=0).fillna(0)
    res.index.name = id_col
    res = res.reset_index()
    return res


def generate_train_cv_splits(
    df: pd.DataFrame,
    cutoffs: pd.DataFrame,
) -> pd.DataFrame:
    """
    based on `cutoffs` (columns `unique_id`, `cutoffs`)
    generates train cv splits using `df`
    """
    df = df.merge(cutoffs, on="unique_id", how="outer")
    df = df.query("ds <= cutoff")
    df = df.reset_index(drop=True)
    return df


class DatasetParams(BaseModel):
    # TODO: make these required
    freq: str | None = Field(description="The frequency of the data", default=None)
    h: int | None = Field(description="The number of periods to forecast", default=None)
    seasonality: int | None = Field(
        description="The seasonal period of the data", default=None
    )


class ExperimentDatasetParser:
    """
    Agent that parses the dataset parameters from the user's query and df.
    """

    parser_agent: Agent

    def __init__(self, **kwargs):
        self.system_prompt = """
        You are an expert forecasting assistant.

        Your task is to analyze a user's natural language question 
        and extract all relevant
        forecasting parameters into a JSON object.

        Always produce only JSON, with no extra explanations.

        Extract the following fields if they appear in the user's text:

        - h: number of periods to forecast (integer). E.g. “12 months,” “next 30 days.”
        - freq: frequency of the data (string). It's usually a pandas frequency string.
        - seasonality: seasonal period (integer), e.g. 7, 12, 52.

        If any parameter is not mentioned, use None.

        The user can provide a head of the time series, use it to infer the frequency
        if it's not provided, or to assign the correct frequency.
        """

        self.parser_agent = Agent(
            output_type=DatasetParams,
            system_prompt=self.system_prompt,
            **kwargs,
        )

    @staticmethod
    def read_df(path: str | Path) -> pd.DataFrame:
        path_str = str(path)
        suffix = Path(path_str).suffix.lstrip(".")
        read_fn_name = f"read_{suffix}"
        if not hasattr(pd, read_fn_name):
            raise ValueError(f"Unsupported file extension: .{suffix}")
        read_fn: Callable = getattr(pd, read_fn_name)
        read_kwargs: dict[str, Any] = {}
        if path_str.startswith(("http://", "https://")):
            import io

            import requests

            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(path_str, headers=headers, timeout=30)
            resp.raise_for_status()

            if suffix in {"csv", "txt"}:
                df = read_fn(io.StringIO(resp.text))  # type: ignore[arg-type]
            elif suffix in {"parquet"}:
                import pyarrow as pa  # noqa: WPS433

                table = pa.ipc.open_file(pa.BufferReader(resp.content)).read_all()
                df = table.to_pandas()
            else:
                df = read_fn(io.BytesIO(resp.content))  # type: ignore[arg-type]
        else:
            df = read_fn(path_str, **read_kwargs)
        return df

    @staticmethod
    def _validate_df(df: pd.DataFrame | str | Path) -> pd.DataFrame:
        if isinstance(df, str | Path):
            df = ExperimentDatasetParser.read_df(df)
        if "unique_id" not in df.columns:
            df["unique_id"] = "series_0"
        return maybe_convert_col_to_datetime(df, "ds")

    @staticmethod
    def _finalize_params(
        params: DatasetParams,
        df: pd.DataFrame,
    ) -> "ExperimentDataset":
        params.freq = params.freq or maybe_infer_freq(df, freq=None)
        params.seasonality = params.seasonality or get_seasonality(params.freq)
        params.h = params.h or 2 * params.seasonality
        return ExperimentDataset(df=df, **params.dict())

    @staticmethod
    def _build_params(
        freq: str | None,
        h: int | None,
        seasonality: int | None,
        query: str | None,
        agent_result: AgentRunResult[DatasetParams] | None,
    ) -> DatasetParams:
        if query and agent_result:
            # passed by user
            # have higher priority than agent_result
            params = agent_result.output
            params.freq = freq or params.freq
            params.seasonality = seasonality or params.seasonality
            params.h = h or params.h
        else:
            params = DatasetParams(freq=freq, h=h, seasonality=seasonality)
        return params

    def parse(
        self,
        df: pd.DataFrame | str | Path,
        freq: str | None = None,
        h: int | None = None,
        seasonality: int | None = None,
        query: str | None = None,
    ) -> "ExperimentDataset":
        df = self._validate_df(df)
        agent_result = (
            self.parser_agent.run_sync(user_prompt=f"User query: {query}")
            if query
            else None
        )
        params = self._build_params(freq, h, seasonality, query, agent_result)
        return self._finalize_params(params, df)

    async def parse_async(
        self,
        df: pd.DataFrame | str | Path,
        freq: str | None = None,
        h: int | None = None,
        seasonality: int | None = None,
        query: str | None = None,
    ) -> "ExperimentDataset":
        df = self._validate_df(df)
        agent_result = (
            await self.parser_agent.run(user_prompt=f"User query: {query}")
            if query
            else None
        )
        params = self._build_params(freq, h, seasonality, query, agent_result)
        return self._finalize_params(params, df)


@dataclass
class ExperimentDataset:
    df: pd.DataFrame
    freq: str
    h: int
    seasonality: int

    def evaluate_forecast_df(
        self,
        forecast_df: pd.DataFrame,
        models: list[str],
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        forecast_df : pd.DataFrame
            df should have columns: unique_id, ds, cutoff, y, and models
        """
        for model in models:
            if forecast_df[model].isna().sum() > 0:
                print(forecast_df.loc[forecast_df[model].isna()]["unique_id"].unique())
                raise ValueError(f"model {model} has NaN values")
        cutoffs = forecast_df[["unique_id", "cutoff"]].drop_duplicates()
        train_cv_splits = generate_train_cv_splits(df=self.df, cutoffs=cutoffs)

        def add_id_cutoff(df: pd.DataFrame):
            df["id_cutoff"] = (
                df["unique_id"].astype(str) + "-" + df["cutoff"].astype(str)
            )

        for df in [cutoffs, train_cv_splits, forecast_df]:
            add_id_cutoff(df)
        partial_mase = partial(mase, seasonality=self.seasonality)
        eval_df = evaluate(
            df=forecast_df,
            train_df=train_cv_splits,
            metrics=[partial_mase],
            models=models,
            id_col="id_cutoff",
        )
        eval_df = eval_df.merge(cutoffs, on=["id_cutoff"])
        eval_df = eval_df.drop(columns=["id_cutoff"])
        eval_df = eval_df[["unique_id", "cutoff", "metric"] + models]
        return eval_df


@dataclass
class ForecastDataset:
    forecast_df: pd.DataFrame
    time_df: pd.DataFrame

    @classmethod
    def from_dir(cls, dir: str | Path):
        dir_ = Path(dir)
        forecast_df = pd.read_parquet(dir_ / "forecast_df.parquet")
        time_df = pd.read_parquet(dir_ / "time_df.parquet")
        return cls(forecast_df=forecast_df, time_df=time_df)

    @staticmethod
    def is_forecast_ready(dir: str | Path):
        dir_ = Path(dir)
        forecast_path = dir_ / "forecast_df.parquet"
        time_path = dir_ / "time_df.parquet"
        return forecast_path.exists() and time_path.exists()

    def save_to_dir(self, dir: str | Path):
        dir_ = Path(dir)
        dir_.mkdir(parents=True, exist_ok=True)
        self.forecast_df.to_parquet(dir_ / "forecast_df.parquet")
        self.time_df.to_parquet(dir_ / "time_df.parquet")
