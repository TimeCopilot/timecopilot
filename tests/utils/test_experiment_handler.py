import json
from functools import partial

import pandas as pd
import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from utilsforecast.data import generate_series
from utilsforecast.evaluation import evaluate
from utilsforecast.processing import (
    backtest_splits,
    drop_index_if_pandas,
    join,
    maybe_compute_sort_indices,
    take_rows,
    vertical_concat,
)

from ..models.conftest import models
from timecopilot.models.utils.forecaster import (
    get_seasonality,
    maybe_convert_col_to_datetime,
)
from timecopilot.utils.experiment_handler import (
    ExperimentDataset,
    ExperimentDatasetParser,
    generate_train_cv_splits,
    mase,
)


def generate_train_cv_splits_from_backtest_splits(
    df: pd.DataFrame,
    n_windows: int,
    h: int,
    freq: str,
    step_size: int = 1,
):
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
    for _, (cutoffs, train, _) in enumerate(splits):
        train_cv = join(train, cutoffs, on="unique_id")
        results.append(train_cv)
    out = vertical_concat(results)
    out = drop_index_if_pandas(out)
    return out


def generate_exp_dataset(
    n_series,
    freq,
    return_df: bool = False,
) -> ExperimentDataset | pd.DataFrame:
    df = generate_series(n_series, freq=freq, min_length=12)
    df["unique_id"] = df["unique_id"].astype(str)
    if return_df:
        return df
    return ExperimentDataset(df=df, freq=freq, h=2, seasonality=7)


def evaluate_cv_from_scratch(
    df: pd.DataFrame,
    fcst_df: pd.DataFrame,
    models: list[str],
    seasonality: int,
) -> pd.DataFrame:
    partial_mase = partial(mase, seasonality=seasonality)
    uids = df["unique_id"].unique()
    results = []
    for uid in uids:  # noqa: B007
        df_ = df.query("unique_id == @uid")
        fcst_df_ = fcst_df.query("unique_id == @uid")
        cutoffs = fcst_df_["cutoff"].unique()
        for cutoff in cutoffs:
            df__ = df_.query("ds <= @cutoff")
            fcst_df__ = fcst_df_.query("cutoff == @cutoff")
            eval_df = evaluate(
                df=fcst_df__,
                train_df=df__,
                metrics=[partial_mase],
                models=models,
            )
            eval_df["cutoff"] = cutoff
            results.append(eval_df)
    out = pd.concat(results)
    out = out[["unique_id", "cutoff", "metric"] + models]
    return out


def sort_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.sort_values(cols).reset_index(drop=True)


def response_agent_fn(payload: dict) -> ModelResponse:
    def _response_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        json_payload = json.dumps(payload)
        return ModelResponse(
            parts=[ToolCallPart(tool_name="final_result", args=json_payload)]
        )

    return _response_fn


def assert_experiment_dataset_equal(
    exp_dataset: ExperimentDataset,
    expected_exp_dataset: ExperimentDataset,
):
    # Using this because problem with python 3.13
    # see https://github.com/TimeCopilot/timecopilot/actions/runs/15988311135/job/45096742719?pr=25
    assert exp_dataset.df.equals(expected_exp_dataset.df)
    assert exp_dataset.freq == expected_exp_dataset.freq
    assert exp_dataset.h == expected_exp_dataset.h
    assert exp_dataset.seasonality == expected_exp_dataset.seasonality


@pytest.mark.parametrize(
    "freq,h,seasonality",
    [
        ("H", 2, 7),
        ("MS", 2, 12),
        ("D", 2, 7),
        ("W-MON", 2, 7),
    ],
)
def test_parse_params_from_complete_query(freq, h, seasonality):
    df = generate_series(n_series=5, freq=freq)
    query = f"""
        I have a time series with frequency {freq}, 
        seasonality {seasonality}, and horizon {h}.
    """
    # In this test we don't pass any parameters to the parser
    # so it should infer them from the query and df
    test_model = FunctionModel(
        response_agent_fn(
            payload={
                "freq": freq,
                "h": h,
                "seasonality": seasonality,
            }
        )
    )
    exp_dataset = ExperimentDatasetParser(model=test_model).parse(
        df,
        freq=None,
        h=None,
        seasonality=None,
        query=query,
    )
    assert_experiment_dataset_equal(
        exp_dataset,
        ExperimentDataset(
            df=df,
            freq=freq,
            h=h,
            seasonality=seasonality,
        ),
    )


@pytest.mark.parametrize(
    "freq,h",
    [
        ("D", 14),
        ("H", 6),
    ],
)
def test_parse_params_from_partial_query(freq, h):
    """If the query omits `seasonality`, the parser should infer it from `freq`."""
    df = generate_series(n_series=3, freq=freq, min_length=12)
    query = (
        f"Please forecast the series with a horizon of {h} and frequency {freq}.\n"
        "No other details."
    )
    test_model = FunctionModel(response_agent_fn(payload={"freq": freq, "h": h}))
    exp_dataset = ExperimentDatasetParser(model=test_model).parse(
        df=df,
        freq=None,
        h=None,
        seasonality=None,
        query=query,
    )
    expected_seasonality = get_seasonality(freq)
    assert_experiment_dataset_equal(
        exp_dataset,
        ExperimentDataset(
            df=df,
            freq=freq,
            h=h,
            seasonality=expected_seasonality,
        ),
    )


def test_parse_params_no_query_infers_all():
    """With no query and no explicit params, parser infers everything from `df`."""
    freq = "MS"
    df = generate_series(
        n_series=1,  # TimeCopilot only works with one series, at this time
        freq=freq,
        min_length=24,
    )
    test_model = FunctionModel(response_agent_fn(payload={}))
    exp_dataset = ExperimentDatasetParser(model=test_model).parse(
        df=df,
        freq=None,
        h=None,
        seasonality=None,
        query=None,
    )
    expected_seasonality = get_seasonality(freq)
    expected_h = 2 * expected_seasonality
    assert_experiment_dataset_equal(
        exp_dataset,
        ExperimentDataset(
            df=df,
            freq=freq,
            h=expected_h,
            seasonality=expected_seasonality,
        ),
    )


@pytest.mark.parametrize(
    "freq,n_windows,h,step_size",
    [
        ("H", 3, 2, 1),
        ("H", 1, 12, None),
        ("MS", 3, 2, 2),
    ],
)
def test_generate_train_cv_splits(freq, n_windows, h, step_size):
    df = generate_series(n_series=5, freq=freq)
    df["unique_id"] = df["unique_id"].astype(int)
    df_cv = generate_train_cv_splits_from_backtest_splits(
        df=df,
        n_windows=n_windows,
        step_size=step_size,
        h=h,
        freq=freq,
    )
    cutoffs = df_cv[["unique_id", "cutoff"]].drop_duplicates()
    train_cv_splits = generate_train_cv_splits(
        df=df,
        cutoffs=cutoffs,
    )
    p_sort_df = partial(sort_df, cols=["unique_id", "cutoff", "ds"])
    pd.testing.assert_frame_equal(
        p_sort_df(df_cv),
        p_sort_df(train_cv_splits),
    )


@pytest.mark.parametrize("model", models)
def test_eval(model):
    freq = "H"
    exp_dataset = generate_exp_dataset(n_series=5, freq=freq)
    fcst_df = model.cross_validation(
        exp_dataset.df,
        h=exp_dataset.h,
        freq=exp_dataset.freq,
    )
    eval_df = exp_dataset.evaluate_forecast_df(
        forecast_df=fcst_df,
        models=[model.alias],
    )
    eval_df_from_scratch = evaluate_cv_from_scratch(
        df=exp_dataset.df,
        fcst_df=fcst_df,
        models=[model.alias],
        seasonality=exp_dataset.seasonality,
    )
    p_sort_df = partial(sort_df, cols=["unique_id", "cutoff", "metric"])
    pd.testing.assert_frame_equal(
        p_sort_df(eval_df),
        p_sort_df(eval_df_from_scratch),
    )
