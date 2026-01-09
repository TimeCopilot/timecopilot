"""
Test that the agent works with a live LLM.
Keeping it separate from the other tests because costs and requires a live LLM.
"""

import logfire
import pytest
from dotenv import load_dotenv
from utilsforecast.data import generate_series

from timecopilot import TimeCopilot
from timecopilot.agent import AsyncTimeCopilot
from timecopilot.models.stats import SeasonalNaive, ZeroModel

load_dotenv()
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()

default_forecasters = [ZeroModel(), SeasonalNaive()]


@pytest.mark.live
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_forecast_custom_forecasters():
    h = 2
    df = generate_series(
        n_series=1,
        freq="D",
        min_length=30,
        static_as_categorical=False,
        with_trend=True,
    )
    tc = TimeCopilot(
        llm="openai:gpt-4o-mini",
        forecasters=[
            ZeroModel(),
        ],
    )
    result = tc.forecast(
        df=df,
        query=f"Please forecast the series with a horizon of {h} and frequency D.",
    )
    assert len(result.fcst_df) == h
    assert result.features_df is not None
    assert result.eval_df is not None


@pytest.mark.live
@pytest.mark.parametrize("n_series", [1, 2])
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_forecast_returns_expected_output(n_series):
    h = 2
    df = generate_series(
        n_series=n_series,
        freq="D",
        min_length=30,
        static_as_categorical=False,
    )
    tc = TimeCopilot(
        llm="openai:gpt-4o-mini",
        forecasters=default_forecasters,
        retries=3,
    )
    result = tc.forecast(
        df=df,
        query=f"Please forecast the series with a horizon of {h} and frequency D.",
    )
    assert len(result.fcst_df) == n_series * h
    assert result.features_df is not None
    assert result.eval_df is not None
    assert result.output.is_better_than_seasonal_naive
    assert result.output.forecast_analysis is not None
    assert result.output.reason_for_selection is not None


@pytest.mark.live
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_is_queryable():
    h = 2
    df = generate_series(
        n_series=2,
        freq="D",
        min_length=30,
        static_as_categorical=False,
    )
    tc = TimeCopilot(
        llm="openai:gpt-4o-mini",
        retries=3,
    )
    assert not tc.is_queryable()
    tc.forecast(
        df=df,
        query=f"Please forecast the series with a horizon of {h} and frequency D.",
    )
    assert tc.is_queryable()
    series_id = df["unique_id"].iloc[0]
    result = tc.query(f"how much will change the series with id {series_id}?")
    print(result.output)


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("n_series", [1, 2])
@pytest.mark.flaky(reruns=3, reruns_delay=2)
async def test_async_forecast_returns_expected_output(n_series):
    h = 2
    df = generate_series(
        n_series=n_series,
        freq="D",
        min_length=30,
        static_as_categorical=False,
    )
    tc = AsyncTimeCopilot(
        llm="openai:gpt-4o-mini",
        forecasters=default_forecasters,
        retries=3,
    )
    result = await tc.forecast(
        df=df,
        query=f"Please forecast the series with a horizon of {h} and frequency D.",
    )
    assert len(result.fcst_df) == n_series * h
    assert result.features_df is not None
    assert result.eval_df is not None
    assert result.output.is_better_than_seasonal_naive
    assert result.output.forecast_analysis is not None
    assert result.output.reason_for_selection is not None


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.flaky(reruns=3, reruns_delay=2)
async def test_async_is_queryable():
    h = 2
    df = generate_series(
        n_series=2,
        freq="D",
        min_length=30,
        static_as_categorical=False,
    )
    tc = AsyncTimeCopilot(
        llm="openai:gpt-4o-mini",
        forecasters=default_forecasters,
        retries=3,
    )
    assert not tc.is_queryable()
    await tc.forecast(
        df=df,
        query=f"Please forecast the series with a horizon of {h} and frequency D.",
    )
    assert tc.is_queryable()
    series_id = df["unique_id"].iloc[0]
    answer = await tc.query(f"how much will change the series with id {series_id}?")
    print(answer.output)


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.flaky(reruns=3, reruns_delay=2)
async def test_async_query_stream():
    h = 2
    df = generate_series(
        n_series=1,
        freq="D",
        min_length=30,
        static_as_categorical=False,
    )
    tc = AsyncTimeCopilot(
        llm="openai:gpt-4o-mini",
        forecasters=default_forecasters,
        retries=3,
    )
    await tc.forecast(
        df=df,
        query=f"Please forecast the series with a horizon of {h} and frequency D.",
    )
    async with tc.query_stream("What is the forecast for the next 2 days?") as result:
        # This will yield a StreamedRunResult, which can be streamed for text
        async for text in result.stream(debounce_by=0.01):
            print(text, end="", flush=True)
