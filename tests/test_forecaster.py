import pytest
from utilsforecast.data import generate_series

from timecopilot.forecaster import TimeCopilotForecaster
from timecopilot.models import SeasonalNaive, ZeroModel
from timecopilot.models.foundation.moirai import Moirai


@pytest.fixture
def models():
    return [SeasonalNaive(), ZeroModel()]


@pytest.mark.parametrize(
    "freq,h",
    [
        ("D", 2),
        ("W-MON", 3),
    ],
)
def test_forecaster_forecast(models, freq, h):
    n_uids = 3
    df = generate_series(n_series=n_uids, freq=freq, min_length=30)
    forecaster = TimeCopilotForecaster(models=models)
    fcst_df = forecaster.forecast(df=df, h=h, freq=freq)
    assert len(fcst_df.columns) == 2 + len(models)
    assert len(fcst_df) == h * n_uids
    for model in models:
        assert model.alias in fcst_df.columns


@pytest.mark.parametrize(
    "freq,h,n_windows,step_size",
    [
        ("D", 2, 2, 1),
        ("W-MON", 3, 2, 2),
    ],
)
def test_forecaster_cross_validation(models, freq, h, n_windows, step_size):
    n_uids = 3
    df = generate_series(n_series=n_uids, freq=freq, min_length=30)
    forecaster = TimeCopilotForecaster(models=models)
    fcst_df = forecaster.cross_validation(
        df=df,
        h=h,
        freq=freq,
        n_windows=n_windows,
        step_size=step_size,
    )
    assert len(fcst_df.columns) == 4 + len(models)
    uids = df["unique_id"].unique()
    for uid in uids:  # noqa: B007
        fcst_df_uid = fcst_df.query("unique_id == @uid")
        assert fcst_df_uid["cutoff"].nunique() == n_windows
        assert len(fcst_df_uid) == n_windows * h
    for model in models:
        assert model.alias in fcst_df.columns


def test_forecaster_forecast_with_level(models):
    n_uids = 3
    level = [80, 90]
    df = generate_series(n_series=n_uids, freq="D", min_length=30)
    forecaster = TimeCopilotForecaster(models=models)
    fcst_df = forecaster.forecast(df=df, h=2, freq="D", level=level)  # type: ignore
    assert len(fcst_df) == 2 * n_uids
    assert len(fcst_df.columns) == 2 + len(models) * (1 + 2 * len(level))
    for model in models:
        assert model.alias in fcst_df.columns
        for lv in level:
            assert f"{model.alias}-lo-{lv}" in fcst_df.columns
            assert f"{model.alias}-hi-{lv}" in fcst_df.columns


def test_forecaster_forecast_with_quantiles(models):
    n_uids = 3
    quantiles = [0.1, 0.9]
    df = generate_series(n_series=n_uids, freq="D", min_length=30)
    forecaster = TimeCopilotForecaster(models=models)
    fcst_df = forecaster.forecast(df=df, h=2, freq="D", quantiles=quantiles)
    assert len(fcst_df) == 2 * n_uids
    assert len(fcst_df.columns) == 2 + len(models) * (1 + len(quantiles))
    for model in models:
        assert model.alias in fcst_df.columns
        for q in quantiles:
            assert f"{model.alias}-q-{int(100 * q)}" in fcst_df.columns


def test_forecaster_fallback_model():
    from timecopilot.models.utils.forecaster import Forecaster

    class FailingModel(Forecaster):
        alias = "FailingModel"

        def forecast(self, df, h, freq=None, level=None, quantiles=None):
            raise RuntimeError("Intentional failure")

    class DummyModel(Forecaster):
        alias = "DummyModel"

        def forecast(self, df, h, freq=None, level=None, quantiles=None):
            # Return a DataFrame with the expected columns
            import pandas as pd

            n = len(df["unique_id"].unique()) * h
            return pd.DataFrame(
                {
                    "unique_id": ["A"] * n,
                    "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
                    "DummyModel": range(n),
                }
            )

    df = generate_series(n_series=1, freq="D", min_length=10)
    forecaster = TimeCopilotForecaster(
        models=[FailingModel()],
        fallback_model=DummyModel(),
    )
    fcst_df = forecaster.forecast(df=df, h=2, freq="D")
    # Should use DummyModel's output
    # and rename the columns to the original model's alias
    assert "FailingModel" in fcst_df.columns
    assert "DummyModel" not in fcst_df.columns
    assert len(fcst_df) == 2


def test_forecaster_no_fallback_raises():
    from timecopilot.models.utils.forecaster import Forecaster

    class FailingModel(Forecaster):
        alias = "FailingModel"

        def forecast(self, df, h, freq=None, level=None, quantiles=None):
            raise RuntimeError("Intentional failure")

    df = generate_series(n_series=1, freq="D", min_length=10)
    forecaster = TimeCopilotForecaster(models=[FailingModel()])
    with pytest.raises(RuntimeError, match="Intentional failure"):
        forecaster.forecast(df=df, h=2, freq="D")


def test_duplicate_aliases_raises_error():
    """Test that TimeCopilotForecaster raises error with duplicate aliases."""
    # Create two models with the same alias
    model1 = Moirai(repo_id="Salesforce/moirai-1.0-R-small", alias="Moirai")
    model2 = Moirai(repo_id="Salesforce/moirai-1.0-R-large", alias="Moirai")

    with pytest.raises(
        ValueError, match="Duplicate model aliases found: \\['Moirai'\\]"
    ):
        TimeCopilotForecaster(models=[model1, model2])


def test_unique_aliases_works():
    """Test that TimeCopilotForecaster works when models have unique aliases."""
    # Create two models with different aliases
    model1 = Moirai(repo_id="Salesforce/moirai-1.0-R-small", alias="MoiraiSmall")
    model2 = Moirai(repo_id="Salesforce/moirai-1.0-R-large", alias="MoiraiLarge")

    # This should not raise an error
    forecaster = TimeCopilotForecaster(models=[model1, model2])
    assert len(forecaster.models) == 2
    assert forecaster.models[0].alias == "MoiraiSmall"
    assert forecaster.models[1].alias == "MoiraiLarge"


def test_mixed_models_unique_aliases():
    """Test that different model classes with unique aliases work together."""
    model1 = SeasonalNaive()
    model2 = ZeroModel()
    model3 = Moirai(repo_id="Salesforce/moirai-1.0-R-small", alias="MoiraiTest")

    # This should not raise an error
    forecaster = TimeCopilotForecaster(models=[model1, model2, model3])
    assert len(forecaster.models) == 3


def test_clean_cache_runs_after_each_model(monkeypatch, models):
    calls = []

    monkeypatch.setattr(
        TimeCopilotForecaster,
        "_clean_model_cache",
        staticmethod(lambda: calls.append("cleaned")),
    )

    df = generate_series(n_series=1, freq="D", min_length=10)
    forecaster = TimeCopilotForecaster(models=models, clean_cache=True)

    forecaster.forecast(df=df, h=2, freq="D")

    assert calls == ["cleaned"] * len(models)
