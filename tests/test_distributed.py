"""Tests for distributed DataFrame support in TimeCopilotForecaster."""


import pandas as pd
import pytest
from utilsforecast.data import generate_series

from timecopilot.forecaster import TimeCopilotForecaster
from timecopilot.models import SeasonalNaive, ZeroModel
from timecopilot.models.foundation.chronos import Chronos
from timecopilot.models.utils.forecaster import Forecaster


class SimpleTestModel(Forecaster):
    """A simple model for testing that doesn't use any parallel processing."""

    alias = "SimpleTestModel"

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        """Return zeros for all forecasts."""
        freq = self._maybe_infer_freq(df, freq)
        unique_ids = df["unique_id"].unique()
        results = []
        for uid in unique_ids:
            uid_df = df[df["unique_id"] == uid]
            last_ds = uid_df["ds"].max()
            future_ds = pd.date_range(start=last_ds, periods=h + 1, freq=freq)[1:]
            results.append(
                pd.DataFrame(
                    {
                        "unique_id": [uid] * h,
                        "ds": future_ds,
                        self.alias: [0.0] * h,
                    }
                )
            )
        return pd.concat(results, ignore_index=True)


@pytest.fixture
def models():
    return [SeasonalNaive(), ZeroModel()]


@pytest.fixture
def simple_models():
    """
    Models that don't use parallel processing internally - safe for
    distributed tests.
    """
    return [SimpleTestModel()]


@pytest.fixture
def foundation_model():
    # return [Chronos(repo_id="autogluon/chronos-bolt-tiny")]
    return [Chronos(repo_id="autogluon/chronos-t5-tiny")]


@pytest.fixture
def sample_df():
    df = generate_series(n_series=3, freq="D", min_length=30)
    df["unique_id"] = df["unique_id"].astype(str)
    return df


@pytest.fixture
def event_df():
    return pd.read_csv(
        "https://timecopilot.s3.amazonaws.com/public/data/events_pageviews.csv",
        parse_dates=["ds"],
    )


# --- Type detection tests ---


def test_is_distributed_df_pandas(sample_df):
    assert TimeCopilotForecaster._is_distributed_df(sample_df) is False


# --- Pandas path tests (baseline) ---


def test_forecast_pandas(models, sample_df):
    forecaster = TimeCopilotForecaster(models=models)
    result = forecaster.forecast(df=sample_df, h=2, freq="D")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2 * 3
    for model in models:
        assert model.alias in result.columns


def test_cross_validation_pandas(models, sample_df):
    forecaster = TimeCopilotForecaster(models=models)
    result = forecaster.cross_validation(
        df=sample_df, h=2, freq="D", n_windows=2, step_size=1
    )
    assert isinstance(result, pd.DataFrame)
    assert "cutoff" in result.columns
    for model in models:
        assert model.alias in result.columns


def test_detect_anomalies_pandas(models, sample_df):
    forecaster = TimeCopilotForecaster(models=models)
    result = forecaster.detect_anomalies(df=sample_df, h=2, freq="D")
    assert isinstance(result, pd.DataFrame)
    assert "cutoff" in result.columns
    for model in models:
        assert model.alias in result.columns
        assert f"{model.alias}-anomaly" in result.columns


# --- Spark tests ---


@pytest.fixture
def spark_session():
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("timecopilot-test")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture
def spark_df(spark_session, sample_df):
    return spark_session.createDataFrame(sample_df)


@pytest.mark.distributed
def test_is_distributed_df_spark(spark_df):
    assert TimeCopilotForecaster._is_distributed_df(spark_df) is True


@pytest.mark.distributed
def test_forecast_spark(simple_models, spark_df):
    forecaster = TimeCopilotForecaster(models=simple_models)
    result = forecaster.forecast(df=spark_df, h=2, freq="D")
    # Result should be a Spark DataFrame
    from pyspark.sql import DataFrame as SparkDataFrame

    assert isinstance(result, SparkDataFrame)
    result_pd = result.toPandas()
    assert len(result_pd) == 2 * 3
    for model in simple_models:
        assert model.alias in result_pd.columns


@pytest.mark.distributed
def test_cross_validation_spark(simple_models, spark_df):
    forecaster = TimeCopilotForecaster(models=simple_models)
    result = forecaster.cross_validation(
        df=spark_df, h=2, freq="D", n_windows=2, step_size=1
    )
    from pyspark.sql import DataFrame as SparkDataFrame

    assert isinstance(result, SparkDataFrame)
    result_pd = result.toPandas()
    assert "cutoff" in result_pd.columns
    for model in simple_models:
        assert model.alias in result_pd.columns


@pytest.mark.distributed
def test_using_level_spark(foundation_model, spark_session):
    # level = [0, 20, 40, 60, 80]  # corresponds to qs [0.1, 0.2, ..., 0.9]
    level: list[int | float] = [20, 80]
    df = generate_series(
        n_series=2, freq="D", max_length=100, static_as_categorical=False
    )
    spark_df = spark_session.createDataFrame(df)
    tcf = TimeCopilotForecaster(models=foundation_model)
    excluded_models = [
        "AutoLGBM",
        "AutoNHITS",
        "AutoTFT",
        "PatchTST-FM",
    ]
    if any(m.alias in excluded_models for m in foundation_model):
        # These models do not support levels yet
        with pytest.raises(ValueError) as excinfo:
            tcf.forecast(
                df=spark_df,
                h=2,
                freq="D",
                level=level,
            )
        assert "not supported" in str(excinfo.value)
        return
    fcst_df = tcf.forecast(
        df=spark_df,
        h=2,
        freq="D",
        level=level,
    )
    fcst_df_pd = fcst_df.toPandas()
    exp_lv_cols = []
    for lv in level:
        for model in foundation_model:
            exp_lv_cols.extend([f"{model.alias}-lo-{lv}", f"{model.alias}-hi-{lv}"])
    assert len(exp_lv_cols) == len(fcst_df_pd.columns) - 3  # 3 is unique_id, ds, point
    assert all(col in fcst_df_pd.columns for col in exp_lv_cols)
    assert not any(("-q-" in col) for col in fcst_df_pd.columns)
    # test monotonicity of levels
    exp_lv_cols = exp_lv_cols[2:]  # remove level 0
    for c1, c2 in zip(exp_lv_cols[:-1:2], exp_lv_cols[1::2], strict=False):
        for model in foundation_model:
            if model.alias == "ZeroModel":
                # ZeroModel is a constant model, so all levels should be the same
                assert fcst_df_pd[c1].eq(fcst_df_pd[c2]).all()
            elif "chronos" in model.alias.lower() or "median" in model.alias.lower():
                # sometimes it gives this condition
                assert fcst_df_pd[c1].le(fcst_df_pd[c2]).all()
            elif "tabpfn" in model.alias.lower():
                # we are testing the mock mode, so we don't care about monotonicity
                continue
            else:
                assert fcst_df_pd[c1].lt(fcst_df_pd[c2]).all()


# --- Dask tests ---


@pytest.fixture
def dask_df(sample_df):
    pytest.importorskip("dask")
    import dask.dataframe as dd

    return dd.from_pandas(sample_df, npartitions=2)


@pytest.mark.distributed
def test_is_distributed_df_dask(dask_df):
    assert TimeCopilotForecaster._is_distributed_df(dask_df) is True


@pytest.mark.distributed
def test_forecast_dask(simple_models, dask_df):
    import dask.dataframe as dd

    forecaster = TimeCopilotForecaster(models=simple_models)
    result = forecaster.forecast(df=dask_df, h=2, freq="D")
    # Result should be a Dask DataFrame
    assert isinstance(result, dd.DataFrame)
    result_pd = result.compute()
    assert len(result_pd) == 2 * 3
    for model in simple_models:
        assert model.alias in result_pd.columns


@pytest.mark.distributed
@pytest.mark.parametrize(
    "n_series,n_partitions",
    [
        (1, 1),
        (3, 1),
        (1, 2),
        (3, 2),
    ],
)
def test_forecast_dask_series(n_series, n_partitions, foundation_model):
    pytest.importorskip("dask")
    import dask.dataframe as dd

    # models = foundation_model()
    models = foundation_model
    h = 2

    df = generate_series(n_series=n_series, freq="D", min_length=30)
    df["unique_id"] = df["unique_id"].astype(str)

    dask_df = dd.from_pandas(df, npartitions=n_partitions)
    forecaster = TimeCopilotForecaster(models=models)
    result = forecaster.forecast(df=dask_df, h=h, freq="D")
    result_pd = result.compute()
    assert len(result_pd) == h * n_series
    for model in models:
        assert model.alias in result_pd.columns


@pytest.mark.distributed
def test_cross_validation_dask(simple_models, dask_df):
    import dask.dataframe as dd

    forecaster = TimeCopilotForecaster(models=simple_models)
    result = forecaster.cross_validation(
        df=dask_df, h=2, freq="D", n_windows=2, step_size=1
    )
    assert isinstance(result, dd.DataFrame)
    result_pd = result.compute()
    assert "cutoff" in result_pd.columns
    for model in simple_models:
        assert model.alias in result_pd.columns


@pytest.mark.distributed
def test_using_level_dask(foundation_model):
    import dask.dataframe as dd

    # level = [0, 20, 40, 60, 80]  # corresponds to qs [0.1, 0.2, ..., 0.9]
    level: list[int | float] = [20, 80]
    n_partitions = 2
    df = generate_series(
        n_series=2, freq="D", max_length=100, static_as_categorical=False
    )
    dask_df = dd.from_pandas(df, npartitions=n_partitions)
    tcf = TimeCopilotForecaster(models=foundation_model)
    excluded_models = [
        "AutoLGBM",
        "AutoNHITS",
        "AutoTFT",
        "PatchTST-FM",
    ]
    if any(m.alias in excluded_models for m in foundation_model):
        # These models do not support levels yet
        with pytest.raises(ValueError) as excinfo:
            tcf.forecast(
                df=dask_df,
                h=2,
                freq="D",
                level=level,
            )
        assert "not supported" in str(excinfo.value)
        return
    fcst_df = tcf.forecast(
        df=dask_df,
        h=2,
        freq="D",
        level=level,
    )
    fcst_df_pd = fcst_df.compute()
    exp_lv_cols = []
    for lv in level:
        for model in foundation_model:
            exp_lv_cols.extend([f"{model.alias}-lo-{lv}", f"{model.alias}-hi-{lv}"])
    assert len(exp_lv_cols) == len(fcst_df_pd.columns) - 3  # 3 is unique_id, ds, point
    assert all(col in fcst_df_pd.columns for col in exp_lv_cols)
    assert not any(("-q-" in col) for col in fcst_df_pd.columns)
    # test monotonicity of levels
    exp_lv_cols = exp_lv_cols[2:]  # remove level 0
    for c1, c2 in zip(exp_lv_cols[:-1:2], exp_lv_cols[1::2], strict=False):
        for model in foundation_model:
            if model.alias == "ZeroModel":
                # ZeroModel is a constant model, so all levels should be the same
                assert fcst_df_pd[c1].eq(fcst_df_pd[c2]).all()
            elif "chronos" in model.alias.lower() or "median" in model.alias.lower():
                # sometimes it gives this condition
                assert fcst_df_pd[c1].le(fcst_df_pd[c2]).all()
            elif "tabpfn" in model.alias.lower():
                # we are testing the mock mode, so we don't care about monotonicity
                continue
            else:
                assert fcst_df_pd[c1].lt(fcst_df_pd[c2]).all()


# --- Ray tests ---


@pytest.fixture
def ray_dataset(sample_df):
    pytest.importorskip("ray")
    import ray

    if not ray.is_initialized():
        # Use local mode to avoid working directory upload issues
        ray.init(
            ignore_reinit_error=True,
            num_cpus=2,
            include_dashboard=False,
            runtime_env={"working_dir": None},
        )
    return ray.data.from_pandas(sample_df)


@pytest.mark.distributed
def test_is_distributed_df_ray(ray_dataset):
    assert TimeCopilotForecaster._is_distributed_df(ray_dataset) is True


@pytest.mark.distributed
def test_forecast_ray(simple_models, ray_dataset):
    import ray.data

    forecaster = TimeCopilotForecaster(models=simple_models)
    result = forecaster.forecast(df=ray_dataset, h=2, freq="D")
    # Result should be a Ray Dataset
    assert isinstance(result, ray.data.Dataset)
    result_pd = result.to_pandas()
    assert len(result_pd) == 2 * 3
    for model in simple_models:
        assert model.alias in result_pd.columns


@pytest.mark.distributed
def test_cross_validation_ray(simple_models, ray_dataset):
    import ray.data

    forecaster = TimeCopilotForecaster(models=simple_models)
    result = forecaster.cross_validation(
        df=ray_dataset, h=2, freq="D", n_windows=2, step_size=1
    )
    assert isinstance(result, ray.data.Dataset)
    result_pd = result.to_pandas()
    assert "cutoff" in result_pd.columns
    for model in simple_models:
        assert model.alias in result_pd.columns


@pytest.mark.distributed
def test_using_level_ray(foundation_model):
    import ray

    # level = [0, 20, 40, 60, 80]  # corresponds to qs [0.1, 0.2, ..., 0.9]
    level: list[int | float] = [20, 80]
    df = generate_series(
        n_series=2, freq="D", max_length=100, static_as_categorical=False
    )
    if not ray.is_initialized():
        # Use local mode to avoid working directory upload issues
        ray.init(
            ignore_reinit_error=True,
            num_cpus=2,
            include_dashboard=False,
            runtime_env={"working_dir": None},
            object_store_memory=78_643_200,
        )
    ray_df = ray.data.from_pandas(df)
    tcf = TimeCopilotForecaster(models=foundation_model)
    excluded_models = [
        "AutoLGBM",
        "AutoNHITS",
        "AutoTFT",
        "PatchTST-FM",
    ]
    if any(m.alias in excluded_models for m in foundation_model):
        # These models do not support levels yet
        with pytest.raises(ValueError) as excinfo:
            tcf.forecast(
                df=ray_df,
                h=2,
                freq="D",
                level=level,
            )
        assert "not supported" in str(excinfo.value)
        return
    fcst_df = tcf.forecast(
        df=ray_df,
        h=2,
        freq="D",
        level=level,
    )
    fcst_df_pd = fcst_df.to_pandas()
    exp_lv_cols = []
    for lv in level:
        for model in foundation_model:
            exp_lv_cols.extend([f"{model.alias}-lo-{lv}", f"{model.alias}-hi-{lv}"])
    assert len(exp_lv_cols) == len(fcst_df_pd.columns) - 3  # 3 is unique_id, ds, point
    assert all(col in fcst_df_pd.columns for col in exp_lv_cols)
    assert not any(("-q-" in col) for col in fcst_df_pd.columns)
    # test monotonicity of levels
    exp_lv_cols = exp_lv_cols[2:]  # remove level 0
    for c1, c2 in zip(exp_lv_cols[:-1:2], exp_lv_cols[1::2], strict=False):
        for model in foundation_model:
            if model.alias == "ZeroModel":
                # ZeroModel is a constant model, so all levels should be the same
                assert fcst_df_pd[c1].eq(fcst_df_pd[c2]).all()
            elif "chronos" in model.alias.lower() or "median" in model.alias.lower():
                # sometimes it gives this condition
                assert fcst_df_pd[c1].le(fcst_df_pd[c2]).all()
            elif "tabpfn" in model.alias.lower():
                # we are testing the mock mode, so we don't care about monotonicity
                continue
            else:
                assert fcst_df_pd[c1].lt(fcst_df_pd[c2]).all()


# --- num_partitions parameter tests ---


@pytest.mark.distributed
def test_forecast_spark_with_num_partitions(simple_models, spark_df):
    forecaster = TimeCopilotForecaster(models=simple_models)
    result = forecaster.forecast(df=spark_df, h=2, freq="D", num_partitions=4)
    from pyspark.sql import DataFrame as SparkDataFrame

    assert isinstance(result, SparkDataFrame)
    result_pd = result.toPandas()
    assert len(result_pd) == 2 * 3


@pytest.mark.distributed
def test_forecast_dask_with_num_partitions(simple_models, dask_df):
    import dask.dataframe as dd

    forecaster = TimeCopilotForecaster(models=simple_models)
    result = forecaster.forecast(df=dask_df, h=2, freq="D", num_partitions=4)
    assert isinstance(result, dd.DataFrame)
    result_pd = result.compute()
    assert len(result_pd) == 2 * 3
