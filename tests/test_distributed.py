"""Tests for distributed DataFrame support in TimeCopilotForecaster."""

import pandas as pd
import pytest
from utilsforecast.data import generate_series

from timecopilot.forecaster import TimeCopilotForecaster
from timecopilot.models import SeasonalNaive, ZeroModel
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
            future_ds = pd.date_range(
                start=last_ds, periods=h + 1, freq=freq
            )[1:]
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
    """Models that don't use parallel processing internally - safe for distributed tests."""
    return [SimpleTestModel()]


@pytest.fixture
def sample_df():
    df = generate_series(n_series=3, freq="D", min_length=30)
    df["unique_id"] = df["unique_id"].astype(str)
    return df


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
