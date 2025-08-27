# Time Series Model Hub


TimeCopilot provides a unified API for time series forecasting, integrating foundation models, classical statistical models, machine learning, and neural network families of models. This approach lets you experiment, benchmark, and deploy a wide range of forecasting models with minimal code changes, so you can choose the best tool for your data and use case.

Here you'll find all the time series forecasting models available in TimeCopilot, organized by family. Click on any model name to jump to its detailed API documentation.

!!! tip "Forecast multiple models using a unified API"

    With the [TimeCopilotForecaster][timecopilot.forecaster.TimeCopilotForecaster] class, you can generate and cross-validate forecasts using a unified API. Here's an example:

    ```python
    import pandas as pd
    from timecopilot.forecaster import TimeCopilotForecaster
    from timecopilot.models.benchmarks.prophet import Prophet
    from timecopilot.models.benchmarks.stats import AutoARIMA, SeasonalNaive
    from timecopilot.models.foundational.toto import Toto

    df = pd.read_csv(
        "https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv",
        parse_dates=["ds"],
    )
    tcf = TimeCopilotForecaster(
        models=[
            AutoARIMA(),
            SeasonalNaive(),
            Prophet(),
            Toto(context_length=256),
        ]
    )

    fcst_df = tcf.forecast(df=df, h=12)
    cv_df = tcf.cross_validation(df=df, h=12)
    ```

---

## Foundation Models

TimeCopilot provides a unified interface to state-of-the-art foundation models for time series forecasting. These models are designed to handle a wide range of forecasting tasks, from classical seasonal patterns to complex, high-dimensional data. Below you will find a list of all available foundation models, each with a dedicated section describing its API and usage.

- [Chronos](api/models/foundational/models.md#timecopilot.models.foundational.chronos) ([arXiv:2403.07815](https://arxiv.org/abs/2403.07815))
- [Moirai](api/models/foundational/models.md#timecopilot.models.foundational.moirai) ([arXiv:2402.02592](https://arxiv.org/abs/2402.02592))
- [Sundial](api/models/foundational/models.md#timecopilot.models.foundational.sundial) ([arXiv:2502.00816](https://arxiv.org/pdf/2502.00816))
- [TabPFN](api/models/foundational/models.md#timecopilot.models.foundational.tabpfn) ([arXiv:2501.02945](https://arxiv.org/abs/2501.02945))
- [TiRex](api/models/foundational/models.md#timecopilot.models.foundational.tirex) ([arXiv:2505.23719](https://arxiv.org/abs/2505.23719))
- [TimeGPT](api/models/foundational/models.md#timecopilot.models.foundational.timegpt) ([arXiv:2310.03589](https://arxiv.org/abs/2310.03589))
- [TimesFM](api/models/foundational/models.md#timecopilot.models.foundational.timesfm) ([arXiv:2310.10688](https://arxiv.org/abs/2310.10688))
- [Toto](api/models/foundational/models.md#timecopilot.models.foundational.toto) ([arXiv:2505.14766](https://arxiv.org/abs/2505.14766))

---

## Statistical & Classical Models

TimeCopilot includes a suite of classical and statistical forecasting models, providing robust baselines and interpretable alternatives to foundation models. These models are ideal for quick benchmarking, transparent forecasting, and scenarios where simplicity and speed are paramount. Below is a list of all available statistical models, each with a dedicated section describing its API and usage.

- [ADIDA](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.ADIDA)
- [AutoARIMA](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.AutoARIMA)
- [AutoCES](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.AutoCES)
- [AutoETS](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.AutoETS)
- [CrostonClassic](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.CrostonClassic)
- [DynamicOptimizedTheta](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.DynamicOptimizedTheta)
- [HistoricAverage](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.HistoricAverage)
- [IMAPA](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.IMAPA)
- [SeasonalNaive](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.SeasonalNaive)
- [Theta](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.Theta)
- [ZeroModel](api/models/benchmarks/stats.md#timecopilot.models.benchmarks.stats.ZeroModel)


### Prophet Model

TimeCopilot integrates the popular Prophet model for time series forecasting, developed by Facebook. Prophet is well-suited for business time series with strong seasonal effects and several seasons of historical data. Below you will find the API reference for the Prophet model.


- [Prophet](api/models/benchmarks/prophet.md/#timecopilot.models.benchmarks.prophet.Prophet)

## Machine Learning Models

TimeCopilot provides access to automated machine learning models for time series forecasting. These models leverage gradient boosting and other ML techniques to automatically select features and optimize hyperparameters for your specific time series data.

- [AutoLGBM](api/models/benchmarks/ml.md#timecopilot.models.benchmarks.ml.AutoLGBM)

## Neural Network Models

TimeCopilot integrates state-of-the-art neural network models for time series forecasting. These models leverage deep learning architectures specifically designed for temporal data, offering powerful capabilities for complex pattern recognition and long-range dependency modeling.

- [AutoNHITS](api/models/benchmarks/neural.md#timecopilot.models.benchmarks.neural.AutoNHITS)
- [AutoTFT](api/models/benchmarks/neural.md#timecopilot.models.benchmarks.neural.AutoTFT)