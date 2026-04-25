# Time Series Model Hub


TimeCopilot provides a unified API for time series forecasting, integrating foundation models, classical statistical models, machine learning, and neural network families of models. This approach lets you experiment, benchmark, and deploy a wide range of forecasting models with minimal code changes, so you can choose the best tool for your data and use case.

Here you'll find all the time series forecasting models available in TimeCopilot, organized by family. Click on any model name to jump to its detailed API documentation.

!!! tip "Forecast multiple models using a unified API"

    With the [TimeCopilotForecaster][timecopilot.forecaster.TimeCopilotForecaster] class, you can generate and cross-validate forecasts using a unified API. Here's an example:

    ```python
    import pandas as pd
    from timecopilot.forecaster import TimeCopilotForecaster
    from timecopilot.models.prophet import Prophet
    from timecopilot.models.stats import AutoARIMA, SeasonalNaive
    from timecopilot.models.foundation.toto import Toto

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

- [Chronos](api/models/foundation/models.md#timecopilot.models.foundation.chronos) ([arXiv:2403.07815](https://arxiv.org/abs/2403.07815))
- [FlowState](api/models/foundation/models.md#timecopilot.models.foundation.flowstate) ([arXiv:2508.05287](https://arxiv.org/abs/2508.05287)) 
- [Moirai](api/models/foundation/models.md#timecopilot.models.foundation.moirai) ([arXiv:2402.02592](https://arxiv.org/abs/2402.02592))
- [PatchTST-FM](api/models/foundation/models.md#timecopilot.models.foundation.patchtst_fm) ([arXiv:2602.06909](https://arxiv.org/abs/2602.06909))
- [Sundial](api/models/foundation/models.md#timecopilot.models.foundation.sundial) ([arXiv:2502.00816](https://arxiv.org/pdf/2502.00816))
- [TabPFN](api/models/foundation/models.md#timecopilot.models.foundation.tabpfn) ([arXiv:2501.02945](https://arxiv.org/abs/2501.02945))
- [TiRex](api/models/foundation/models.md#timecopilot.models.foundation.tirex) ([arXiv:2505.23719](https://arxiv.org/abs/2505.23719))
- [TimeGPT](api/models/foundation/models.md#timecopilot.models.foundation.timegpt) ([arXiv:2310.03589](https://arxiv.org/abs/2310.03589))
- [TimesFM](api/models/foundation/models.md#timecopilot.models.foundation.timesfm) ([arXiv:2310.10688](https://arxiv.org/abs/2310.10688))
- [Toto](api/models/foundation/models.md#timecopilot.models.foundation.toto) ([arXiv:2505.14766](https://arxiv.org/abs/2505.14766))

---

## Statistical & Classical Models

TimeCopilot includes a suite of classical and statistical forecasting models, providing robust baselines and interpretable alternatives to foundation models. These models are ideal for quick benchmarking, transparent forecasting, and scenarios where simplicity and speed are paramount. Below is a list of all available statistical models, each with a dedicated section describing its API and usage.

- [ADIDA](api/models/stats.md#timecopilot.models.stats.ADIDA)
- [AutoARIMA](api/models/stats.md#timecopilot.models.stats.AutoARIMA)
- [AutoCES](api/models/stats.md#timecopilot.models.stats.AutoCES)
- [AutoETS](api/models/stats.md#timecopilot.models.stats.AutoETS)
- [CrostonClassic](api/models/stats.md#timecopilot.models.stats.CrostonClassic)
- [DynamicOptimizedTheta](api/models/stats.md#timecopilot.models.stats.DynamicOptimizedTheta)
- [HistoricAverage](api/models/stats.md#timecopilot.models.stats.HistoricAverage)
- [IMAPA](api/models/stats.md#timecopilot.models.stats.IMAPA)
- [SeasonalNaive](api/models/stats.md#timecopilot.models.stats.SeasonalNaive)
- [Theta](api/models/stats.md#timecopilot.models.stats.Theta)
- [ZeroModel](api/models/stats.md#timecopilot.models.stats.ZeroModel)


### Prophet Model

TimeCopilot integrates the popular Prophet model for time series forecasting, developed by Facebook. Prophet is well-suited for business time series with strong seasonal effects and several seasons of historical data. Below you will find the API reference for the Prophet model.


- [Prophet](api/models/prophet.md/#timecopilot.models.prophet.Prophet)

## Machine Learning Models

TimeCopilot provides access to automated machine learning models for time series forecasting. These models leverage gradient boosting and other ML techniques to automatically select features and optimize hyperparameters for your specific time series data.

- [AutoCatboost](api/models/ml.md#timecopilot.models.ml.AutoCatboost)
- [AutoElasticNet](api/models/ml.md#timecopilot.models.ml.AutoElasticNet)
- [AutoLasso](api/models/ml.md#timecopilot.models.ml.AutoLasso)
- [AutoLGBM](api/models/ml.md#timecopilot.models.ml.AutoLGBM)
- [AutoLinearRegression](api/models/ml.md#timecopilot.models.ml.AutoLinearRegression)
- [AutoRandomForest](api/models/ml.md#timecopilot.models.ml.AutoRandomForest)
- [AutoRidge](api/models/ml.md#timecopilot.models.ml.AutoRidge)
- [AutoXGBoost](api/models/ml.md#timecopilot.models.ml.AutoXGBoost)

## Neural Network Models

TimeCopilot integrates state-of-the-art neural network models for time series forecasting. These models leverage deep learning architectures specifically designed for temporal data, offering powerful capabilities for complex pattern recognition and long-range dependency modeling.

- [AutoDeepAR](api/models/neural.md#timecopilot.models.neural.AutoDeepAR)
- [AutoNBEATS](api/models/neural.md#timecopilot.models.neural.AutoNBEATS)
- [AutoNHITS](api/models/neural.md#timecopilot.models.neural.AutoNHITS)
- [AutoPatchTST](api/models/neural.md#timecopilot.models.neural.AutoPatchTST)
- [AutoTFT](api/models/neural.md#timecopilot.models.neural.AutoTFT)