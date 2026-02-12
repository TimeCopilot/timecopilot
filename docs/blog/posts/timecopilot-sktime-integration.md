---
date: 2026-02-07
authors:
  - khuyentran
categories:
  - General
description: >
  Learn how to extend TimeCopilot with 200+ sktime forecasting models, combine them with foundation models, and run unified cross-validation.
title: "TimeCopilot + sktime: Access 200+ Additional Forecasting Models"
slug: "timecopilot-sktime-integration"
---

# TimeCopilot + sktime: Access 200+ Additional Forecasting Models

Choosing the right forecasting model typically requires testing multiple approaches and comparing results manually. TimeCopilot simplifies this by combining state-of-the-art foundation models (Chronos, TimesFM, MOIRAI) and ML, deep learning, and statistical models with automated cross-validation and LLM-powered analysis.

But what if you need capabilities beyond the defaults? The sktime integration opens up 200+ additional forecasters—from complex seasonality models to sklearn regressors.

In this article, you'll learn how to:

- Add sktime forecasters to TimeCopilot with a single line
- Combine sktime models with foundation models like Chronos
- Run unified cross-validation across all models

## What is sktime?

sktime is a unified Python framework for time series machine learning. It provides a consistent, scikit-learn-style API across 200+ forecasting models (with classification and other tasks coming soon), including:

- Statistical models (ARIMA, ETS, Theta)
- Complex seasonality (BATS, TBATS)
- Multivariate methods (VAR, VECM)
- Neural networks and ensembles
- sklearn regressors as forecasters
- And more

All models use the familiar scikit-learn `fit`/`predict` interface:

```python
from sktime.forecasting.trend import TrendForecaster
from sktime.datasets import load_airline

y = load_airline()
forecaster = TrendForecaster()
forecaster.fit(y)
y_pred = forecaster.predict(fh=[1, 2, 3])
```

This familiar interface makes sktime easy to learn if you know scikit-learn.

## Setup

Start by installing the required packages:

```bash
pip install timecopilot sktime
```

*This article uses timecopilot v0.0.23 and sktime v0.40.1*

Load the sample dataset:

```python
import pandas as pd

# Load air passengers dataset (TimeCopilot format: unique_id, ds, y)
df = pd.read_csv(
    "https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv",
    parse_dates=["ds"],
)

print(df.head())
```

```text
  unique_id          ds      y
0  AirPassengers  1949-01-01  112.0
1  AirPassengers  1949-02-01  118.0
2  AirPassengers  1949-03-01  132.0
3  AirPassengers  1949-04-01  129.0
4  AirPassengers  1949-05-01  121.0
```

For notebooks only, enable async support:

```python
import nest_asyncio

nest_asyncio.apply()
```

## Adding sktime Models

Create an sktime forecaster and pass it to TimeCopilot:

```python
import timecopilot
from sktime.forecasting.trend import TrendForecaster

# Create sktime model
trend_forecaster = TrendForecaster()

# Create agent with combined models
tc = timecopilot.TimeCopilot(
    llm="openai:gpt-4o",
    forecasters=[trend_forecaster],
)
```

Generate a forecast:

```python
result = tc.forecast(df=df)
```

Access the LLM analysis of time series features:

```python
print(result.output.tsfeatures_analysis)
```

```text
The time series for 'AirPassengers' exhibits a strong seasonal component with a period of 12 months, as indicated by the 'seasonal_period' feature and high 'seasonal_strength'. The 'unitroot_kpss' test suggests that the series is non-stationary, typically requiring differencing before modeling. The 'entropy' value indicates moderate unpredictability, while high 'trend' suggests a persistent upward direction. Initial 'hw_parameters' suggest Holt-Winters' seasonal smoothing potential, while 'acf_features' show significant autocorrelation that models should address.
```

Access the forecast DataFrame:

```python
result.fcst_df.head()
```

```text
       unique_id         ds  sktime.TrendForecaster
0  AirPassengers 1961-01-01              473.023018
1  AirPassengers 1961-02-01              475.729097
2  AirPassengers 1961-03-01              478.173296
3  AirPassengers 1961-04-01              480.879374
4  AirPassengers 1961-05-01              483.498159
```

## Custom Aliases with SKTimeAdapter

If you want custom names instead of the auto-generated aliases (like `sktime.TrendForecaster`), use `SKTimeAdapter`:

```python
from timecopilot.models.adapters.sktime import SKTimeAdapter

# Wrap with custom alias
manually_adapted_model = SKTimeAdapter(
    model=TrendForecaster(),
    alias="TrendForecaster",
)

tc = timecopilot.TimeCopilot(
    llm="openai:gpt-4o",
    forecasters=[manually_adapted_model],
)
```

The `alias` parameter sets the model name that appears in forecast results and comparisons:

```python
result = tc.forecast(df=df)

result.fcst_df.head()
```

```text
      unique_id         ds  TrendForecaster
0  AirPassengers 1961-01-01       473.023018
1  AirPassengers 1961-02-01       475.729097
2  AirPassengers 1961-03-01       478.173296
3  AirPassengers 1961-04-01       480.879374
4  AirPassengers 1961-05-01       483.498159
```

## Cross-Validation Comparison

With TimeCopilot, you can easily compare your sktime model against other forecasters and find the best fit for your data:

```python
from timecopilot import TimeCopilotForecaster
from timecopilot.models.stats import AutoARIMA, AutoETS

# Wrap sktime model with SKTimeAdapter
trend = SKTimeAdapter(
    model=TrendForecaster(),
    alias="TrendForecaster"
)

# Create forecaster with sktime + statsforecast models
forecaster = TimeCopilotForecaster(
    models=[trend, AutoARIMA(), AutoETS()]
)

# Run cross-validation
cv_results = forecaster.cross_validation(
    df=df,
    h=12,           # Forecast horizon: 12 months
    n_windows=3     # Number of CV folds
)
```

Use `utilsforecast` to calculate metrics from the cross-validation results:

```python
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse, mape

eval_df = evaluate(
    cv_results.drop(columns=["cutoff"]),
    metrics=[mae, rmse, mape],
)
print(eval_df)
```

```text
       unique_id metric  TrendForecaster  AutoARIMA    AutoETS
0  AirPassengers    mae        52.926293  27.824331  32.821513
1  AirPassengers   rmse        69.589829  33.140656  40.355791
2  AirPassengers   mape         0.117029   0.067294   0.072262
```

The results show MAE, RMSE, and MAPE for each model across all cross-validation windows, making it easy to identify the best performer.

Generate forecasts with all models:

```python
# Forecast with all models
fcst_df = forecaster.forecast(df=df, h=12)

print(fcst_df.head())
```

```text
       unique_id         ds       Trend   AutoARIMA     AutoETS
0  AirPassengers 1961-01-01  473.023018  444.309570  442.357178
1  AirPassengers 1961-02-01  475.729097  418.213745  428.267365
2  AirPassengers 1961-03-01  478.173296  446.243408  492.974792
3  AirPassengers 1961-04-01  480.879374  488.234222  477.369995
4  AirPassengers 1961-05-01  483.498159  499.237061  477.602814
```

## Extending the Default Model List

To include your sktime model alongside TimeCopilot's default models (Chronos, AutoARIMA, etc.), extend the `DEFAULT_MODELS` list:

```python
# Copy default models and add sktime model
model_list = timecopilot.agent.DEFAULT_MODELS.copy()
model_list.append(TrendForecaster())

# Create agent with extended model list
tc = timecopilot.TimeCopilot(
    llm="openai:gpt-4o",
    forecasters=model_list,
)
```

This approach lets TimeCopilot evaluate your sktime model against its full suite of foundation and statistical models.

## Next Steps

The sktime integration is one of several ways to extend TimeCopilot. Explore other integrations:

- **[Chronos Family](https://timecopilot.dev/examples/chronos-family/)**: Configure Chronos-Bolt and original Chronos models
- **[Google LLMs](https://timecopilot.dev/examples/google-llms/)**: Use Gemini as the reasoning backend
- **[AWS Bedrock](https://timecopilot.dev/examples/aws-bedrock/)**: Deploy with AWS-hosted LLMs
- **[Foundation Model Comparison](https://timecopilot.dev/examples/foundation-models/)**: Benchmark TimesFM, MOIRAI, and others

For more sktime model options, see the [sktime forecasting documentation](https://www.sktime.net/en/stable/api_reference/forecasting.html).
