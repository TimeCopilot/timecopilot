<div align="center">
      <img src="https://github.com/user-attachments/assets/4b269fb3-3edc-42ad-a08c-6f79d2fe6aca#gh-dark-mode-only" alt="TimeCopilot">
      <img src="https://github.com/user-attachments/assets/5b6fb92e-a460-48cb-a218-d8321e9b54f5#gh-light-mode-only" alt="TimeCopilot">
</div>
<div align="center">
  <em>The GenAI Forecasting Agent · LLMs × Foundation Time Series Models</em>
</div>
<div align="center">
  <a href="https://github.com/AzulGarza/TimeCopilot/actions/workflows/ci.yaml"><img src="https://github.com/AzulGarza/TimeCopilot/actions/workflows/ci.yaml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://pypi.python.org/pypi/timecopilot"><img src="https://img.shields.io/pypi/v/timecopilot.svg" alt="PyPI"></a>
  <a href="https://github.com/AzulGarza/timecopilot"><img src="https://img.shields.io/pypi/pyversions/timecopilot.svg" alt="versions"></a>
  <a href="https://github.com/AzulGarza/timecopilot/blob/main/LICENSE"><img src="https://img.shields.io/github/license/azulgarza/timecopilot.svg" alt="license"></a>
  <a href="https://discord.gg/7GEdHR6Pfg"><img src="https://img.shields.io/discord/1387291858513821776?label=discord" alt="Join Discord" /></a>
</div>

---

TimeCopilot is an open-source forecasting agent that combines the power of large language models with state-of-the-art time series foundation models. It automates and explains complex forecasting workflows, making time series analysis more accessible while maintaining professional-grade accuracy.

Developed with 💙 at [timecopilot.dev](https://timecopilot.dev/).

---

Want to stay updated on TimeCopilot's latest developments? Have a feature request or interested in testing it for production loads? Fill out [our form](https://docs.google.com/forms/d/e/1FAIpQLSeQWKVHjYKe1ayEso-K2My9nQsoaSWxzht0S6D4yrCln7BECQ/viewform?usp=dialog) or join our [Discord community](https://discord.gg/7GEdHR6Pfg) to be part of the conversation!


## How It Works

TimeCopilot operates as a generative agent that follows a systematic approach to forecasting:

<p align="center">
  <img src="https://github.com/user-attachments/assets/3ae3c8cb-bcc5-46cd-b80b-a606671ba553" alt="Diagram" width="500"/>
</p>

The agent leverages LLMs to:

- Interpret statistical features and patterns
- Guide model selection based on data characteristics
- Explain technical decisions in natural language
- Answer domain-specific questions about forecasts

## Quick-start (30 sec)

TimeCopilot can pull a public time series dataset directly from the web and forecast it in one command.  No local files, no Python script, just run it with [uvx](https://docs.astral.sh/uv/):

```bash
# Baseline run (uses default model: openai:gpt-4o-mini)
uvx timecopilot forecast https://otexts.com/fpppy/data/AirPassengers.csv
```

Want to try a different LLM?

```bash
uvx timecopilot forecast https://otexts.com/fpppy/data/AirPassengers.csv \
  --llm openai:gpt-4o
```

Need faster inference for long time series?

```bash
uvx timecopilot forecast https://otexts.com/fpppy/data/AirPassengers.csv \
  --llm openai:gpt-4o \
  --max_length 100
```

Have a specific question?

```bash
uvx timecopilot forecast https://otexts.com/fpppy/data/AirPassengers.csv \
  --llm openai:gpt-4o \
  --query "How many air passengers are expected in total in the next 12 months?"
```

## Installation

TimeCopilot is available on PyPI as [`timecopilot`](https://pypi.org/project/timecopilot/) for Python development so installation is as simple as:

```bash
pip install timecopilot
```

or 

```bash
uv add timecopilot
```

(Requires Python 3.10+)

## Hello World Example

```python
# Import libraries
import pandas as pd
from timecopilot import TimeCopilot

# Load the dataset
# The DataFrame must include at least the following columns:
# - unique_id: Unique identifier for each time series (string)
# - ds: Date column (datetime format)
# - y: Target variable for forecasting (float format)
# The pandas frequency will be inferred from the ds column, if not provided.
# If the seasonality is not provided, it will be inferred based on the frequency. 
# If the horizon is not set, it will default to 2 times the inferred seasonality.
df = pd.read_csv("https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv")

# Initialize the forecasting agent
# You can use any LLM by specifying the model parameter
# Optional: Set max_length to use only the last N observations for faster inference
tc = TimeCopilot(
    llm="openai:gpt-4o",
    retries=3,
    max_length=100,  # Use only last 100 observations for faster inference
)

# Generate forecast
# You can optionally specify the following parameters:
# - freq: The frequency of your data (e.g., 'D' for daily, 'M' for monthly)
# - h: The forecast horizon, which is the number of periods to predict
# - seasonality: The seasonal period of your data, which can be inferred if not provided
# - max_length: Maximum number of observations to use from the end of each series
result = tc.forecast(df=df)

# The output contains:
# - tsfeatures_results: List of calculated time series features
# - tsfeatures_analysis: Natural language analysis of the features
# - selected_model: The best performing model chosen
# - model_details: Technical details about the selected model
# - cross_validation_results: Performance comparison of different models
# - model_comparison: Analysis of why certain models performed better/worse
# - is_better_than_seasonal_naive: Boolean indicating if model beats baseline
# - reason_for_selection: Explanation for model choice
# - forecast: List of future predictions with dates
# - forecast_analysis: Interpretation of the forecast results
# - user_query_response: Response to the user prompt, if any
print(result.output)
```
<details> <summary>Click to expand full forecast output</summary>

```python
"""
tsfeatures_results=['hurst: 1.04', 'unitroot_pp: -6.57', 'unitroot_kpss: 2.74', 
'nperiods: 1,seasonal_period: 12', 'trend: 1.00', 'entropy: 0.43', 'x_acf1: 0.95', 
'seasonal_strength: 0.98'] tsfeatures_analysis="The time series presents a strong seasonality 
with a seasonal period of 12 months, indicated by a strong seasonal strength of 0.98. The 
high trend component suggests an upward motion over the periods. The KPSS statistic indicates 
non-stationarity as it's greater than the typical threshold of 0.5, confirming the presence 
of a trend. The Auto-ARIMA model indicated adjustments for non-stationarity through 
differencing. The strong correlation captured by high ACF values further supports the need 
for integrated models due to persistence and gradual changes over time." 
selected_model='AutoARIMA' model_details='The AutoARIMA model automatically selects the 
differencing order, order of the autoregressive (AR) terms, and the moving average (MA) 
terms based on the data. It is particularly suitable for series with trend and seasonality, 
and performs well in scenarios where automatic model selection for differencing is required 
to obtain stationary data. It uses AIC for model selection among a candidate pool, ensuring 
a balanced complexity and goodness of fit.' cross_validation_results=['ADIDA: 3.12', 
'AutoARIMA: 1.82', 'AutoETS: 4.03', 'Theta: 3.50', 'SeasonalNaive: 4.03'] 
model_comparison='AutoARIMA performed best with a cross-validation score of 1.82, indicating 
its effectiveness in capturing the underlying trend and seasonal patterns successfully as it 
adjusts for trend and seasonality through differencing and parameter tuning. The seasonal 
naive model did not compete well perhaps due to the deeper complex trends in the data beyond 
mere seasonal repetition. Both AutoETS and Theta lacked the comparable accuracy, potentially 
due to inadequate adjustment for non-stationary trend components.' 
is_better_than_seasonal_naive=True reason_for_selection="AutoARIMA was chosen due to its 
lowest cross-validation score, demonstrating superior accuracy compared to other models by 
effectively handling both trend and seasonal components in a non-stationary series, which 
aligns well with the data's characteristics." forecast=['1961-01-01: 476.33', '1961-02-01: 
504.00', '1961-03-01: 512.06', '1961-04-01: 507.34', '1961-05-01: 498.92', '1961-06-01: 
493.23', '1961-07-01: 492.49', '1961-08-01: 495.79', '1961-09-01: 500.90', '1961-10-01: 
505.86', '1961-11-01: 509.70', '1961-12-01: 512.38', '1962-01-01: 514.38', '1962-02-01: 
516.24', '1962-03-01: 518.31', '1962-04-01: 520.68', '1962-05-01: 523.28', '1962-06-01: 
525.97', '1962-07-01: 528.63', '1962-08-01: 531.22', '1962-09-01: 533.74', '1962-10-01: 
536.23', '1962-11-01: 538.71', '1962-12-01: 541.21'] forecast_analysis="The forecast 
indicates a continuation of the upward trend with periodic seasonal fluctuations that align 
with historical patterns. The strong seasonality is evident in the periodic peaks, with 
slight smoothing over time due to parameter adjustment for stability. The forecasts are 
reliable given the past performance metrics and the model's rigorous tuning. However, 
potential uncertainties could arise from structural breaks or changes in pattern, not 
reflected in historical data." user_query_response='The analysis determined the best 
performing model and generated forecasts considering seasonality and trend, aiming for 
accuracy and reliability surpassing basic seasonal models.'
"""
```

</details>

## Ask about the future

You can ask questions about the forecast in natural language. The agent will analyze the data, generate forecasts, and provide detailed answers to your queries.

```python
# Ask specific questions about the forecast
result = tc.forecast(
    df=df,
    query="how many air passengers are expected in the next 12 months?",
)

# The output will include:
# - All the standard forecast information
# - user_query_response: Detailed answer to your specific question
#   analyzing the forecast in the context of your query
print(result.output.user_query_response)

"""
The total expected air passengers for the next 12 months is approximately 5,919.
"""
```

You can ask various types of questions:

- Trend analysis: "What's the expected passenger growth rate?"
- Seasonal patterns: "Which months have peak passenger traffic?"
- Specific periods: "What's the passenger forecast for summer months?"
- Comparative analysis: "How does passenger volume compare to last year?"
- Business insights: "Should we increase aircraft capacity next quarter?"

## Next Steps

1. **Try TimeCopilot**: 
    - Check out the examples above
    - Join our [Discord server](https://discord.gg/7GEdHR6Pfg) for community support
    - Share your use cases and feedback
2. **Contribute**:
    - We are passionate about contributions!
    - Submit [feature requests and bug reports](https://timecopilot.dev/help/)
    - Pick an item from the [roadmap](https://timecopilot.dev/roadmap/)
    - Follow our [contributing](https://timecopilot.dev/contributing/) guide
3. **Stay Updated**:
    - Star the repository 
    - Watch for new releases!




