<div align="center">
      <img src="https://github.com/user-attachments/assets/3d5e8595-f486-40ee-b34f-743849896462#gh-dark-mode-only" alt="TimeCopilot">
      <img src="https://github.com/user-attachments/assets/ca31b9cf-d1e6-45ff-b757-d961e1299f74#gh-light-mode-only" alt="TimeCopilot">
</div>
<div align="center">
  <em>The GenAI Forecasting Agent · LLMs × Time Series Foundation Models</em>
</div>
<div align="center">
  <a href="https://github.com/TimeCopilot/timecopilot/actions/workflows/ci.yaml"><img src="https://github.com/TimeCopilot/timecopilot/actions/workflows/ci.yaml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://pypi.python.org/pypi/timecopilot"><img src="https://img.shields.io/pypi/v/timecopilot.svg" alt="PyPI"></a>
  <a href="https://github.com/TimeCopilot/timecopilot"><img src="https://img.shields.io/pypi/pyversions/timecopilot.svg" alt="versions"></a>
  <a href="https://github.com/TimeCopilot/timecopilot/blob/main/LICENSE"><img src="https://img.shields.io/github/license/TimeCopilot/timecopilot.svg" alt="license"></a>
  <a href="https://discord.gg/7GEdHR6Pfg"><img src="https://img.shields.io/discord/1387291858513821776?label=discord" alt="Join Discord" /></a>
</div>

---

TimeCopilot is an open-source forecasting agent that combines the power of large language models with state-of-the-art time series foundation models (Amazon Chronos, Salesforce Moirai, Google TimesFM, Nixtla TimeGPT, etc.). It automates and explains complex forecasting workflows, making time series analysis more accessible while maintaining professional-grade accuracy.

Developed with 💙 at [timecopilot.dev](https://timecopilot.dev/).

---

!!! tip "Want the latest on TimeCopilot?"
    Have ideas or want to test it in real-world use? Join our [Discord community](https://discord.gg/7GEdHR6Pfg) and help shape the future.

---

## 🚀 Key Capabilities

- **Unified Forecasting Layer**. Combines 30+ time-series foundation models (Chronos, Moirai, TimesFM, TimeGPT…) with LLM reasoning for automated model selection and explanation.

- **Natural-Language Forecasting**. Ask questions in plain English and get forecasts, analysis, validation, and model comparisons. No scripts, pipelines, or dashboards needed.

- **One-Line Forecasting**. Run end-to-end forecasts on any dataset in seconds with a single command (`uvx timecopilot forecast <url>`).

## 📰 News

- **#1 on the GIFT-Eval benchmark**. TimeCopilot reached the [top position](https://timecopilot.dev/experiments/gift-eval/) on the global GIFT-Eval benchmark above AWS, Salesforce, Google, IBM, and top universities.  
- **Accepted at NeurIPS (BERTs Workshop)**. Our [work on agentic forecasting](https://arxiv.org/pdf/2509.00616) was [accepted](https://berts-workshop.github.io/accepted-papers/) at NeurIPS 2025.
- **New Manifesto: “Forecasting, the Agentic Way”.**  Our [founding essay](https://timecopilot.dev/blog/forecasting-the-agentic-way/) on agentic forecasting, TSFMs, and the future of time series.

## How It Works

TimeCopilot is a generative agent that applies a systematic forecasting approach using large language models (LLMs) to:

- Interpret statistical features and patterns
- Guide model selection based on data characteristics
- Explain technical decisions in natural language
- Answer domain-specific questions about forecasts

Here is an schematic of TimeCopilot's architecture:

<p align="center">
  <img src="https://github.com/user-attachments/assets/3ae3c8cb-bcc5-46cd-b80b-a606671ba553" alt="Diagram" width="500"/>
</p>

--- 

## Quickstart in 30 sec

TimeCopilot can pull a public time series dataset directly from the web and forecast it in one command.  No local files, no Python script, just run it with [uvx](https://docs.astral.sh/uv/):

```bash
# Baseline run (uses default model: openai:gpt-4o-mini)
uvx timecopilot forecast https://otexts.com/fpppy/data/AirPassengers.csv
```

Want to try a different LL​M?

```bash
uvx timecopilot forecast https://otexts.com/fpppy/data/AirPassengers.csv \
  --llm openai:gpt-4o
```

Have a specific question?

```bash
uvx timecopilot forecast https://otexts.com/fpppy/data/AirPassengers.csv \
  --llm openai:gpt-4o \
  --query "How many air passengers are expected in total in the next 12 months?"
```

---

## Installation and Setup

TimeCopilot is available on PyPI as [`timecopilot`](https://pypi.org/project/timecopilot/) for Python development. Installation and setup is a simple three-step process:

1. Install TimeCopilot by running:
  ```bash
  pip install timecopilot
  ```
  or 
  ```bash
  uv add timecopilot
  ```
2. Generate an OpenAI API Key:
    1. Create an [openai](https://auth.openai.com/log-in) account.
    2. Visit the [API key](https://platform.openai.com/api-keys) page.
    3. Generate a new secret key.  
    Make sure to copy it, as you’ll need it in the next step. 

3. Export your OpenAI API key as an environment variable by running:

   On Linux:
    ```
    export OPENAI_API_KEY="your-new-secret-key"
    ```
   On Windows (PowerShell):
    ```
    setx OPENAI_API_KEY "your-new-secret-key"
    ```
   Remember to restart session after doing so in order to preserve the changes in the environment variables (Windows).
   You can also do this through python:

   ```
   import openai
   os.environ["OPENAI_API_KEY"] = "your-new-secret-key"
   ```
   
and that's it!

!!! Important
    - TimeCopilot requires Python 3.10+. Additionally, it currently does not support macOS running on Intel processors (x86_64). If you’re using this setup, you may encounter installation issues with some dependencies like PyTorch. If you need support for this architecture, please create a new issue.
    - If on Windows, Python 3.10 is recommended due to some of the packages' current architecture. 


---

## Hello World Example
Here is an example to test TimeCopilot:

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
tc = TimeCopilot(
    llm="openai:gpt-4o",
    retries=3,
)

# Generate forecast
# You can optionally specify the following parameters:
# - freq: The frequency of your data (e.g., 'D' for daily, 'M' for monthly)
# - h: The forecast horizon, which is the number of periods to predict
# - seasonality: The seasonal period of your data, which can be inferred if not provided
result = tc.forecast(df=df, freq="MS")

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

# You can also access the forecast results in the same shape of the
# provided input dataframe.  
print(result.fcst_df)

"""
        unique_id         ds       Theta
0   AirPassengers 1961-01-01  440.969208
1   AirPassengers 1961-02-01  429.249237
2   AirPassengers 1961-03-01  490.693176
...
21  AirPassengers 1962-10-01  472.164032
22  AirPassengers 1962-11-01  411.458160
23  AirPassengers 1962-12-01  462.795227
"""
```
<details> <summary>Click to expand the full forecast output</summary>

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

---

## Non-OpenAI LLM endpoints

TimeCopilot uses [Pydantic](https://docs.pydantic.dev/latest/) to make calls to LLM endpoints, so it should be compatible with all endpoints [pydantic supports](https://ai.pydantic.dev/models/overview/). Instructions on using other models/endpoints with Pydantic can be found on the matching pydantic docs page, such as this page for [Google's models](https://ai.pydantic.dev/models/google/#api-key-generative-language-api).

For more details go to the LLM Providers example in [TimeCopilot's documentation](http://timecopilot.dev/examples/llm-providers/).

Note: models need support for tool use to function properly with TimeCopilot.

---

## Ask about the future

With TimeCopilot, you can ask questions about the forecast in natural language. The agent will analyze the data, generate forecasts, and provide detailed answers to your queries. 

Let's for example ask: **"how many air passengers are expected in the next 12 months?**"

```python
# Ask specific questions about the forecast
result = tc.forecast(
    df=df,
    freq="MS",
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

You can ask other types of questions:

- **Trend analysis**:  
"What's the expected passenger growth rate?"
- **Seasonal patterns**:  
"Which months have peak passenger traffic?"
- **Specific periods**:  
"What's the passenger forecast for summer months?"
- **Comparative analysis**:  
"How does passenger volume compare to last year?"
- **Business insights**:  
"Should we increase aircraft capacity next quarter?"

---

## Next Steps

1. **Try TimeCopilot**: 
    - Check out the examples above
    - Join our [Discord server](https://discord.gg/7GEdHR6Pfg) for community support
    - Share your use cases and feedback
2. **Contribute**:
    - We are passionate about contributions!
    - Submit [feature requests and bug reports](https://timecopilot.dev/community/help/)
    - Pick an item from the [roadmap](https://timecopilot.dev/community/roadmap/)
    - Follow our [contributing](https://timecopilot.dev/contributing/) guide
3. **Stay Updated**:
    - Star the repository 
    - Watch for new releases!


## How to cite?

Our pre-print paper is [available in arxiv](https://arxiv.org/abs/2509.00616).

```
@misc{garza2025timecopilot,
      title={TimeCopilot}, 
      author={Azul Garza and Renée Rosillo},
      year={2025},
      eprint={2509.00616},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.00616}, 
}
```


