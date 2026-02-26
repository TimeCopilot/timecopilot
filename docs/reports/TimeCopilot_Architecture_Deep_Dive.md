# TimeCopilot Architecture Deep Dive

## What Is TimeCopilot?

TimeCopilot is an **agentic time series forecasting system** that marries Large Language Models (for reasoning, parameter inference, and natural-language explanation) with a broad roster of forecasting models (statistical, ML, neural, and foundation models). The key idea: you hand it a dataset and optionally a question in plain English, and it figures out the rest — frequency, seasonality, horizon, model selection, cross-validation, and interpretation.

---

## The Two Core Interfaces: Agent vs. Forecaster

TimeCopilot exposes two distinct entry points. Everything else in the codebase serves one or both of them.

### 1. `TimeCopilot` (the Agent)

This is the **high-level, LLM-orchestrated** interface. When you call `tc.forecast()`, the Agent runs the entire pipeline end-to-end:

```
Data In → Parameter Inference (LLM) → Feature Extraction → Model Training & CV
→ Model Selection → Final Forecast → LLM Analysis → Structured Output
```

The Agent uses the LLM at **four distinct decision points**:

1. **Parameter parsing** — extracting `freq`, `h`, and `seasonality` from a natural-language query
2. **Feature analysis** — interpreting statistical features of the time series
3. **Model comparison** — explaining why certain models beat others
4. **Forecast interpretation** — summarizing the forecast and answering the user's question

Key constructor and methods:

```python
tc = TimeCopilot(
    llm="openai:gpt-4o",   # LLM provider string
    retries=3,               # retry failed LLM calls
    forecasters=None          # optional list of extra models to include
)

result = tc.forecast(
    df=df,                    # DataFrame with unique_id, ds, y
    freq=None,                # override frequency
    h=None,                   # override horizon
    seasonality=None,         # override seasonal period
    query=None                # natural-language question
)

answer = tc.query("Will sales increase next quarter?")
```

The result is a `ForecastAgentOutput` object containing every intermediate artifact (features, CV scores, explanations, the forecast itself, and a query response).

### 2. `TimeCopilotForecaster` (the Forecaster)

This is the **lower-level, programmatic** interface. No LLM involved — you choose the models explicitly, and it trains, cross-validates, and returns DataFrames.

```python
from timecopilot import TimeCopilotForecaster
from timecopilot.models.stats import AutoARIMA, AutoETS

tcf = TimeCopilotForecaster(models=[AutoARIMA(), AutoETS()])
fcst_df = tcf.forecast(df=df, h=12, freq="MS")
cv_df   = tcf.cross_validation(df=df, h=12, n_windows=3)
tcf.plot(df, fcst_df)
```

Use the Forecaster when you already know which models you want, you want fine-grained control, or you don't need natural-language explanations.

---

## The Full Pipeline: Step by Step

Below is the exact sequence of operations that fires when you call `tc.forecast(df, query="...")` on the Agent.

### Step 1: Parameter Inference (Three-Tier Precedence)

TimeCopilot needs three parameters before it can do anything: **frequency** (`freq`), **forecast horizon** (`h`), and **seasonal period** (`seasonality`). It resolves them in this priority order:

| Priority | Source | Example |
|---|---|---|
| **1 (highest)** | Natural-language query | `"Forecast the next 24 months with monthly frequency"` → LLM extracts `h=24`, `freq="MS"` |
| **2** | Explicit keyword args | `tc.forecast(df, freq="MS", h=12)` |
| **3 (lowest)** | Automatic inference from data | `maybe_infer_freq(df)` reads timestamp spacing; `get_seasonality(freq)` maps freq to conventional period |

Automatic seasonality defaults:

- Hourly → 24
- Daily → 7
- Weekly → 52
- Monthly → 12
- Quarterly → 4

If horizon is still unset after all three tiers, it defaults to `2 × seasonality`.

### Step 2: Time Series Feature Extraction

The system calculates statistical features using the **tsfeatures** library. These features characterize the shape, complexity, and behavior of the series:

**Stationarity tests:**
- `unitroot_pp` — Phillips-Perron test statistic (low = non-stationary)
- `unitroot_kpss` — KPSS test statistic (high = non-stationary)
- `hurst` — Hurst exponent (>0.5 = trending/persistent, <0.5 = mean-reverting)

**Seasonality metrics:**
- `seasonal_period` — detected cycle length
- `seasonal_strength` — 0–1 scale of how dominant the seasonal component is

**Trend and autocorrelation:**
- `trend` — linear trend coefficient (0–1)
- `x_acf1` — first autocorrelation of the series

**Complexity:**
- `entropy` — spectral entropy (0 = perfectly predictable, 1 = white noise)
- `nperiods` — total number of observations

These features are stored in `result.output.tsfeatures_results` as a list of key-value strings.

### Step 3: LLM Feature Analysis

The extracted features are sent to the LLM with a prompt asking it to interpret what the numbers mean for model selection. The LLM produces a natural-language analysis explaining:

- Whether the series is stationary or needs differencing
- How strong seasonality is and what period it follows
- Whether there's a trend and in which direction
- What model characteristics are likely needed

This analysis is stored in `result.output.tsfeatures_analysis`.

### Step 4: Model Pool Assembly

TimeCopilot assembles a pool of candidate models. The default pool spans four families:

**Statistical / Classical:**
- `AutoARIMA` — automatic ARIMA with AIC-based order selection
- `AutoETS` — automatic exponential smoothing (error/trend/seasonal state space)
- `AutoCES` — complex exponential smoothing
- `Theta` / `DynamicOptimizedTheta` — theta method variants
- `SeasonalNaive` — baseline (repeats last seasonal cycle)
- `HistoricAverage`, `ADIDA`, `IMAPA`, `CrostonClassic` — intermittent demand models
- `ZeroModel` — trivial baseline

**Foundation Models (pre-trained on massive time series corpora):**
- `Chronos` (Amazon) — tokenizes time series, probabilistic
- `Moirai` (Salesforce) — universal forecasting foundation model
- `TimeGPT` (Nixtla) — generative time series model
- `TimesFM` (Google) — Google's foundation model
- `FlowState`, `Sundial`, `Toto`, `TiRex`, `TabPFN` — newer entrants

**Machine Learning:**
- `AutoLGBM` — LightGBM with automatic feature engineering (lags, rolling stats, calendar features)

**Neural Networks:**
- `AutoNHITS` — hierarchical interpolation for time series
- `AutoTFT` — Temporal Fusion Transformer with attention

**Other:**
- `Prophet` — Facebook's additive decomposition model
- `MedianEnsemble` — combines multiple model predictions via median
- `SKTimeAdapter` — wraps any of 200+ sktime models

If you pass `forecasters=[...]` to the Agent constructor, those models are **added** to the default pool.

### Step 5: Cross-Validation

This is the heart of model comparison. TimeCopilot uses **expanding-window time series cross-validation** — never shuffling, always respecting temporal order.

**How it works for a series with 144 observations, h=12, n_windows=3:**

```
Fold 1:  Train on obs 1–96   → Forecast obs 97–108   → Compute error
Fold 2:  Train on obs 1–108  → Forecast obs 109–120  → Compute error
Fold 3:  Train on obs 1–120  → Forecast obs 121–132  → Compute error

Final score = average error across all folds
```

Each fold:
1. Fits the model on all data up to the split point
2. Generates `h` predictions into the future
3. Compares predictions against actual held-out values
4. Computes error metrics

**Primary metric: MASE** (Mean Absolute Scaled Error)
- Normalized by the in-sample seasonal naive error
- MASE < 1.0 means the model beats the seasonal naive baseline
- Scale-independent, so it works across different series

Secondary metrics available: MAE, RMSE, MAPE.

**Example CV results (Air Passengers):**

```
Model             MASE
───────────────────────
AutoARIMA         1.82  ← Winner
Chronos           2.40
Prophet           2.10
ADIDA             3.12
Theta             3.50
SeasonalNaive     4.03  ← Baseline
AutoETS           4.03
```

CV results are stored in `result.output.cross_validation_results`.

### Step 6: LLM Model Comparison

The CV scores are sent to the LLM, which produces:

- An explanation of why the winning model outperformed others
- Connection back to the data characteristics (e.g., "AutoARIMA won because the series is non-stationary with strong trend, and ARIMA's differencing handles this well")
- Whether the best model beats the seasonal naive baseline
- A `reason_for_selection` summary

Stored in `result.output.model_comparison`, `result.output.selected_model`, `result.output.is_better_than_seasonal_naive`, and `result.output.reason_for_selection`.

### Step 7: Final Forecast Generation

The **winning model** (from CV) is retrained on the **full historical dataset** and asked to produce `h` future predictions:

```python
best_model.fit(full_training_data)
forecasts = best_model.predict(h=h)
```

Output format in `result.fcst_df`:

```
  unique_id         ds          AutoARIMA
  AirPassengers  1961-01-01    440.969
  AirPassengers  1961-02-01    429.249
  AirPassengers  1961-03-01    490.693
  ...
```

Some models also produce confidence intervals (columns like `AutoARIMA-lo-95`, `AutoARIMA-hi-95`).

The forecast values are also stored as date:value strings in `result.output.forecast`.

### Step 8: LLM Forecast Analysis

The final forecasts are sent to the LLM for interpretation:

- What patterns are visible in the forecast
- Whether trends and seasonality from history continue
- Confidence caveats
- Direct answer to the user's original query (if one was provided)

Stored in `result.output.forecast_analysis` and `result.output.user_query_response`.

---

## Feature Engineering for ML Models (AutoLGBM Detail)

Foundation models and statistical models work directly on the raw time series. But AutoLGBM needs tabular features. TimeCopilot auto-generates:

```
Raw series → Lag features (y_{t-1}, y_{t-2}, ..., y_{t-k})
           → Rolling statistics (mean, std, min, max over windows of 7, 14, 30)
           → Calendar features (month, quarter, day_of_week, week_of_year)
           → Trend features (linear trend coefficient, detrended values)
           → Seasonal features (seasonal decomposition, seasonal indices)
```

LightGBM then uses its built-in feature importance to weight these, effectively performing feature selection internally.

---

## The `ForecastAgentOutput` Object

Everything the Agent produces is packaged into this structured output:

```python
result.output.tsfeatures_results          # list[str] — raw feature values
result.output.tsfeatures_analysis         # str — LLM interpretation of features
result.output.selected_model              # str — name of winning model
result.output.model_details               # str — technical description of winning model
result.output.cross_validation_results    # list[str] — MASE per model
result.output.model_comparison            # str — LLM explanation of why models ranked as they did
result.output.is_better_than_seasonal_naive  # bool — does winner beat baseline?
result.output.reason_for_selection        # str — summary of selection rationale
result.output.forecast                    # list[str] — "date: value" pairs
result.output.forecast_analysis           # str — LLM interpretation of forecast
result.output.user_query_response         # str | None — answer to natural-language question

result.fcst_df                            # pd.DataFrame — forecast in tabular form
```

---

## Interactive Mode (CLI Entry Point)

TimeCopilot also ships a CLI that acts as the "app" entry point:

```bash
timecopilot
```

This launches an interactive session:
1. **LLM selection** — choose provider (OpenAI, Anthropic, Google, AWS Bedrock, etc.)
2. **API key setup** — enter credentials
3. **Chat loop** — type commands:
   - `forecast /path/to/data.csv` — load data and run full pipeline
   - Free-text questions — ask about existing forecasts via `tc.query()`
   - `llm` — switch LLM provider mid-session
   - `help` / `exit`

The CLI wraps the same Agent pipeline described above.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     User Input                                    │
│              DataFrame + optional query string                    │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Parameter Inference  │
              │  (LLM + kwargs +      │
              │   auto-detection)     │
              │  → freq, h, season    │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Feature Extraction   │
              │  (tsfeatures library) │
              │  → hurst, kpss, pp,  │
              │    trend, entropy,    │
              │    seasonal_strength  │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  LLM Feature Analysis │
              │  "Strong seasonality, │
              │   non-stationary..."  │
              └───────────┬───────────┘
                          │
                          ▼
     ┌────────────────────┼────────────────────┐
     │                    │                    │
     ▼                    ▼                    ▼
┌──────────┐      ┌──────────────┐     ┌──────────────┐
│Statistical│      │ Foundation   │     │   ML/Neural  │
│AutoARIMA │      │ Chronos      │     │  AutoLGBM    │
│AutoETS   │      │ Moirai       │     │  AutoNHITS   │
│Theta     │      │ TimesFM      │     │  AutoTFT     │
│Prophet   │      │ TimeGPT      │     │  Prophet     │
│SeasonalN │      │ Toto, etc.   │     │              │
└────┬─────┘      └──────┬───────┘     └──────┬───────┘
     │                   │                    │
     └───────────────────┼────────────────────┘
                         │
                         ▼
              ┌───────────────────────┐
              │   Cross-Validation    │
              │  Expanding window     │
              │  MASE per model       │
              │  Multiple folds       │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ LLM Model Comparison  │
              │ "AutoARIMA won        │
              │  because..."          │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Best Model Retrain   │
              │  on full dataset      │
              │  → Generate h-step    │
              │    forecast           │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ LLM Forecast Analysis │
              │ + Query Response      │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  ForecastAgentOutput  │
              │  (all artifacts +     │
              │   explanations +      │
              │   forecast DataFrame) │
              └───────────────────────┘
```

---

## Summary Table: Agent vs. Forecaster

| Aspect | Agent (`TimeCopilot`) | Forecaster (`TimeCopilotForecaster`) |
|---|---|---|
| LLM required | Yes | No |
| Natural-language queries | Yes | No |
| Parameter inference | Automatic (3-tier) | Must provide explicitly |
| Feature extraction | Automatic + LLM analysis | Not included |
| Model selection | LLM-guided from default pool | You choose the models |
| Cross-validation | Automatic | Call `.cross_validation()` manually |
| Explanations | Full NL explanations at every step | None (raw DataFrames) |
| Output type | `ForecastAgentOutput` (rich) | `pd.DataFrame` (simple) |
| Best for | End users, exploration, Q&A | Developers, benchmarking, pipelines |
