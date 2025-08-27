This section describes how [`TimeCopilot`][timecopilot.agent.TimeCopilot] determines the three core forecasting parameters: `freq`, `h`, and `seasonality` when you call
[`TimeCopilot.forecast(...)`][timecopilot.agent.TimeCopilot.forecast].  

You can:

- let the assistant infer everything automatically (recommended for a quick
  start),
- provide the values in plain-English inside the `query`, or
- override them explicitly with keyword arguments.

### What do these terms mean?

* **`freq`**: the pandas frequency string that describes the spacing of your
  timestamps (`"H"` for hourly, `"D"` for daily, `"MS"` for monthly-start,
  etc.).  It tells the models how many observations occur in one unit of
  *seasonality*.
* **`seasonality`**: the length of the dominant seasonal cycle expressed in
  number of `freq` periods (24 for hourly data with a daily cycle, 12 for
  monthly‐start data with a yearly cycle, …).  See
  [`get_seasonality`][timecopilot.models.utils.forecaster.get_seasonality] for the default mapping.
* **`h` (horizon)**: how many future periods you want to forecast.

!!! tip "Pandas available frequencies"
    You can see [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases) the complete list of available frequencies.  

With these concepts in mind, let's see how [`TimeCopilot`][timecopilot.agent.TimeCopilot] chooses their values.

## Where do the parameters come from?

`TimeCopilot` follows these precedence rules:

1. **Natural-language query wins.**
   If the query text mentions any of the three parameters they are extracted by
   an LLM agent and used first.
2. **Explicit keyword arguments are next.**
   Any argument you pass directly to `parse()` ( `freq=`, `h=`,
   `seasonality=` ) fills the gaps left by the query.
3. **Automatic inference is the fallback.**
   If a value is still unknown it is inferred from the data frame:
    * `freq`: [`maybe_infer_freq(df)`][timecopilot.models.utils.forecaster.maybe_infer_freq]
    * `seasonality`: [`get_seasonality(freq)`][timecopilot.models.utils.forecaster.get_seasonality]
    * `h`: `2 * seasonality`

!!! tip "Summary"
    Text -> kwargs -> automatic inference (from your data, `df`)

## Passing parameters in a natural-language query

Sometimes it's easier to embed the settings directly in your query:

```python
import pandas as pd
from timecopilot.agent import TimeCopilot

df = pd.read_csv("https://timecopilot.s3.amazonaws.com/public/data/air_passengers.csv")
query = """
  Which months will have peak passenger traffic in the next 24 months? 
  use 12 as seasonality
""" 

tc = TimeCopilot(llm="gpt-4o")

# Passing `None` simply uses the defaults; they are shown
# here for clarity but can be omitted.
result = tc.forecast(
    df=df,
    query=query,
    freq=None,          # default, infer from query/df
    h=None,             # default, infer from query/df
    seasonality=None,   # default, infer from query/df
)

print(result.output.user_query_response)
# Based on the forecast, peak passenger traffic 
# in the next 24 months is expected to occur in the months of July and August 
# both in 1961 and 1962.
```

??? note "How does the inference happen?"
    Under the hood the LLM receives a system prompt like:

    > "Extract the following fields if they appear in the user text…"

    …and returns a JSON tool call that is validated against the
    `DatasetParams` schema.


## Supplying the parameters programmatically (skip the LLM)

If you already know the values you can skip the LLM entirely:

```python
result = tc.forecast(
    df=df,
    freq="MS",     # monthly-start
    h=12,           # one year ahead
    seasonality=12, # yearly
    query=None,     # no natural-language query
)
```

Because every field is supplied, no inference or LLM call happens.

## Mixed approach (query + kwargs)

You can combine both techniques. The parser fills the *missing* fields from the
kwargs or, if still empty, infers them:

```python
query = "Which months will have peak passenger traffic in the next 24 months?"
result = tc.forecast(
    df=df,
    freq="MS",       # explicit override
    h=None,           # default, pulled from query (24)
    seasonality=None, # default, inferred as 12
    query=query,
)
```

## Choosing sensible defaults

When you let [`TimeCopilot`][timecopilot.agent.TimeCopilot] infer the parameters:

* `freq` should be either present in the query **or** directly deducible from
your `ds` column (regular timestamps with no gaps).
* `seasonality` defaults to the conventional period for the frequency
  (e.g. 7 for daily, 12 for monthly). Override it if your data behaves
differently.
* `h` defaults to twice the seasonal period—large enough for
  meaningful evaluation while staying quick to compute.

!!! note
    These defaults aim to keep the *first run* friction-free. Fine-tune them
    as soon as you have more insight into your particular dataset. 