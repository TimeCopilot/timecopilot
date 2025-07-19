# Max Length Parameter

The `max_length` parameter has been added to improve inference times by using only the last N values of each time series.

## Usage

### Python API

```python
from timecopilot import TimeCopilot

# Initialize with max_length
tc = TimeCopilot(llm="openai:gpt-4o", max_length=100)

# Or set it per forecast
result = tc.forecast(df=df, max_length=50)
```

### CLI

```bash
# Use max_length parameter
timecopilot forecast data.csv --max_length 100

# With other parameters
timecopilot forecast data.csv --llm openai:gpt-4o --max_length 50
```

## How it works

When `max_length` is set, each time series is truncated to use only the last N observations before training and inference. This can significantly improve performance for long time series while often maintaining good forecast accuracy.

## Example

```python
import pandas as pd
from timecopilot import TimeCopilot

# Create long time series
df = pd.DataFrame({
    'unique_id': ['series_1'] * 1000,
    'ds': pd.date_range('2020-01-01', periods=1000, freq='D'),
    'y': range(1000)
})

# Use only last 100 observations
tc = TimeCopilot(llm="openai:gpt-4o", max_length=100)
result = tc.forecast(df=df, h=10)
```

The forecaster will automatically use only the last 100 observations from each series, potentially improving speed while maintaining accuracy.