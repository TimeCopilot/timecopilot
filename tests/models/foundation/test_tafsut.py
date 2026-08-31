import numpy as np
import pandas as pd

from timecopilot import TimeCopilotForecaster
from timecopilot.models.foundation.tafsut import Tafsut


def test_tafsut_h1_single_uid():
    ds = pd.date_range("2024-01-01", periods=20, freq="W")
    df = pd.DataFrame({"unique_id": "u1", "ds": ds, "y": np.arange(20)})

    tcf = TimeCopilotForecaster(models=[Tafsut(context_length=512, batch_size=1)])

    fcst = tcf.forecast(df=df, h=1, freq="W")

    assert isinstance(fcst, pd.DataFrame)
    assert len(fcst) == 1
    assert "unique_id" in fcst.columns
    assert "ds" in fcst.columns
