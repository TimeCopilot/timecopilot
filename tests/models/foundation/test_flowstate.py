import pandas as pd
import numpy as np

from timecopilot import TimeCopilotForecaster
from timecopilot.models.foundation.flowstate import FlowState


def test_flowstate_h1_single_uid():
    # create simple weekly data for one unique_id
    ds = pd.date_range("2024-01-01", periods=20, freq="W")
    df = pd.DataFrame({
        "unique_id": "u1",
        "ds": ds,
        "y": np.arange(20)
    })

    tcf = TimeCopilotForecaster(models=[FlowState()])

    # this used to crash before the fix
    fcst = tcf.forecast(df=df, h=1, freq="W")

    # basic checks
    assert isinstance(fcst, pd.DataFrame)
    assert len(fcst) == 1
    assert "unique_id" in fcst.columns
    assert "ds" in fcst.columns
