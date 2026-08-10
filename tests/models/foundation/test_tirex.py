import sys

import pytest
from utilsforecast.data import generate_series

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="TiRex requires Python >= 3.11",
)


def test_is_tirex2_dispatch():
    from timecopilot.models.foundation.tirex import TiRex

    assert not TiRex(repo_id="NX-AI/TiRex")._is_tirex2()
    assert TiRex(repo_id="NX-AI/TiRex-2")._is_tirex2()
    assert TiRex(repo_id="NX-AI/TiRex-2/")._is_tirex2()


def test_tirex2_forecast():
    from timecopilot.models.foundation.tirex import TiRex

    df = generate_series(2, freq="D", min_length=50, max_length=50)
    df["unique_id"] = df["unique_id"].astype(str)
    model = TiRex(repo_id="NX-AI/TiRex-2", alias="TiRex-2", batch_size=2)
    fcst = model.forecast(df, h=3, freq="D")
    assert fcst.shape == (6, 3)
    assert "TiRex-2" in fcst.columns
