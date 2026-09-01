import sys

import pytest

from timecopilot.models.foundation.chronos import Chronos
from timecopilot.models.foundation.moirai import Moirai
from timecopilot.models.foundation.tafsut import Tafsut
from timecopilot.models.foundation.timegpt import TimeGPT
from timecopilot.models.foundation.timesfm import TimesFM
from timecopilot.models.foundation.toto import Toto
from timecopilot.models.utils.forecaster import Forecaster

SHIM_MODELS = [Chronos, Moirai, Tafsut, TimesFM, Toto, TimeGPT]

if sys.version_info >= (3, 11):
    from timecopilot.models.foundation.tirex import TiRex

    SHIM_MODELS.append(TiRex)

if sys.version_info >= (3, 11) and sys.version_info < (3, 14):
    from timecopilot.models.foundation.flowstate import FlowState
    from timecopilot.models.foundation.patchtst_fm import PatchTSTFM
    from timecopilot.models.foundation.t0 import T0

    SHIM_MODELS.extend([FlowState, PatchTSTFM, T0])

if sys.version_info < (3, 13):
    from timecopilot.models.foundation.sundial import Sundial
    from timecopilot.models.foundation.tabpfn import TabPFN

    SHIM_MODELS.extend([Sundial, TabPFN])


@pytest.mark.parametrize("model_cls", SHIM_MODELS)
def test_foundation_shim_is_forecaster(model_cls):
    assert issubclass(model_cls, Forecaster)


@pytest.mark.parametrize("model_cls", [Chronos, Moirai, TimesFM, Toto])
def test_foundation_shim_has_core_methods(model_cls):
    assert hasattr(model_cls, "forecast")
    assert hasattr(model_cls, "cross_validation")
    assert hasattr(model_cls, "detect_anomalies")
