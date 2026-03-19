import sys

import pytest

from timecopilot.models.ensembles.median import MedianEnsemble
from timecopilot.models.foundation.chronos import Chronos
from timecopilot.models.foundation.flowstate import FlowState
from timecopilot.models.foundation.moirai import Moirai
from timecopilot.models.foundation.patchtst_fm import PatchTSTFM
from timecopilot.models.foundation.timesfm import TimesFM
from timecopilot.models.foundation.toto import Toto
from timecopilot.models.ml import AutoLGBM
from timecopilot.models.neural import AutoNHITS, AutoTFT
from timecopilot.models.prophet import Prophet
from timecopilot.models.stats import (
    ADIDA,
    AutoARIMA,
    SeasonalNaive,
    ZeroModel,
)


@pytest.fixture(autouse=True)
def disable_mps_session(monkeypatch):
    # Make torch.backends.mps report unavailable
    try:
        import torch

        monkeypatch.setattr(
            torch.backends.mps, "is_available", lambda: False, raising=False
        )
        monkeypatch.setattr(
            torch.backends.mps, "is_built", lambda: False, raising=False
        )
    except Exception:
        # torch might not be installed in some envs; ignore
        pass


models = [
    AutoLGBM(num_samples=2, cv_n_windows=2),
    AutoNHITS(
        num_samples=2,
        config=dict(
            max_steps=1,
            val_check_steps=1,
            input_size=12,
            mlp_units=3 * [[8, 8]],
        ),
    ),
    AutoTFT(
        num_samples=2,
        config=dict(
            max_steps=1,
            val_check_steps=1,
            input_size=12,
            hidden_size=8,
        ),
    ),
    AutoARIMA(),
    SeasonalNaive(),
    ZeroModel(),
    ADIDA(),
    Prophet(),
    Chronos(repo_id="amazon/chronos-bolt-tiny", alias="Chronos-Bolt"),
    Chronos(repo_id="amazon/chronos-2", alias="Chronos-2"),
    Chronos(repo_id="amazon/chronos-2", alias="Chronos-2", batch_size=2),
    FlowState(repo_id="ibm-research/flowstate"),
    FlowState(
        repo_id="ibm-granite/granite-timeseries-flowstate-r1",
        alias="FlowState-Granite",
    ),
    PatchTSTFM(
        context_length=2_048,
    ),
    Toto(context_length=256, batch_size=2),
    Moirai(
        context_length=256,
        batch_size=2,
        repo_id="Salesforce/moirai-1.1-R-small",
    ),
    TimesFM(
        repo_id="google/timesfm-1.0-200m-pytorch",
        context_length=256,
    ),
    TimesFM(
        repo_id="google/timesfm-2.5-200m-pytorch",
        context_length=256,
    ),
    MedianEnsemble(
        models=[
            Chronos(repo_id="amazon/chronos-t5-tiny", alias="Chronos-T5"),
            Chronos(repo_id="amazon/chronos-bolt-tiny", alias="Chronos-Bolt"),
            SeasonalNaive(),
        ],
    ),
    Moirai(
        context_length=256,
        batch_size=2,
        repo_id="Salesforce/moirai-2.0-R-small",
    ),
]
if sys.version_info >= (3, 11):
    from timecopilot.models.foundation.tirex import TiRex

    models.append(TiRex())

if sys.version_info < (3, 13):
    from tabpfn_time_series import TabPFNMode

    from timecopilot.models.foundation.tabpfn import TabPFN

    models.append(TabPFN(mode=TabPFNMode.MOCK))
