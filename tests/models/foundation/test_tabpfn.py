from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

pytest.importorskip("tabpfn_time_series", reason="TabPFN requires Python < 3.13")

from tabpfn.errors import TabPFNLicenseError  # noqa: E402
from tabpfn_time_series import TabPFNMode  # noqa: E402
from utilsforecast.data import generate_series  # noqa: E402

from timecopilot.models.foundation.tabpfn import (  # noqa: E402
    TABPFN_V2_MODEL,
    TABPFN_V3_MODEL,
    TabPFN,
)

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

pytestmark = pytest.mark.models

DEFAULT_QUANTILES = [round(i * 0.1, 1) for i in range(1, 10)]
DEFAULT_LEVEL = [0, 20, 40, 60, 80]

TABPFN_CASES = [
    pytest.param(TABPFN_V2_MODEL, 4096, "TabPFN-2", id="v2"),
    pytest.param(TABPFN_V3_MODEL, 32768, "TabPFN-3", id="v3"),
]


def _require_tabpfn_token() -> None:
    if not os.environ.get("TABPFN_TOKEN"):
        pytest.skip("TABPFN_TOKEN not set")


def test_tabpfn_default_model_path() -> None:
    model = TabPFN()
    assert model.model_path == TABPFN_V2_MODEL


def test_tabpfn_v3_model_path() -> None:
    model = TabPFN(model_path=TABPFN_V3_MODEL, context_length=32768)
    assert model.model_path == TABPFN_V3_MODEL
    assert model.context_length == 32768


def test_tabpfn_predictor_receives_model_path() -> None:
    with patch(
        "foundationforecast.models.tabpfn.TabPFNTimeSeriesPredictor"
    ) as predictor_cls:
        predictor_cls.return_value = MagicMock()
        model = TabPFN(model_path=TABPFN_V3_MODEL, mode=TabPFNMode.LOCAL)
        with model._get_model():
            pass
        predictor_cls.assert_called_once_with(
            tabpfn_mode=TabPFNMode.LOCAL,
            tabpfn_config={"model_path": TABPFN_V3_MODEL},
        )


@pytest.fixture(scope="module")
def tabpfn_df():
    return generate_series(n_series=1, freq="D", min_length=30, max_length=30)


def _make_model(model_path: str, context_length: int, alias: str) -> TabPFN:
    return TabPFN(
        model_path=model_path,
        mode=TabPFNMode.LOCAL,
        context_length=context_length,
        alias=alias,
    )


def _forecast_or_skip_v3(model: TabPFN, *args, **kwargs):
    try:
        return model.forecast(*args, **kwargs)
    except TabPFNLicenseError as exc:
        if model.model_path == TABPFN_V3_MODEL:
            pytest.skip(
                "TabPFN-3 license not accepted; accept at https://ux.priorlabs.ai"
            )
        raise exc


@pytest.mark.parametrize("model_path,context_length,alias", TABPFN_CASES)
def test_tabpfn_local_point_forecast(
    tabpfn_df, model_path: str, context_length: int, alias: str
) -> None:
    _require_tabpfn_token()
    fcst = _forecast_or_skip_v3(
        _make_model(model_path, context_length, alias),
        tabpfn_df,
        h=3,
        freq="D",
    )
    assert fcst.shape == (3, 3)
    assert alias in fcst.columns


@pytest.mark.parametrize("model_path,context_length,alias", TABPFN_CASES)
def test_tabpfn_local_quantile_forecast(
    tabpfn_df, model_path: str, context_length: int, alias: str
) -> None:
    _require_tabpfn_token()
    fcst = _forecast_or_skip_v3(
        _make_model(model_path, context_length, alias),
        tabpfn_df,
        h=3,
        freq="D",
        quantiles=DEFAULT_QUANTILES,
    )
    q_cols = [f"{alias}-q-{int(100 * q)}" for q in DEFAULT_QUANTILES]
    assert len(fcst.columns) == 3 + len(q_cols)
    assert all(col in fcst.columns for col in q_cols)
    assert not any("-lo-" in col or "-hi-" in col for col in fcst.columns)
    for c1, c2 in zip(q_cols[:-1], q_cols[1:], strict=False):
        assert fcst[c1].le(fcst[c2]).mean() >= 0.8


@pytest.mark.parametrize("model_path,context_length,alias", TABPFN_CASES)
def test_tabpfn_local_level_forecast(
    tabpfn_df, model_path: str, context_length: int, alias: str
) -> None:
    _require_tabpfn_token()
    fcst = _forecast_or_skip_v3(
        _make_model(model_path, context_length, alias),
        tabpfn_df,
        h=3,
        freq="D",
        level=DEFAULT_LEVEL,
    )
    lv_cols = []
    for lv in DEFAULT_LEVEL:
        lv_cols.extend([f"{alias}-lo-{lv}", f"{alias}-hi-{lv}"])
    assert len(fcst.columns) == 3 + len(lv_cols)
    assert all(col in fcst.columns for col in lv_cols)
    assert not any("-q-" in col for col in fcst.columns)
    for lo, hi in zip(lv_cols[2::2], lv_cols[3::2], strict=False):
        assert fcst[lo].le(fcst[hi]).all()
