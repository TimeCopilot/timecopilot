from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timecopilot.models.ensembles.toto_2_fnf import (
    PUBLISHED_MODEL_ORDER,
    FamilyAndFriends,
    _canonical_freq,
    _categorical_freq,
    _infer_term,
)
from timecopilot.models.utils.forecaster import Forecaster, QuantileConverter


class DummyQuantileModel(Forecaster):
    def __init__(self, alias: str, value: float):
        self.alias = alias
        self.value = value

    def forecast(self, df, h, freq=None, level=None, quantiles=None):
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        ids = df["unique_id"].drop_duplicates()
        rows = []
        for uid in ids:
            last = df.loc[df["unique_id"] == uid, "ds"].max()
            for ds in pd.date_range(last, periods=h + 1, freq=freq)[1:]:
                row = {"unique_id": uid, "ds": ds, self.alias: self.value}
                for q in qc.quantiles or []:
                    row[f"{self.alias}-q-{int(q * 100)}"] = self.value + q
                rows.append(row)
        return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("freq", "expected"),
    [("W-TUE", "W"), ("QE-DEC", "Q"), ("15min", "T"), ("h", "H")],
)
def test_canonical_freq(freq, expected):
    assert _canonical_freq(freq) == expected


def test_categorical_freq():
    assert _categorical_freq("h") == "H"
    assert _categorical_freq("W-TUE") == "W-TUE"


def test_infer_term():
    assert _infer_term("D", 30) == "short"
    assert _infer_term("D", 300) == "medium"
    assert _infer_term("D", 450) == "long"


def _dummy_pool() -> list[DummyQuantileModel]:
    return [
        DummyQuantileModel(alias, float(i))
        for i, alias in enumerate(PUBLISHED_MODEL_ORDER)
    ]


def test_rejects_non_published_order(monkeypatch):
    monkeypatch.setattr(
        FamilyAndFriends,
        "_build_published_pool",
        lambda self: [DummyQuantileModel("chronos-2", 1.0)],
    )
    with pytest.raises(RuntimeError, match="PUBLISHED_MODEL_ORDER"):
        FamilyAndFriends()


def test_weighted_forecast_without_live_artifacts(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        FamilyAndFriends,
        "_build_published_pool",
        lambda self: _dummy_pool(),
    )
    model = FamilyAndFriends(artifacts_dir=tmp_path)

    monkeypatch.setattr(
        model,
        "_weights",
        lambda **kwargs: pd.DataFrame(
            {
                name: [1.0 / len(PUBLISHED_MODEL_ORDER)]
                for name in PUBLISHED_MODEL_ORDER
            },
            index=["a"],
        ),
    )

    ds = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame({"unique_id": "a", "ds": ds, "y": np.arange(20)})
    result = model.forecast(df, h=2, freq="D", quantiles=[0.1, 0.5, 0.9])

    mean_value = np.mean(np.arange(len(PUBLISHED_MODEL_ORDER)))
    assert np.allclose(result[model.alias], mean_value + 0.5)
    assert np.allclose(result[f"{model.alias}-q-10"], mean_value + 0.1)
    assert np.allclose(result[f"{model.alias}-q-90"], mean_value + 0.9)


def test_rejects_unpublished_quantile(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        FamilyAndFriends,
        "_build_published_pool",
        lambda self: _dummy_pool(),
    )
    model = FamilyAndFriends(artifacts_dir=tmp_path)
    ds = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame({"unique_id": "a", "ds": ds, "y": np.arange(20)})
    with pytest.raises(ValueError, match="published quantiles"):
        model.forecast(df, h=2, freq="D", quantiles=[0.05])


def test_calls_hardcoded_pool_sequentially(monkeypatch, tmp_path: Path):
    calls = []

    class RecordingModel(DummyQuantileModel):
        def forecast(self, *args, **kwargs):
            calls.append(self.alias)
            return super().forecast(*args, **kwargs)

    monkeypatch.setattr(
        FamilyAndFriends,
        "_build_published_pool",
        lambda self: [
            RecordingModel(alias, float(i))
            for i, alias in enumerate(PUBLISHED_MODEL_ORDER)
        ],
    )

    model = FamilyAndFriends(artifacts_dir=tmp_path)

    monkeypatch.setattr(
        model,
        "_weights",
        lambda **kwargs: pd.DataFrame(
            {
                name: [1.0 / len(PUBLISHED_MODEL_ORDER)]
                for name in PUBLISHED_MODEL_ORDER
            },
            index=["a"],
        ),
    )

    ds = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame({"unique_id": "a", "ds": ds, "y": np.arange(20)})

    model.forecast(df=df, h=2, freq="D", quantiles=[0.1, 0.5, 0.9])

    assert calls == PUBLISHED_MODEL_ORDER
