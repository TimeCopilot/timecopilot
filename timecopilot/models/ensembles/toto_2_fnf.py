import base64
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xgboost as xgb
from huggingface_hub import snapshot_download
from tsfeatures import tsfeatures

from ..foundation.chronos import Chronos
from ..foundation.flowstate import FlowState
from ..foundation.patchtst_fm import PatchTSTFM
from ..foundation.timesfm import TimesFM
from ..foundation.tirex import TiRex
from ..foundation.toto import Toto
from ..utils.forecaster import Forecaster, QuantileConverter, get_seasonality

PUBLISHED_MODEL_ORDER = [
    "chronos-2",
    "timesfm-2.5",
    "flowstate",
    "tirex",
    "patchtst-fm",
    "toto-2.0-4m",
    "toto-2.0-22m",
    "toto-2.0-313m",
    "toto-2.0-1b",
    "toto-2.0-2.5b",
]
PUBLISHED_QUANTILES = [round(i / 10, 1) for i in range(1, 10)]

# GIFT-Eval's base prediction lengths. The published heads were trained in
# short/medium/long buckets with multipliers 1/10/15 respectively.
_BASE_HORIZON = {
    "A": 6,
    "Y": 6,
    "Q": 8,
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "MIN": 48,
    "S": 60,
}
_TERM_MULTIPLIER = {"short": 1, "medium": 10, "long": 15}


def _canonical_freq(freq: str) -> str:
    """Convert pandas aliases to the frequency keys used by the boosters."""
    freq = freq.replace("QE-", "Q-").replace("YE-", "A-").replace("ME-", "M-")
    name = pd.tseries.frequencies.to_offset(freq).name.upper()
    name = name.split("-", 1)[0]
    return {
        "YE": "A",
        "Y": "A",
        "QE": "Q",
        "ME": "M",
        "MIN": "T",
    }.get(name, name)


def _categorical_freq(freq: str) -> str:
    """Use the legacy pandas spelling present in the training categories."""
    name = pd.tseries.frequencies.to_offset(freq).name
    prefix, *suffix = name.split("-", 1)
    prefix = {
        "YE": "A",
        "Y": "A",
        "QE": "Q",
        "ME": "M",
        "h": "H",
        "min": "T",
        "s": "S",
        "us": "U",
    }.get(prefix, prefix)
    return "-".join([prefix, *suffix]) if suffix else prefix


def _infer_term(freq: str, h: int) -> str:
    canonical = _canonical_freq(freq)
    if canonical not in _BASE_HORIZON:
        raise ValueError(
            "FamilyAndFriends has no published horizon"
            " mapping for frequency {freq!r}. "
            "Pass term='short', 'medium', or 'long' explicitly."
        )
    base = _BASE_HORIZON[canonical]
    return min(
        _TERM_MULTIPLIER,
        key=lambda term: abs(h - base * _TERM_MULTIPLIER[term]),
    )


def _feature_context_cap(seasonality: int) -> int:
    return max(2048, seasonality * 32)


class FamilyAndFriends(Forecaster):
    """Toto-2.0 Family-and-Friends FFORMA-style meta-ensemble.

    Parameters
    ----------
    models
        Base TimeCopilot forecasters used by the meta-learner. Their aliases
        must follow the published Toto-2.0 FnF model order.
    alias
        Output column name. Defaults to ``"Toto-2.0-FnF"``.
    repo_id
        Hugging Face repository containing the published FnF artifacts.
        Defaults to ``"Datadog/Toto-2.0-Family-and-Friends"``.
    artifacts_dir
        Optional local artifact directory. If omitted, artifacts are downloaded
        from ``repo_id``.
    domain
        Optional domain category used by the gating model. Defaults to ``None``.
    term
        Horizon bucket used by the gating model. Defaults to ``"auto"``.
    tsfeatures_threads
        Number of threads used by ``tsfeatures``. Defaults to ``4``.
    """

    def __init__(
        self,
        *,
        alias: str = "Toto-2.0-FnF",
        repo_id: str = "Datadog/Toto-2.0-Family-and-Friends",
        artifacts_dir: str | Path | None = None,
        domain: str | None = None,
        term: Literal["auto", "short", "medium", "long"] = "auto",
        tsfeatures_threads: int = 4,
    ):
        self.models = self._build_published_pool()
        aliases = [model.alias for model in self.models]
        if aliases != PUBLISHED_MODEL_ORDER:
            raise RuntimeError(
                "FnF base-model aliases must match PUBLISHED_MODEL_ORDER exactly."
            )

        self.alias = alias
        self.repo_id = repo_id
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self.domain = domain
        self.term = term
        self.tsfeatures_threads = tsfeatures_threads
        self._bundle: dict | None = None

    def _build_published_pool(self) -> list[Forecaster]:
        return [
            Chronos(repo_id="amazon/chronos-2", alias="chronos-2"),
            TimesFM(repo_id="google/timesfm-2.5-200m-pytorch", alias="timesfm-2.5"),
            FlowState(repo_id="ibm-research/flowstate", alias="flowstate"),
            TiRex(repo_id="NX-AI/TiRex", alias="tirex"),
            PatchTSTFM(repo_id="ibm-research/patchtst-fm-r1", alias="patchtst-fm"),
            Toto(repo_id="Datadog/Toto-2.0-4m", alias="toto-2.0-4m"),
            Toto(repo_id="Datadog/Toto-2.0-22m", alias="toto-2.0-22m"),
            Toto(repo_id="Datadog/Toto-2.0-313m", alias="toto-2.0-313m"),
            Toto(repo_id="Datadog/Toto-2.0-1B", alias="toto-2.0-1b"),
            Toto(repo_id="Datadog/Toto-2.0-2.5B", alias="toto-2.0-2.5b"),
        ]

    def _call_base_models_sequentially(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str,
    ) -> pd.DataFrame:
        """Run base models one at a time to avoid keeping all weights in memory."""
        import gc

        base: pd.DataFrame | None = None
        for model in self._build_published_pool():
            model_forecast = model.forecast(
                df=df,
                h=h,
                freq=freq,
                level=None,
                quantiles=PUBLISHED_QUANTILES,
            )
            keep_cols = [
                "unique_id",
                "ds",
                *[
                    col
                    for col in model_forecast.columns
                    if col.startswith(f"{model.alias}-q-")
                ],
            ]
            model_forecast = model_forecast[keep_cols]
            base = (
                model_forecast
                if base is None
                else base.merge(model_forecast, on=["unique_id", "ds"])
            )

            del model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        if base is None:
            raise RuntimeError("No base forecasts were produced.")
        return base

    def _artifact_root(self) -> Path:
        if self.artifacts_dir is not None:
            return self.artifacts_dir
        return Path(
            snapshot_download(
                repo_id=self.repo_id,
                repo_type="model",
                allow_patterns=[
                    "booster_manifest.json",
                    "feature_columns.json",
                    "categories.json",
                    "models.json",
                ],
            )
        )

    def _load_bundle(self) -> dict:
        if self._bundle is not None:
            return self._bundle
        root = self._artifact_root()
        with open(root / "models.json") as f:
            model_order = json.load(f)
        if model_order != PUBLISHED_MODEL_ORDER:
            raise RuntimeError(
                "FnF artifact model order differs \
                               from the published order."
            )
        with open(root / "feature_columns.json") as f:
            feature_columns = json.load(f)
        with open(root / "categories.json") as f:
            categories = json.load(f)
        with open(root / "booster_manifest.json") as f:
            manifest = json.load(f)
        self._bundle = {
            "feature_columns": feature_columns,
            "categories": categories,
            "manifest": manifest,
        }
        return self._bundle

    def _load_booster(self, freq: str, term: str) -> xgb.Booster:
        manifest = self._load_bundle()["manifest"]
        key = f"{_canonical_freq(freq)}|{term}"
        if key not in manifest:
            raise ValueError(
                f"No published Toto2FnF booster for bucket {key!r}. "
                f"Available buckets: {sorted(manifest)}"
            )
        booster = xgb.Booster()
        booster.load_model(bytearray(base64.b64decode(manifest[key])))
        return booster

    def _features(self, df: pd.DataFrame, freq: str, h: int) -> pd.DataFrame:
        bundle = self._load_bundle()
        feature_columns = bundle["feature_columns"]
        categories = bundle["categories"]
        scalar = {"seasonality", "prediction_length", "num_variates"}
        categorical = {"freq", "domain"}
        canonical_ts_features = [
            col for col in feature_columns if col not in scalar | categorical
        ]

        seasonality = get_seasonality(freq)
        cap = _feature_context_cap(seasonality)
        panel = (
            df.sort_values(["unique_id", "ds"])
            .groupby("unique_id", observed=True, sort=False)
            .tail(cap)[["unique_id", "ds", "y"]]
        )
        feats = tsfeatures(panel, freq=seasonality, threads=self.tsfeatures_threads)
        feats = feats.set_index("unique_id").reindex(
            df["unique_id"].drop_duplicates().tolist()
        )
        feats = feats.reindex(columns=canonical_ts_features).astype(np.float32)
        feats["seasonality"] = np.float32(seasonality)
        feats["prediction_length"] = np.float32(h)
        # TimeCopilot's long format represents independent univariate series.
        feats["num_variates"] = np.float32(1)
        feats["freq"] = _categorical_freq(freq)
        feats["domain"] = self.domain
        feats = feats.reindex(columns=feature_columns)
        for col in categorical:
            feats[col] = feats[col].astype(
                pd.CategoricalDtype(categories=categories[col])
            )
        return feats

    def _weights(self, df: pd.DataFrame, freq: str, h: int, term: str) -> pd.DataFrame:
        features = self._features(df=df, freq=freq, h=h)
        booster = self._load_booster(freq=freq, term=term)
        raw = booster.predict(
            xgb.DMatrix(features, enable_categorical=True),
            output_margin=True,
        ).reshape(len(features), len(PUBLISHED_MODEL_ORDER))
        raw -= raw.max(axis=1, keepdims=True)
        weights = np.exp(raw)
        weights /= weights.sum(axis=1, keepdims=True)
        return pd.DataFrame(
            weights,
            index=features.index,
            columns=PUBLISHED_MODEL_ORDER,
        )

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        requested_quantiles = qc.quantiles
        unsupported = sorted(set(requested_quantiles or []) - set(PUBLISHED_QUANTILES))
        if unsupported:
            raise ValueError(
                "Toto2FnF only supports the published quantiles "
                f"{PUBLISHED_QUANTILES}; unsupported: {unsupported}."
            )

        # The published learner gates nine fixed quantile forecasts. Always run
        # that grid, then return only the quantiles requested by the caller.
        base = self._call_base_models_sequentially(
            df=df,
            h=h,
            freq=freq,
        )
        term = _infer_term(freq, h) if self.term == "auto" else self.term
        weights = self._weights(df=df, freq=freq, h=h, term=term)

        out = base[["unique_id", "ds"]].copy()
        row_weights = weights.reindex(base["unique_id"]).to_numpy()
        q_output_cols = []
        for q in PUBLISHED_QUANTILES:
            pct = int(q * 100)
            values = base[
                [f"{name}-q-{pct}" for name in PUBLISHED_MODEL_ORDER]
            ].to_numpy()
            col = f"{self.alias}-q-{pct}"
            out[col] = (values * row_weights).sum(axis=1)
            q_output_cols.append(col)

        # The submission's central forecast is its 0.5 quantile.
        out[self.alias] = out[f"{self.alias}-q-50"]
        out = out[["unique_id", "ds", self.alias, *q_output_cols]]

        if requested_quantiles is None:
            return out[["unique_id", "ds", self.alias]]
        keep = [f"{self.alias}-q-{int(q * 100)}" for q in requested_quantiles]
        out = out[["unique_id", "ds", self.alias, *keep]]
        return qc.maybe_convert_quantiles_to_level(out, models=[self.alias])


Toto2FnF = FamilyAndFriends
