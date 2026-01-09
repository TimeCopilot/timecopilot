import sys
from functools import partial

import pandas as pd

# Remember to set your project path!

# Added for standarization
from datasetsforecast.m4 import M4, Monthly
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mase, smape

from timecopilot.models.ensembles.median import MedianEnsemble
from timecopilot.models.ensembles.trimmed import TrimmedEnsemble
from timecopilot.models.foundation.chronos import Chronos
from timecopilot.models.foundation.flowstate import FlowState
from timecopilot.models.foundation.timesfm import TimesFM
from timecopilot.models.foundation.tirex import TiRex
from timecopilot.models.stats import AutoARIMA, SeasonalNaive, Theta

# -----------------------------
# helpers
# -----------------------------
def normalize_month_start(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    p = pd.to_datetime(out["ds"]).dt.to_period("M")
    out["ds"] = p.dt.to_timestamp()  # month start by default
    return out


def debug_df(name, df):
    print(f"\n[{name}] shape={df.shape}")
    print(df.head(3))
    print(f"[{name}] dtypes:\n{df.dtypes}")
    print(f"[{name}] unique_id n={df['unique_id'].nunique()}")
    print(f"[{name}] ds min/max: {df['ds'].min()} -> {df['ds'].max()}")
    print(f"[{name}] y NaNs: {df['y'].isna().sum()}")
    lens = df.groupby("unique_id").size()
    print(f"[{name}] per-series length:\n{lens.to_string()}")


def debug_forecast_output(name, fcst, alias):
    print(f"\n[{name}] forecast shape={fcst.shape}")
    print(fcst.head(3))
    print(f"[{name}] columns: {list(fcst.columns)}")
    if alias not in fcst.columns:
        raise RuntimeError(f"[{name}] missing point column: {alias}")
    na_point = fcst[alias].isna().sum()
    print(f"[{name}] point NaNs ({alias}): {na_point}/{len(fcst)}")


def hard_align_or_die(test_df, fcst_df, alias, name):
    merged = test_df.merge(fcst_df, on=["unique_id", "ds"], how="inner")
    if len(merged) != len(test_df):
        # show a tiny forensic sample
        t0 = test_df["ds"].sort_values().head(3).to_list()
        f0 = fcst_df["ds"].sort_values().head(3).to_list()
        raise RuntimeError(
            f"[{name}] Broken alignment: merged={len(merged)} test={len(test_df)}. "
            f"ds(test) head={t0} ds(fcst) head={f0}"
        )
    if merged[alias].isna().any():
        raise RuntimeError(f"[{name}] Alignment ok but predictions contain NaNs.")
    return merged


# -----------------------------
# data (M4 Monthly via datasetsforecast)
# -----------------------------
DATA_DIR = "data"
GROUP = "Monthly"

H = int(Monthly.horizon)  # 18
SEAS = int(Monthly.seasonality)  # 12
FREQ = str(Monthly.freq)  # 'M' (we still normalize ds ourselves)

y_df, *_ = M4.load(directory=DATA_DIR, group=GROUP)
y_df = normalize_month_start(y_df)

# Official split: last H points per series are test
test_all = y_df.groupby("unique_id", sort=False).tail(H).copy()
train_all = y_df.drop(test_all.index).copy()

# keep only series with >=70 TRAIN points, then pick 50
len_by_id = train_all.groupby("unique_id").size()
eligible = len_by_id[len_by_id >= 70].index
series_ids = eligible[:50].to_numpy()

train_df = train_all[train_all["unique_id"].isin(series_ids)].copy()
test_df = test_all[test_all["unique_id"].isin(series_ids)].copy()

print(f"[setup] eligible(>=70 train)={len(eligible)}; using={len(series_ids)}")
print(f"[setup] horizon h={H} seasonality={SEAS} freq={FREQ}")

debug_df("train_df", train_df)
debug_df("test_df", test_df)

# -----------------------------
# models
# -----------------------------
batch_size = 64
base_models = [
    Chronos(repo_id="amazon/chronos-2", batch_size=batch_size),
    TimesFM(repo_id="google/timesfm-2.5-200m-pytorch", batch_size=batch_size),
    TiRex(batch_size=batch_size),
    SeasonalNaive(),
    AutoARIMA(),
    Theta(),
    FlowState(),
]

median_ens = MedianEnsemble(models=base_models, alias="Median")
trimmed_ens = TrimmedEnsemble(models=base_models, alias="Trimmed")


# -----------------------------
# run + eval
# -----------------------------
def run_and_score(ens, name):
    print(f"\n=== running {name} ===")
    fcst = ens.forecast(df=train_df, h=H, freq="M")  # keep your call
    fcst = normalize_month_start(fcst)

    debug_forecast_output(name, fcst, ens.alias)

    # HARD alignment check (no silent NaN merges)
    merged = hard_align_or_die(test_df, fcst, ens.alias, name)
    print(f"[{name}] merge rows={len(merged)} (expected {len(test_df)})")

    # Standardized eval (sMAPE + MASE)
    monthly_mase = partial(mase, seasonality=SEAS)
    scores = evaluate(
        merged,
        metrics=[smape, monthly_mase],
        train_df=train_df,  # needed for MASE
    )

    # Extract sMAPE per series for this model
    smape_rows = scores[scores["metric"] == "smape"][
        ["unique_id", ens.alias]
    ].set_index("unique_id")
    per_series = smape_rows[ens.alias]
    overall = float(per_series.mean())

    print(f"[{name}] sMAPE overall: {overall:.2f}")

    # Optional: also print MASE overall
    mase_rows = scores[scores["metric"] == "mase"][["unique_id", ens.alias]].set_index(
        "unique_id"
    )
    print(f"[{name}] MASE overall: {float(mase_rows[ens.alias].mean()):.4f}")

    return overall, per_series, fcst


median_overall, median_per, median_fcst = run_and_score(median_ens, "MedianEnsemble")
trim_overall, trim_per, trim_fcst = run_and_score(trimmed_ens, "TrimmedEnsemble")

print("\n=== summary (sMAPE ↓ better) ===")
print(f"MedianEnsemble : {median_overall:.2f}")
print(f"TrimmedEnsemble: {trim_overall:.2f}")
