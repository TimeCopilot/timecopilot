import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/el-cacomixtle/timecopilot/")

from timecopilot.models.ensembles.median import MedianEnsemble
from timecopilot.models.ensembles.trimmed import TrimmedEnsemble
from timecopilot.models.foundation.chronos import Chronos
from timecopilot.models.foundation.timesfm import TimesFM
from timecopilot.models.foundation.tirex import TiRex
from timecopilot.models.stats import SeasonalNaive, Theta


# -----------------------------
# metric
# -----------------------------
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )


# -----------------------------
# data fetch
# -----------------------------
def ensure_m4_monthly_csvs(data_dir="data/m4"):
    os.makedirs(data_dir, exist_ok=True)
    base = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"
    files = {
        "Monthly-train.csv": f"{base}/Train/Monthly-train.csv",
        "Monthly-test.csv": f"{base}/Test/Monthly-test.csv",
    }
    for fname, url in files.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"[download] {fname}")
            urllib.request.urlretrieve(url, path)
    return (
        os.path.join(data_dir, "Monthly-train.csv"),
        os.path.join(data_dir, "Monthly-test.csv"),
    )


def melt_train_test_monthly_period(train_wide, test_wide, ids, start="2000-01"):
    """Backward-compatible wrapper around the optimized monthly melt function.

    This preserves the original API while delegating to the faster,
    vectorized implementation in ``melt_train_test_monthly_fast``.
    """
    return melt_train_test_monthly_fast(train_wide, test_wide, ids, start=start)
def melt_train_test_monthly_fast(train_wide, test_wide, ids, start="2000-01"):
    # --- train long ---
    tr = train_wide[train_wide["V1"].isin(ids)]
    tr_long = (
        tr.melt(id_vars="V1", var_name="t", value_name="y")
        .dropna()
        .rename(columns={"V1": "unique_id"})
    )
    tr_long["k"] = tr_long.groupby("unique_id").cumcount()

    # train lengths per series
    n_train = tr_long.groupby("unique_id")["k"].max().add(1).astype(int)

    # --- test long ---
    te = test_wide[test_wide["V1"].isin(ids)]
    te_long = (
        te.melt(id_vars="V1", var_name="t", value_name="y")
        .dropna()
        .rename(columns={"V1": "unique_id"})
    )
    te_long["k"] = te_long.groupby("unique_id").cumcount()
    n_test = te_long.groupby("unique_id")["k"].max().add(1).astype(int)

    # total length per series and a global calendar
    # big enough for max total length
    n_total = (n_train + n_test).astype(int)
    max_total = int(n_total.max())

    # month-end timeline once
    cal = pd.period_range(start=start, periods=max_total, freq="M").to_timestamp(
        how="end"
    )

    # assign ds by indexing into cal (vectorized via numpy take)
    tr_ds_idx = tr_long["k"].to_numpy()
    tr_long["ds"] = cal.take(tr_ds_idx)

    te_ds_idx = (te_long["k"] + te_long["unique_id"].map(n_train)).to_numpy()
    te_long["ds"] = cal.take(te_ds_idx)

    train_df = tr_long[["unique_id", "ds", "y"]]
    test_df = te_long[["unique_id", "ds", "y"]]
    return train_df, test_df


# -----------------------------
# Fix B: datetime ds + freq="M"
# -----------------------------
def melt_monthly_datetime(df, ids):
    df = df[df["V1"].isin(ids)]
    long = (
        df.melt(id_vars="V1", var_name="t", value_name="y")
        .dropna()
        .rename(columns={"V1": "unique_id"})
    )
    # integer time index per series
    long["k"] = long.groupby("unique_id").cumcount()
    # fake monthly datetimes (spacing is what's relevant here)
    long["ds"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(
        long["k"] * 30, unit="D"
    )
    long = long.drop(columns=["t", "k"])
    return long[["unique_id", "ds", "y"]]


# -----------------------------
# debug helpers
# -----------------------------
def debug_df(name, df):
    print(f"\n[{name}] shape={df.shape}")
    print(df.head(3))
    print(f"[{name}] dtypes:\n{df.dtypes}")
    print(f"[{name}] unique_id n={df['unique_id'].nunique()}")
    print(f"[{name}] ds min/max: {df['ds'].min()} -> {df['ds'].max()}")
    print(f"[{name}] y NaNs: {df['y'].isna().sum()}")
    # per-series lengths
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

    # show any quantile cols if present
    qcols = [c for c in fcst.columns if c.startswith(f"{alias}-q-")]
    if qcols:
        na_q = fcst[qcols].isna().sum().sort_values(ascending=False)
        print(f"[{name}] quantile cols: {qcols}")
        print(f"[{name}] quantile NaNs (top):\n{na_q.head(5).to_string()}")
    else:
        print(f"[{name}] no quantile columns found (ok if you didn't request quantiles).")


# -----------------------------
# run
# -----------------------------
train_path, test_path = ensure_m4_monthly_csvs()
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# lengths in the wide file = count of
# non-null values across columns V2..end
len_by_id = train.set_index("V1").notna().sum(axis=1) - 0
len_by_id = train.set_index("V1").iloc[:, 1:].notna().sum(axis=1)

eligible = len_by_id[len_by_id >= 70].index
series_ids = eligible[:50].to_numpy()  # or .sample(50, random_state=0)
print(f"[setup] eligible(>=70)={len(eligible)}; using={len(series_ids)}")

# train_df = melt_train_test_monthly_datetime(train, series_ids)
# test_df  = melt_trian_test_monthly_datetime(test, series_ids)
# Fast version works for transforming bigger amounts of data.
train_df, test_df = melt_train_test_monthly_fast(train, test, series_ids)


debug_df("train_df", train_df)
debug_df("test_df", test_df)

# horizon = test length per series
h = int(test_df.groupby("unique_id").size().iloc[0])
print(f"\n[setup] horizon h={h}")

# -----------------------------
# models
# -----------------------------
batch_size = 64
base_models = [
    Chronos(repo_id="amazon/chronos-2", batch_size=batch_size),
    TimesFM(repo_id="google/timesfm-2.5-200m-pytorch", batch_size=batch_size),
    TiRex(batch_size=batch_size),
    SeasonalNaive(),
    Theta(),
]

median_ens = MedianEnsemble(models=base_models, alias="Median")
trimmed_ens = TrimmedEnsemble(models=base_models, alias="Trimmed")


def run_and_score(ens, name):
    print(f"\n=== running {name} ===")
    fcst = ens.forecast(df=train_df, h=h, freq="M")  # Fix B call
    debug_forecast_output(name, fcst, ens.alias)

    merged = fcst.merge(
        test_df, on=["unique_id", "ds"], how="inner", suffixes=("", "_true")
    )
    print(f"[{name}] merge rows={len(merged)} (expected ~ {len(test_df)})")
    print(
        f"[{name}] merged NaNs: y={merged['y'].isna().sum()}, "
        f"pred={merged[ens.alias].isna().sum()}"
    )

    # if merge got weird (ds mismatch), this will expose it fast
    if len(merged) == 0:
        print(
            f"[{name}] ERROR: merge is empty. "
            f"ds alignment is wrong (train/test ds mismatch)."
        )
        return None, None, fcst

    # per-series smape
    per_series = merged.groupby("unique_id").apply(
        lambda x: smape(x["y"], x[ens.alias])
    )
    overall = float(per_series.mean())

    print(f"[{name}] sMAPE per series:\n{per_series.round(2).to_string()}")
    print(f"[{name}] sMAPE overall: {overall:.2f}")

    return overall, per_series, fcst


median_overall, median_per, median_fcst = run_and_score(median_ens, "MedianEnsemble")
trim_overall, trim_per, trim_fcst = run_and_score(trimmed_ens, "TrimmedEnsemble")

print("\n=== summary (sMAPE ↓ better) ===")
if median_overall is not None:
    print(f"MedianEnsemble : {median_overall:.2f}")
if trim_overall is not None:
    print(f"TrimmedEnsemble: {trim_overall:.2f}")
