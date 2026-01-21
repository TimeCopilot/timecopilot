from __future__ import annotations

import timecopilot
from timecopilot import TimeCopilotForecaster
from timecopilot.models.stats import SeasonalNaive, Theta
from timecopilot.models.foundation.chronos import Chronos

from timecopilot.models.prophet import Prophet
from timecopilot.models.stats import AutoARIMA, AutoETS, SeasonalNaive
from timecopilot.models.foundation.moirai import Moirai

from pathlib import Path
from typing import Iterable

import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
MDA_DIR = Path("data/mda")   # <-- change
MTR_DIR = Path("data/mtr")   # <-- change
GLOB = "*.csv"               # or "*.txt"
ENCODING = "utf-8"           # try "latin-1" if needed

COL_DATE = "Fecha"
COL_HOUR = "Hora"
COL_NODE = "Clave del nodo"
COL_PML = "Precio marginal local ($/MWh)"

KEEP_NODES = {
    "01CEI-230",  # VDM Norte
    "03GDU-230",  # Guadalajara
    "05GPL-230",  # Laguna
    "06HUI-230",  # Monterrey
    "04HLI-230",  # Hermosillo
}


# ----------------------------
# IO: CENACE CSV parsing
# ----------------------------
def _find_header_line(fp: Path) -> int:
    """
    CENACE files have a variable-length text header before the CSV header line.
    We find the line that contains the CSV header and return how many rows to skip.
    """
    with fp.open("r", encoding=ENCODING, errors="replace") as f:
        for i, line in enumerate(f):
            if "Fecha" in line and "Hora" in line and "Clave del nodo" in line:
                return i
    raise ValueError(f"Could not find CSV header line in: {fp}")


def read_cenace_folder(folder: Path, label: str) -> pd.DataFrame:
    files = sorted(folder.glob(GLOB))
    if not files:
        raise FileNotFoundError(f"No files found in {folder.resolve()} matching {GLOB}")

    print(f"[READ] {label.upper()} folder={folder.resolve()} files={len(files)}")

    dfs: list[pd.DataFrame] = []
    for k, fp in enumerate(files, start=1):
        skiprows = _find_header_line(fp)

        df = pd.read_csv(fp, skiprows=skiprows, encoding=ENCODING)

        # Normalize column names (CENACE often has leading spaces)
        df.columns = df.columns.str.replace('"', "", regex=False).str.strip()

        # Keep only what we need
        missing = {COL_DATE, COL_HOUR, COL_NODE, COL_PML} - set(df.columns)
        if missing:
            raise ValueError(f"[READ:{label}] Missing columns {missing} in {fp.name}. Got={list(df.columns)}")

        df = df[[COL_DATE, COL_HOUR, COL_NODE, COL_PML]].copy()

        # Clean values
        df[COL_NODE] = df[COL_NODE].astype(str).str.strip()
        df = df[df[COL_NODE].isin(KEEP_NODES)]
        if df.empty:
            if k % 25 == 0 or k == len(files):
                print(f"  [READ:{label}] {k}/{len(files)} {fp.name}: 0 rows after node filter")
            continue

        df[COL_PML] = pd.to_numeric(df[COL_PML], errors="coerce")
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce").dt.date
        df[COL_HOUR] = pd.to_numeric(df[COL_HOUR], errors="coerce").astype("Int64")

        df = df.dropna(subset=[COL_DATE, COL_HOUR, COL_NODE, COL_PML])

        dfs.append(df)

        if k % 25 == 0 or k == len(files):
            print(f"  [READ:{label}] {k}/{len(files)} {fp.name}: kept_rows={len(df):,}")

    if not dfs:
        raise ValueError(f"[READ] No rows found for label={label} after filtering. Check KEEP_NODES and files.")

    out = pd.concat(dfs, ignore_index=True)

    # Dedupe
    out = (
        out.sort_values([COL_NODE, COL_DATE, COL_HOUR])
        .drop_duplicates([COL_NODE, COL_DATE, COL_HOUR], keep="last")
        .reset_index(drop=True)
    )

    out = out.rename(columns={COL_PML: f"pml_{label}"})

    print(
        f"[READ] {label.upper()} done: rows={len(out):,} nodes={out[COL_NODE].nunique():,} "
        f"date_range={out[COL_DATE].min()}..{out[COL_DATE].max()}"
    )
    return out


def build_spread_df() -> pd.DataFrame:
    mda = read_cenace_folder(MDA_DIR, "mda")
    mtr = read_cenace_folder(MTR_DIR, "mtr")

    print("[MERGE] MDA x MTR by (node,date,hour)")
    merged = mda.merge(
        mtr,
        on=[COL_NODE, COL_DATE, COL_HOUR],
        how="inner",
        validate="one_to_one",
    )

    # CENACE hour is 1..24; map hour=1 -> 00:00; hour=24 -> 23:00
    merged["datetime"] = pd.to_datetime(merged[COL_DATE]) + pd.to_timedelta(
        merged[COL_HOUR].astype(int) - 1, unit="h"
    )

    merged["spread"] = merged["pml_mtr"] - merged["pml_mda"]
    merged = merged.sort_values([COL_NODE, "datetime"]).reset_index(drop=True)

    print(
        f"[MERGE] done: rows={len(merged):,} nodes={merged[COL_NODE].nunique():,} "
        f"dt_range={merged['datetime'].min()}..{merged['datetime'].max()}"
    )
    return merged


def transform_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["datetime", COL_NODE, "spread"]].copy()
    out.columns = ["ds", "unique_id", "y"]
    out["ds"] = pd.to_datetime(out["ds"])
    return out


# ----------------------------
# Forecasting: TimeCopilot daily execution (72h, keep last 24h)
# ----------------------------
# sign vote ensemble works as an alternative to the median ensemble and is
# enforced due to sign precission importance in the spread forecasting process
def _sign_vote_ensemble(fcst: pd.DataFrame) -> pd.DataFrame:
    """
    Ensemble across model point forecasts by:
      - sign: majority vote on sign(model_yhat)
      - magnitude: median(abs(model_yhat))
      - output: voted_sign * median_abs

    Requires fcst to be "wide": one row per (unique_id, ds) and one column per model forecast.
    """
    if not {"ds", "unique_id"} <= set(fcst.columns):
        raise ValueError(f"Forecast output missing ds/unique_id. Got columns={list(fcst.columns)}")

    # Numeric columns (candidate model outputs)
    numeric_cols = [
        c for c in fcst.columns
        if c not in ("ds", "unique_id") and pd.api.types.is_numeric_dtype(fcst[c])
    ]
    if not numeric_cols:
        raise ValueError(f"No numeric forecast columns found. Got columns={list(fcst.columns)}")

    # Drop likely interval columns if present (we want point forecasts only)
    drop_like = ("lo", "hi", "lower", "upper", "pi_", "level")
    point_cols = [c for c in numeric_cols if not any(tok in c.lower() for tok in drop_like)]
    if not point_cols:
        point_cols = numeric_cols  # fallback

    vals = fcst[point_cols]

    # --- sign vote ---
    # sign matrix: -1, 0, +1
    sign_mat = vals.apply(lambda col: col.map(lambda v: 0 if pd.isna(v) else (1 if v > 0 else (-1 if v < 0 else 0))))
    vote = sign_mat.sum(axis=1)

    # voted sign: +1 if vote>0, -1 if vote<0, tie -> 0 (or fallback rule below)
    voted_sign = vote.map(lambda v: 1 if v > 0 else (-1 if v < 0 else 0))

    # Tie-breaker (optional but recommended):
    # If vote==0, use sign of the median point forecast (still robust)
    median_point = vals.median(axis=1)
    tie_mask = voted_sign == 0
    if tie_mask.any():
        voted_sign.loc[tie_mask] = median_point.loc[tie_mask].map(lambda v: 0 if pd.isna(v) else (1 if v > 0 else (-1 if v < 0 else 0)))

    # --- magnitude (robust size) ---
    mag = vals.abs().median(axis=1)

    out = fcst[["unique_id", "ds"]].copy()
    out["yhat_spread"] = voted_sign.astype(float) * mag.astype(float)

    # Optional debug columns (handy while testing); comment out if you want it lean
    out["vote_sum"] = vote
    out["voted_sign"] = voted_sign
    out["mag_med_abs"] = mag

    return out

def _median_point_forecast(fcst: pd.DataFrame) -> pd.DataFrame:
    """
    TimeCopilot output may contain multiple model point columns.
    We take the median across numeric forecast columns (excluding ds/unique_id and interval cols).
    """
    if not {"ds", "unique_id"} <= set(fcst.columns):
        raise ValueError(f"Forecast output missing ds/unique_id. Got columns={list(fcst.columns)}")

    # keep numeric columns only
    numeric_cols = [c for c in fcst.columns if c not in ("ds", "unique_id") and pd.api.types.is_numeric_dtype(fcst[c])]

    if not numeric_cols:
        raise ValueError(f"No numeric forecast columns found to take median over. Got columns={list(fcst.columns)}")

    # Drop likely interval columns if present (common patterns)
    drop_like = ("lo", "hi", "lower", "upper", "pi_", "level")
    point_cols = [c for c in numeric_cols if not any(tok in c.lower() for tok in drop_like)]
    if not point_cols:
        # fallback: just use numeric_cols
        point_cols = numeric_cols

    out = fcst[["unique_id", "ds"]].copy()
    out["yhat_spread"] = fcst[point_cols].median(axis=1)
    return out

def _sign_series(s: pd.Series) -> pd.Series:
    # returns -1, 0, +1
    return s.apply(lambda v: 0 if pd.isna(v) else (1 if v > 0 else (-1 if v < 0 else 0)))


def _mcc_binary(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Matthews correlation coefficient for binary labels in {0,1}.
    Returns NaN if undefined (e.g., all one class).
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return float("nan")
    return (tp * tn - fp * fn) / (denom ** 0.5)

def daily_executor(
    transformed: pd.DataFrame,
    cutoff_ts: pd.Timestamp,
    target_day: pd.Timestamp,
    *,
    h: int = 72,
    level: Iterable[int] = (80, 90),
    save_csv: bool = False,
) -> pd.DataFrame:
    """
    Use history up to cutoff_ts (inclusive), forecast h=72,
    and return ONLY the 24 hours inside target_day (00:00..23:00),
    using MEDIAN across model point forecasts.
    """
    cutoff_ts = pd.Timestamp(cutoff_ts)
    target_day = pd.Timestamp(target_day).normalize()

    day_start = target_day
    day_end = target_day + pd.Timedelta(hours=23)

    hist = transformed.loc[transformed["ds"] <= cutoff_ts].copy()
    if hist.empty:
        raise ValueError(f"No history available at or before cutoff_ts={cutoff_ts}")

    # ---- TimeCopilot setup (assumes these are available in your env)
    tcf = TimeCopilotForecaster(
        models=[
            Chronos(repo_id="amazon/chronos-bolt-mini"),
            Theta(),
            AutoETS(),
            Moirai(),
            Prophet(),
            AutoARIMA(),
            SeasonalNaive()
        ]
    )

    fcst = tcf.forecast(df=hist, h=h)

    # Median point forecast across model columns
    # out = _median_point_forecast(fcst)
    out = _sign_vote_ensemble(fcst)
    out["cutoff_ts"] = cutoff_ts
    out["target_day"] = target_day

    # Keep only the target-day 24 hours
    out = out[(out["ds"] >= day_start) & (out["ds"] <= day_end)].copy()

    if save_csv and not out.empty:
        out.to_csv(f"{target_day.date()}_last24_median.csv", index=False)

    return out


def run_months(transformed: pd.DataFrame, start_day: str, end_day: str) -> pd.DataFrame:
    """
    Executes daily runs for each target day in [start_day, end_day] inclusive,
    using cutoff = target_day - 72 hours.
    Collects ONLY the target-day 24h slice per run.
    """
    start = pd.Timestamp(start_day).normalize()
    end = pd.Timestamp(end_day).normalize()
    days = pd.date_range(start, end, freq="D")

    print(f"[RUN] days={len(days)} range={start.date()}..{end.date()} (cutoff = day - 72h)")
    all_24: list[pd.DataFrame] = []

    for i, target_day in enumerate(days, start=1):
        cutoff_ts = target_day - pd.Timedelta(hours=72)

        print(f"[RUN] {i:03d}/{len(days)} target_day={target_day.date()} cutoff={cutoff_ts}")

        try:
            fcst_24 = daily_executor(
                transformed=transformed,
                cutoff_ts=cutoff_ts,
                target_day=target_day,
                h=72,
                level=(80, 90),
                save_csv=False,
            )

            if fcst_24.empty:
                print(f"  [RUN] -> 0 rows (forecast did not cover target day or data gap)")
                continue

            got = len(fcst_24)
            nodes = fcst_24["unique_id"].nunique()
            print(f"  [RUN] -> kept last24 rows={got:,} nodes={nodes:,}")

            all_24.append(fcst_24)

        except Exception as e:
            print(f"  [WARN] failed target_day={target_day.date()} cutoff={cutoff_ts}: {e}")

    if not all_24:
        print("[RUN] No forecast rows collected.")
        return pd.DataFrame(columns=["unique_id", "ds", "yhat_spread", "cutoff_ts", "target_day"])

    out = pd.concat(all_24, ignore_index=True)
    print(f"[RUN] done: collected_rows={len(out):,} days_with_data={out['target_day'].nunique():,}")
    return out

def evaluate_forecasts(fcst_24_df: pd.DataFrame, actuals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...
    joined = fcst_24_df.merge(actuals, on=["unique_id", "ds"], how="left")

    # --- SIGN METRICS (add here) ---
    joined["sign_y"] = _sign_series(joined["y"])
    joined["sign_yhat"] = _sign_series(joined["yhat_spread"])
    joined["sign_hit"] = (joined["sign_y"] == joined["sign_yhat"])

    # Optional: ignore exact zeros (common if you want "direction" not "flat")
    # joined_nonzero = joined[(joined["sign_y"] != 0) & (joined["sign_yhat"] != 0) & joined["y"].notna()].copy()

    # For sign scoring, we should only score rows where actual y exists
    sign_scored = joined.dropna(subset=["y"]).copy()

    # Binary view for MCC: drop zeros, map negative->0, positive->1
    nz = sign_scored[(sign_scored["sign_y"] != 0) & (sign_scored["sign_yhat"] != 0)].copy()
    if not nz.empty:
        y_true_bin = (nz["sign_y"] > 0).astype(int)
        y_pred_bin = (nz["sign_yhat"] > 0).astype(int)
        overall_mcc = _mcc_binary(y_true_bin, y_pred_bin)
    else:
        overall_mcc = float("nan")

    # --- existing error columns ---
    joined["err"] = joined["yhat_spread"] - joined["y"]
    joined["abs_err"] = joined["err"].abs()
    joined["sq_err"] = joined["err"] ** 2

    scored = joined.dropna(subset=["y"]).copy()
    if scored.empty:
        metrics = pd.DataFrame(columns=["unique_id", "MAE", "RMSE", "n", "sign_acc", "recall_pos", "recall_neg", "mcc_sign"])
        return joined, metrics

    # --- per-node metrics (add sign metrics) ---
    def _recall_pos(g: pd.DataFrame) -> float:
        actual_pos = g[g["y"] > 0]
        if len(actual_pos) == 0:
            return float("nan")
        return (actual_pos["yhat_spread"] > 0).mean()

    def _recall_neg(g: pd.DataFrame) -> float:
        actual_neg = g[g["y"] < 0]
        if len(actual_neg) == 0:
            return float("nan")
        return (actual_neg["yhat_spread"] < 0).mean()

    def _mcc_sign_group(g: pd.DataFrame) -> float:
        g = g.dropna(subset=["y"]).copy()
        g["sy"] = _sign_series(g["y"])
        g["syh"] = _sign_series(g["yhat_spread"])
        g = g[(g["sy"] != 0) & (g["syh"] != 0)]
        if g.empty:
            return float("nan")
        y_true = (g["sy"] > 0).astype(int)
        y_pred = (g["syh"] > 0).astype(int)
        return _mcc_binary(y_true, y_pred)

    metrics_by_node = (
        scored.groupby("unique_id", as_index=False)
        .agg(
            MAE=("abs_err", "mean"),
            RMSE=("sq_err", lambda s: (s.mean()) ** 0.5),
            n=("abs_err", "size"),
            sign_acc=("sign_hit", "mean"),
        )
        .merge(
            scored.groupby("unique_id").apply(_recall_pos).rename("recall_pos"),
            on="unique_id",
            how="left",
        )
        .merge(
            scored.groupby("unique_id").apply(_recall_neg).rename("recall_neg"),
            on="unique_id",
            how="left",
        )
        .merge(
            scored.groupby("unique_id").apply(_mcc_sign_group).rename("mcc_sign"),
            on="unique_id",
            how="left",
        )
        .sort_values("sign_acc", ascending=False)
        .reset_index(drop=True)
    )

    overall = pd.DataFrame([{
        "unique_id": "__overall__",
        "MAE": scored["abs_err"].mean(),
        "RMSE": (scored["sq_err"].mean()) ** 0.5,
        "n": int(scored["abs_err"].size),
        "sign_acc": sign_scored["sign_hit"].mean(),
        "recall_pos": _recall_pos(sign_scored),
        "recall_neg": _recall_neg(sign_scored),
        "mcc_sign": overall_mcc,
    }])

    metrics = pd.concat([overall, metrics_by_node], ignore_index=True)
    return joined, metrics

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    # 1) Build spread dataframe from raw CENACE
    spread_df = build_spread_df()

    # 2) Transform into TimeCopilot format
    transformed = transform_df(spread_df)

    print(
        f"[DATA] transformed rows={len(transformed):,} nodes={transformed['unique_id'].nunique():,} "
        f"ds_range={transformed['ds'].min()}..{transformed['ds'].max()}"
    )

    # 3) Run daily executions for Nov+Dec 2025
    #    (for Nov 1, cutoff = Oct 29 00:00, matching your “start Oct 29” logic)
    fcst_24_df = run_months(transformed, "2025-11-01", "2025-12-31")

    # 4) Evaluate against real spread (already in transformed)
    joined_df, metrics_df = evaluate_forecasts(fcst_24_df, transformed)

    print("\n[METRICS] Top rows:")
    print(metrics_df.head(20).to_string(index=False))

    # Optional saves
    # fcst_24_df.to_parquet("spread_fcst_last24_median_NovDec.parquet", index=False)
    # joined_df.to_parquet("spread_fcst_eval_NovDec.parquet", index=False)
    # metrics_df.to_csv("spread_fcst_metrics_median_NovDec.csv", index=False)

