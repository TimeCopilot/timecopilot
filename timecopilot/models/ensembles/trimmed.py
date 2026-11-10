import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from ... import TimeCopilotForecaster
from ..utils.forecaster import Forecaster, QuantileConverter


class TrimmedEnsemble(Forecaster):
    """
    TrimmedEnsemble (alternate ensemble to MedianEnsemble)

    Purpose
    -------
    A robust ensemble that aggregates model forecasts using a
    *trimmed mean* for quantiles
    (and optionally for point forecasts), with safety rails:

      1) Fixed trimming per row (unique_id, ds): we first compute the
        *minimum* number of
         available contributors across all requested quantiles for that row (n_min).
         Then we decide how much to trim based on n_min, and apply the same trimming
         intensity to every quantile in that row.

      2) Minimum contributor quota: if n_min < min_quota, we fall back to the *median*
         aggregation for quantiles for that row (and skip isotonic repair).

      3) Monotone quantiles: when a full quantile vector exists (no NaNs) and the row
         did not fallback, we run isotonic regression to enforce:
             q10 <= q50 <= q90 <= ...
         This is a "repair" step only for monotonicity, not a modeling step.

    Notes
    -----
    - This ensemble tolerates missing quantile columns per model (point-only models).
    - When quantiles include 0.5, point forecast is set to the ensemble
    q50 for coherence.
    """

    def __init__(
        self,
        models: list[Forecaster],
        alias: str = "TrimmedEnsemble",
        min_quota: int = 2,
        trim_low_threshold: int = 8,  # n_min >= this -> switch to lower trim percentage
        trim_10p_threshold: int | None = None,
    ):
        self.tcf = TimeCopilotForecaster(models=models, fallback_model=None)
        self.alias = alias
        self.min_quota = int(min_quota)
        # Backward-compatible handling: `trim_10p_threshold` (deprecated) overrides
        # `trim_low_threshold` if explicitly provided.
        effective_threshold = (
            trim_10p_threshold if trim_10p_threshold is not None else trim_low_threshold
        )
        self.trim_low_threshold = int(effective_threshold)
        # Preserve old attribute name for any existing internal uses.
        self.trim_10p_threshold = self.trim_low_threshold

    # ---------- trimming policy (fixed per row based on n_min) ----------

    def _trim_k_from_nmin(self, n_min: int) -> int:
        """
        Decide how many values to trim from each tail (k) given n_min contributors.

        Rule agreed:
          - n_min 3–4  -> trim 1 each side
          - n_min 5–7  -> trim 20%
          - n_min >= 8 -> trim 10%  (simple fixed choice; avoids a "10–20%" ambiguity)

        Returns:
            k (int): how many to drop from each tail.
        """
        if n_min <= 7:
            return 0
        if n_min <= 8:
            return 1
        if n_min <= 9:
            return int(np.floor(0.20 * n_min))
        return int(np.floor(0.10 * n_min))

    @staticmethod
    def _trimmed_mean_1d(values: np.ndarray, k: int) -> float:
        """
        Trim k from each tail, then mean. Ignores NaNs.

        If trimming would remove everything (2k >= n), we fall back to plain mean.
        """
        x = values.astype(float)
        x = x[~np.isnan(x)]
        n = x.size
        if n == 0:
            return np.nan
        if k <= 0 or (2 * k) >= n:
            return float(np.mean(x))
        x.sort()
        return float(np.mean(x[k : n - k]))

    @staticmethod
    def _nanmedian_1d(values: np.ndarray) -> float:
        """Median ignoring NaNs; returns NaN if all values are NaN."""
        x = values.astype(float)
        return float(np.nanmedian(x)) if np.any(~np.isnan(x)) else np.nan

    # ---------- main API ----------

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        qc = QuantileConverter(level=level, quantiles=quantiles)

        # Call all models; merged output includes each model alias column (point),
        # and (if provided by the model) alias-q-{pct} columns for each quantile.
        _fcst_df = self.tcf._call_models(
            "forecast",
            merge_on=["unique_id", "ds"],
            df=df,
            h=h,
            freq=freq,
            level=None,
            quantiles=qc.quantiles,
        )

        fcst_df = _fcst_df[["unique_id", "ds"]].copy()
        model_cols = [m.alias for m in self.tcf.models]

        # Point forecast:
        # Keep median for robustness (same as MedianEnsemble baseline).
        # If q50 is requested later, we overwrite with ensemble q50 for coherence.
        fcst_df[self.alias] = _fcst_df[model_cols].median(axis=1)

        # No probabilistic output requested -> done.
        if qc.quantiles is None:
            return fcst_df

        # Quantile setup
        qs = sorted(qc.quantiles)
        q_cols = [f"{self.alias}-q-{int(q * 100)}" for q in qs]

        # Map pct -> existing per-model quantile columns (some may be missing)
        models_q_cols_map: dict[int, list[str]] = {}
        for q in qs:
            pct = int(q * 100)
            expected = [f"{alias}-q-{pct}" for alias in model_cols]
            models_q_cols_map[pct] = [c for c in expected if c in _fcst_df.columns]

        n_rows = len(_fcst_df)
        fallback_mask = np.zeros(n_rows, dtype=bool)
        k_by_row = np.zeros(n_rows, dtype=int)

        # Decide trimming ONCE per row:
        # - Compute n_min = min contributors across requested
        # quantiles (after NaN filtering)
        # - If n_min < min_quota -> fallback to median for ALL quantiles in that row
        # - Else compute k = trim_k_from_nmin(n_min)
        for i in range(n_rows):
            counts = []
            row_idx = _fcst_df.index[i]

            for q in qs:
                pct = int(q * 100)
                cols_here = models_q_cols_map[pct]
                if not cols_here:
                    counts.append(0)
                    continue
                vals = _fcst_df.loc[row_idx, cols_here].to_numpy(dtype=float)
                counts.append(int(np.sum(~np.isnan(vals))))

            n_min = int(min(counts)) if counts else 0

            if n_min < self.min_quota:
                fallback_mask[i] = True
                k_by_row[i] = 0
            else:
                k_by_row[i] = self._trim_k_from_nmin(n_min)

        # Aggregate quantiles (trimmed mean if not fallback; otherwise median)
        for q in qs:
            pct = int(q * 100)
            cols_here = models_q_cols_map[pct]
            out_col = f"{self.alias}-q-{pct}"

            if not cols_here:
                # Nobody produced this quantile column.
                fcst_df[out_col] = np.nan
                continue

            vals_mat = _fcst_df[cols_here].to_numpy(dtype=float)
            out = np.empty(n_rows, dtype=float)

            for i in range(n_rows):
                if fallback_mask[i]:
                    out[i] = self._nanmedian_1d(vals_mat[i])
                else:
                    out[i] = self._trimmed_mean_1d(vals_mat[i], k=int(k_by_row[i]))

            fcst_df[out_col] = out

        # Isotonic monotonicity repair:
        # Only valid when:
        #   - row did NOT fallback, AND
        #   - all requested quantiles exist for that row
        # (no NaNs in the ensemble quantiles)
        # Otherwise: skip (do not "repair" partial/broken vectors).
        ir = IsotonicRegression(increasing=True)
        q_vals = fcst_df[q_cols].to_numpy(dtype=float)
        repaired = q_vals.copy()

        for i in range(n_rows):
            if fallback_mask[i]:
                continue
            if np.any(np.isnan(repaired[i])):
                continue
            repaired[i] = ir.fit_transform(qs, repaired[i])

        fcst_df[q_cols] = repaired

        # If q50 requested, make point forecast equal to median quantile output.
        if 0.5 in qc.quantiles:
            fcst_df[self.alias] = fcst_df[f"{self.alias}-q-50"].values

        # One-line disclosure if any fallback occurred.
        n_fallback = int(fallback_mask.sum())
        if n_fallback > 0:
            print(
                f"{self.alias}: quantiles fallback->median \
                for {n_fallback}/{n_rows} rows "
                f"(min_quota={self.min_quota}); isotonic \
                skipped on fallback/NaN rows."
            )

        # Convert quantiles to levels if user requested `level=...`
        fcst_df = qc.maybe_convert_quantiles_to_level(fcst_df, models=[self.alias])
        return fcst_df
