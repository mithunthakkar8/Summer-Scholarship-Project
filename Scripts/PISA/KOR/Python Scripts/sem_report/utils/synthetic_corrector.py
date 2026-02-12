import os
import glob
import shutil
import pandas as pd
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class SyntheticDataCorrector:
    """
    Cleans, validates, and corrects synthetic CSV files
    per technique directory. Writes corrected CSVs and
    correction logs.
    """

    # ============================================================
    # RULES (class-level, explicit, auditable)
    # ============================================================

    LIKERT_4_COLS = ["ST268Q01JA", "ST268Q04JA", "ST268Q07JA"]

    PROP_0_100_COLS = [
        "SC064Q05WA", "SC064Q06WA", "SC064Q01TA", "SC064Q02TA",
        "SC064Q04NA", "SC064Q03TA", "SC064Q07WA"
    ]

    CATEGORICAL_RULES = {
        "ST004D01T": [1, 2],
        "IMMIG": [1, 2, 3]
    }

    ORDINAL_RANGE_COLS = {
        "ST001D01T": (7, 99),
        "MISCED": (1, 10),
    }


    

    # ============================================================
    # CODEBOOK CONTINUOUS BOUNDS
    # ============================================================

    CODEBOOK_CONTINUOUS_BOUNDS = {
        "AGE": (15.17, 16.42),
        "ESCS": (-6.84, 7.38),
        "SCHSIZE": (1.0, 19201.0),
        "MCLSIZE": (13.0, 53.0),

        # Plausible values
        "PV1MATH": (0.0, 943.04),
        "PV2MATH": (45.95, 933.84),
        "PV3MATH": (55.54, 946.79),
        "PV4MATH": (0.0, 911.46),
        "PV5MATH": (46.92, 896.69),
        "PV6MATH": (3.28, 909.52),
        "PV7MATH": (52.6, 928.84),
        "PV8MATH": (0.0, 912.73),
        "PV9MATH": (39.8, 903.02),
        "PV10MATH": (33.1, 915.23),
    }


    # ============================================================
    # Public API
    # ============================================================

    def __init__(self, technique_paths: dict, max_rows: int, real_df: pd.DataFrame):
        self.technique_paths = technique_paths
        self.max_rows = max_rows
        self.real_df = real_df
        self._drop_stats = []

    def run(self):
        logger.info("=" * 60)
        logger.info("Running synthetic data correction pipeline")

        for tech, base_path in self.technique_paths.items():
            self._process_technique(tech, base_path)

        # ---------------------------------
        # NEW: write global drop summary
        # ---------------------------------
        self._write_drop_summary()

        logger.info("Synthetic correction complete")


    # ============================================================
    # Internal logic
    # ============================================================

    def _process_technique(self, tech: str, base_path: str):

        logger.info(f"[CORRECT] {tech}")

        if not os.path.exists(base_path):
            logger.warning(f"[SKIP] Directory not found: {base_path}")
            return

        self._delete_old_files(base_path, "*_corrected.csv")
        self._delete_old_files(base_path, "correction_log_*.csv")

        csv_files = self._get_csv_files(base_path)
        if not csv_files:
            logger.warning("No CSV files found")
            return

        log_rows = []

        for csv_path in csv_files:
            fname = os.path.basename(csv_path)
            logger.info(f"Processing: {fname}")

            df = pd.read_csv(csv_path)
            n_before = len(df)

            df_corr = self._correct_dataframe(df, log_rows, fname, tech)
            n_after = len(df_corr)

            out_path = csv_path.replace(".csv", "_corrected.csv")
            df_corr.to_csv(out_path, index=False)

            # ---------------------------------
            # NEW: record drop stats
            # ---------------------------------
            self._drop_stats.append({
                "Technique": tech,
                "File": fname,
                "Rows_Before": n_before,
                "Rows_After": n_after,
                "Rows_Dropped": n_before - n_after,
                "Pct_Dropped": (
                    100 * (n_before - n_after) / n_before
                    if n_before > 0 else np.nan
                )
            })

        self._move_rep_files(base_path)

        if log_rows:
            log_df = pd.DataFrame(log_rows)
            log_path = os.path.join(base_path, f"correction_log_{tech}.csv")
            log_df.to_csv(log_path, index=False)
            logger.info(f"Log written: {log_path}")

    # ============================================================
    # Helpers
    # ============================================================

    def _write_drop_summary(self):
        if not self._drop_stats:
            logger.warning("No drop statistics collected")
            return

        df = pd.DataFrame(self._drop_stats)

        summary = (
            df
            .groupby("Technique")
            .agg(
                Runs=("File", "count"),
                Mean_Rows_Before=("Rows_Before", "mean"),
                Mean_Rows_After=("Rows_After", "mean"),
                Mean_Rows_Dropped=("Rows_Dropped", "mean"),
                Mean_Pct_Dropped=("Pct_Dropped", "mean"),
            )
            .round(2)
            .reset_index()
        )

        # Central, auditable location
        out_path = Path("tables/000_summary/table_rows_dropped_after_correction.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        summary.to_csv(out_path, index=False)

        logger.info(f"Row-drop summary written to: {out_path}")


    # ============================================================
    # REAL-based PV bounds
    # ============================================================

    def _get_real_continuous_bounds(self):
        """
        Compute unified REAL ±1 SD bounds clipped to codebook limits.
        Cached after first computation.
        """
        if hasattr(self, "_real_cont_bounds"):
            return self._real_cont_bounds

        bounds = {}

        for c, (cb_lo, cb_hi) in self.CODEBOOK_CONTINUOUS_BOUNDS.items():
            if c not in self.real_df.columns:
                logger.warning(f"[REAL STATS] Missing {c} in real_df")
                continue

            v = pd.to_numeric(self.real_df[c], errors="coerce").dropna()
            if v.empty:
                continue

            real_min = v.min()
            real_max = v.max()
            real_std = v.std(ddof=1)

            lo = max(cb_lo, real_min - real_std)
            hi = min(cb_hi, real_max + real_std)

            bounds[c] = {
                "lo": lo,
                "hi": hi,
                "real_min": real_min,
                "real_max": real_max,
                "real_std": real_std,
                "cb_min": cb_lo,
                "cb_max": cb_hi,
            }

        self._real_cont_bounds = bounds
        return bounds


    @staticmethod
    def _get_csv_files(base_path):
        return [
            f for f in glob.glob(os.path.join(base_path, "*.csv"))
            if (
                not f.endswith("_corrected.csv")
                and "correction_log" not in f.lower()
            )
        ]

    @staticmethod
    def _delete_old_files(base_path, pattern):
        for f in glob.glob(os.path.join(base_path, pattern)):
            os.remove(f)

    @staticmethod
    def _move_rep_files(base_path):
        old_reps = os.path.join(base_path, "old reps")
        os.makedirs(old_reps, exist_ok=True)

        for f in glob.glob(os.path.join(base_path, "*.csv")):
            name = os.path.basename(f).lower()
            if "corrected" not in name:
                shutil.move(f, os.path.join(old_reps, os.path.basename(f)))

    # ============================================================
    # Core correction
    # ============================================================

    def _correct_dataframe(self, df, log_rows, fname, tech):

        df = df.copy()
        df = df.dropna()

        def log(var, action, pct):
            log_rows.append({
                "technique": tech,
                "file": fname,
                "variable": var,
                "action": action,
                "percent_rows_affected": round(100 * pct, 3)
            })

        def drop_and_log(mask, var, reason):
            nonlocal df
            if mask.any():
                log(var, f"drop_rows_{reason}", mask.mean())
                df = df.loc[~mask].copy()

        # ------------------------------------------------------------
        # Ordinal Likert (1–4)
        # ------------------------------------------------------------
        for c in self.LIKERT_4_COLS:
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce").round()
                drop_and_log((v < 1) | (v > 4), c, "invalid_likert")
                df[c] = v.astype("Int64").astype("category")

        # ------------------------------------------------------------
        # Proportions [0, 100]
        # ------------------------------------------------------------
        for c in self.PROP_0_100_COLS:
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce")
                drop_and_log((v < 0) | (v > 100), c, "invalid_prop")
                df[c] = v

        # ------------------------------------------------------------
        # Nominal / ordinal categorical (codebook)
        # ------------------------------------------------------------
        for c, valid in self.CATEGORICAL_RULES.items():
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce").round()
                drop_and_log(~v.isin(valid), c, "invalid_category")
                df[c] = v.astype("Int64").astype("category")

        for c, (lo, hi) in self.ORDINAL_RANGE_COLS.items():
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce").round()
                drop_and_log((v < lo) | (v > hi), c, "invalid_ordinal")
                df[c] = v.astype("Int64").astype("category")


        # ------------------------------------------------------------
        # Unified continuous variables (codebook + REAL ±1 SD)
        # ------------------------------------------------------------
        cont_bounds = self._get_real_continuous_bounds()

        for c, b in cont_bounds.items():
            if c not in df.columns:
                continue

            v = pd.to_numeric(df[c], errors="coerce")
            mask = (v < b["lo"]) | (v > b["hi"])
            drop_and_log(mask, c, "invalid_continuous_codebook_real_1sd")
            df[c] = v

        # ------------------------------------------------------------
        # Enforce REAL row-count upper bound
        # ------------------------------------------------------------
        if len(df) > self.max_rows:
            df = df.iloc[: self.max_rows].copy()

        return df


