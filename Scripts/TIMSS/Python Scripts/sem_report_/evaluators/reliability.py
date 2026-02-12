from __future__ import annotations

from pathlib import Path
import logging
import pandas as pd
import numpy as np


from sem_report.loaders.sem_comparison_loader import SEMComparisonLoader
from sem_report.utils.exporter import export_table
from sem_report.utils.export_config import ExportConfig
from sem_report.utils.thresholds import (
    RELIABILITY_THRESHOLDS,
    RELIABILITY_RANGE_THRESHOLDS,
)

# Display symbols (plain Unicode, black by default)
PASS = "✓"
FAIL = "✗"
NA = "—"


logger = logging.getLogger(__name__)


METRIC_COLS = {
    "alpha": "alpha",
    "rhoC": "rhoC",
    "rhoA": "rhoA",
    "AVE": "AVE",
}


# =====================================================
# RELIABILITY EVALUATOR
# =====================================================

class ReliabilityEvaluator:
    """
    Reliability evaluation vs REAL.

    Implements:
      (a) Threshold benchmarking
      (b) Mean Absolute Deviation vs REAL
      (c) Stability across replicates (range-based)
    """

    def __init__(
        self,
        loader: SEMComparisonLoader,
        cnt: str,
        techniques: list[str],
        out_dir: Path,
        export_config: ExportConfig,
    ):
        self.loader = loader
        self.cnt = cnt
        self.techniques = techniques
        self.out_dir = out_dir
        self.export_config = export_config

        self.tables_dir = out_dir / "tables" / "003_reliability"
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------

    @staticmethod
    def _long_format(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts wide REAL / TECH__ columns to long format.
        """
        id_cols = ["Construct"]
        value_cols = [c for c in df.columns if "__" in c]

        long = df.melt(
            id_vars=id_cols,
            value_vars=value_cols,
            var_name="Source",
            value_name="Value",
        )

        long["Technique"] = long["Source"].str.split("__").str[0]
        long["Metric"] = long["Source"].str.split("__").str[1]

        return long.drop(columns="Source")

    # -------------------------------------------------
    # (a) Threshold benchmarking
    # -------------------------------------------------

    def benchmark_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for _, row in df.iterrows():
            construct = row["Construct"]

            for tech in ["REAL"] + self.techniques:
                entry = {
                    "Construct": construct,
                    "Technique": tech,
                }

                for short, col in METRIC_COLS.items():
                    val = row.get(f"{tech}__{col}", np.nan)
                    thr = RELIABILITY_THRESHOLDS[short]

                    if pd.isna(val):
                        entry[short] = NA
                    elif val >= thr:
                        entry[short] = PASS
                    else:
                        entry[short] = FAIL

                rows.append(entry)

        # ---- build table AFTER all constructs are processed ----
        out = pd.DataFrame(rows)

        out = out.rename(columns={
            "alpha": "α ≥ .7",
            "rhoC": "ρC ≥ .7",
            "rhoA": "ρA ≥ .7",
            "AVE": "AVE ≥ .5",
        })

        return out


    # -------------------------------------------------
    # (b) Mean Absolute Deviation vs REAL
    # -------------------------------------------------

    def mean_absolute_deviation(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for _, row in df.iterrows():
            construct = row["Construct"]

            real_vals = {
                k: row.get(f"REAL__{v}", np.nan)
                for k, v in METRIC_COLS.items()
            }


            for tech in self.techniques:
                entry = {
                    "Construct": construct,
                    "Technique": tech,
                }

                for short, col in METRIC_COLS.items():
                    syn = row.get(f"{tech}__{col}", np.nan)
                    real = real_vals[short]

                    entry[f"Δ{short}"] = (
                        abs(syn - real)
                        if not (pd.isna(syn) or pd.isna(real))
                        else np.nan
                    )

                rows.append(entry)

        df = pd.DataFrame(rows)

        num_cols = [c for c in df.columns if c.startswith("Δ")]
        df[num_cols] = df[num_cols].round(3)

        return df


    # -------------------------------------------------
    # (c) Stability across replicates
    # -------------------------------------------------

    def stability_from_ranges(self, range_df: pd.DataFrame) -> pd.DataFrame:
        rows = []

        for _, row in range_df.iterrows():
            construct = row["Construct"]

            for tech in self.techniques:
                entry = {
                    "Construct": construct,
                    "Technique": tech,
                }

                unstable = []

                for short, col in METRIC_COLS.items():
                    lo = row.get(f"{tech}__{col}_min", np.nan)
                    hi = row.get(f"{tech}__{col}_max", np.nan)

                    rng = hi - lo if not (pd.isna(lo) or pd.isna(hi)) else np.nan
                    entry[f"{short}_Range"] = rng

                    thr = RELIABILITY_RANGE_THRESHOLDS[short]
                    if not pd.isna(rng) and rng > thr:
                        unstable.append(short)

                rows.append(entry)

        df = pd.DataFrame(rows)

        range_cols = [c for c in df.columns if c.endswith("_Range")]
        df[range_cols] = df[range_cols].round(3)

        return df


    # -------------------------------------------------
    # Public runner
    # -------------------------------------------------

    def run(self):
        logger.info("Running section: Reliability")

        mean_df = self.loader.reliability(self.cnt)

        try:
            range_df = self.loader.reliability_range(self.cnt)
        except Exception:
            range_df = None

        # (a) Threshold benchmarking
        bench = self.benchmark_thresholds(mean_df)
        export_table(
            bench,
            path=self.tables_dir / "reliability_thresholds",
            config=self.export_config,
            index=False,
        )

        # (b) Mean absolute deviation
        mad = self.mean_absolute_deviation(mean_df)
        export_table(
            mad,
            path=self.tables_dir / "reliability_mad_vs_real",
            config=self.export_config,
            index=False,
        )

        # (c) Stability
        if range_df is not None:
            stability = self.stability_from_ranges(range_df)
            export_table(
                stability,
                path=self.tables_dir / "reliability_stability",
                config=self.export_config,
                index=False,
            )
