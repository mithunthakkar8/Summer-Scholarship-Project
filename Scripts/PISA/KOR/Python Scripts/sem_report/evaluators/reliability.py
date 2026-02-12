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

        metric_labels = {
            "alpha": "α ≥ .7",
            "rhoC": "ρC ≥ .7",
            "rhoA": "ρA ≥ .7",
            "AVE": "AVE ≥ .5",
        }



        # ---- INCLUDE REAL explicitly for threshold table ----
        techs = ["REAL"] + [t for t in self.techniques if t != "REAL"]

        for _, row in df.iterrows():
            construct = row["Construct"]

            for short, col in METRIC_COLS.items():
                thr = RELIABILITY_THRESHOLDS[short]

                out_row = {
                    "Construct": construct,
                    "Metric": metric_labels[short],
                }

                for tech in techs:
                    val = row.get(f"{self.tech_to_col(tech)}__{col}", np.nan)

                    if pd.isna(val):
                        out_row[tech] = NA
                    elif val >= thr:
                        out_row[tech] = PASS
                    else:
                        out_row[tech] = FAIL

                rows.append(out_row)

        out = pd.DataFrame(rows)

        col_order = ["Construct", "Metric"] + techs
        return out[col_order]

    def tech_to_col(self, tech: str) -> str:
        """
        Convert display technique name to dataframe-safe column prefix.
        Example:
        'GReaT (DistilGPT2)' -> 'GReaT_DistilGPT2'
        """
        return (
            tech
            .replace(" (", "_")
            .replace(")", "")
        )

    # -------------------------------------------------
    # (b) Mean Absolute Deviation vs REAL
    # -------------------------------------------------

    def mean_absolute_deviation(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        DELTA_LABELS = {
            "alpha": "Absolute error (α)",
            "rhoC": "Absolute error (ρC)",
            "rhoA": "Absolute error (ρA)",
            "AVE": "Absolute error (AVE)",
        }

        for _, row in df.iterrows():
            construct = row["Construct"]

            for short, col in METRIC_COLS.items():
                real = row.get(f"REAL__{col}", np.nan)

                out_row = {
                    "Construct": construct,
                    "Metric": DELTA_LABELS[short],
                }

                for tech in self.techniques:
                    syn = row.get(f"{self.tech_to_col(tech)}__{col}", np.nan)
                    out_row[tech] = (
                        abs(syn - real)
                        if not (pd.isna(syn) or pd.isna(real))
                        else np.nan
                    )

                rows.append(out_row)

        out = pd.DataFrame(rows)

        # round only numeric technique columns
        tech_cols = [t for t in self.techniques if t in out.columns]
        out[tech_cols] = out[tech_cols].round(2)

        return out



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
                    lo = row.get(f"{self.tech_to_col(tech)}__{col}_min", np.nan)
                    hi = row.get(f"{self.tech_to_col(tech)}__{col}_max", np.nan)

                    rng = hi - lo if not (pd.isna(lo) or pd.isna(hi)) else np.nan
                    entry[f"{short}_Range"] = rng

                    thr = RELIABILITY_RANGE_THRESHOLDS[short]
                    if not pd.isna(rng) and rng > thr:
                        unstable.append(short)

                rows.append(entry)

        df = pd.DataFrame(rows)

        range_cols = [c for c in df.columns if c.endswith("_Range")]
        df[range_cols] = df[range_cols].round(2)

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

        # CSV / BOTH
        export_table(
            df=bench,
            path=self.tables_dir / "reliability_thresholds",
            config=self.export_config,
            table_number=29,
            title="Reliability Threshold Compliance by Construct",
            note="Check marks indicate compliance with recommended reliability thresholds (α, ρC, ρA ≥ .70; AVE ≥ .50).",
            index=False,
        )


        # HTML-only: bold Metric column
        if self.export_config.export_html:
            bench_html = bench.copy()
            

            bench_html.index.name = None
            bench_html.columns.name = None

            from sem_report.utils.export_config import ExportFormat

            export_table(
                df=bench,
                path=self.tables_dir / "reliability_thresholds",
                config=self.export_config,
                table_number=29,
                title="Reliability Threshold Compliance by Construct",
                note="Check marks indicate compliance with recommended reliability thresholds (α, ρC, ρA ≥ .70; AVE ≥ .50).",
                index=False,
            )



        # (b) Mean absolute deviation
        mad = self.mean_absolute_deviation(mean_df)

        # CSV / BOTH
        export_table(
            df=mad,
            path=self.tables_dir / "reliability_mad_vs_real",
            config=self.export_config,
            table_number=30,
            title="Mean Absolute Deviation of Reliability Metrics from Real Data",
            note="Δ values represent absolute deviations between synthetic and real reliability estimates.",
            index=False,
        )


        # HTML-only: bold Metric column
        if self.export_config.export_html:
            mad_html = mad.copy()

            # avoid index/column-name leakage
            mad_html.index.name = None
            mad_html.columns.name = None

            from sem_report.utils.export_config import ExportFormat

            export_table(
                df=mad_html,
                path=self.tables_dir / "reliability_mad_vs_real",
                config=self.export_config,
                table_number=30,
                title="Mean Absolute Deviation of Reliability Metrics from Real Data",
                note="Δ values represent absolute deviations between synthetic and real reliability estimates.",
                index=False,
            )



        # (c) Stability
        if range_df is not None:
            stability = self.stability_from_ranges(range_df)
            export_table(
                df=stability,
                path=self.tables_dir / "reliability_stability",
                config=self.export_config,
                table_number=31,
                title="Stability of Reliability Metrics Across Synthetic Replicates",
                note="Reported values correspond to observed metric ranges across replicates; larger ranges indicate lower stability.",
                index=False,
            )

