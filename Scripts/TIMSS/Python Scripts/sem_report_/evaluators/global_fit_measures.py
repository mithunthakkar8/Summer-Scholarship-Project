# sem_report/evaluators/global_fit_measures.py

from pathlib import Path
import logging
import pandas as pd
import numpy as np

from sem_report.utils.exporter import export_table
from sem_report.utils.export_config import ExportConfig
from sem_report.utils.thresholds import CBSEM_GLOBAL_FIT_THRESHOLDS

logger = logging.getLogger(__name__)


class GlobalFitEvaluator:
    """
    CB-SEM Global Fit & Latent R² evaluator.

    Computes:
      - Mean CB-SEM global fit measures
      - Stability (SD + min–max)
      - Threshold compliance
      - Latent R² mean + stability
      - Structural fidelity vs REAL
    """

    GLOBAL_FIT_METRICS = ["RMSEA", "SRMR", "CFI", "TLI", "AIC"]
    DISPLAY_NAMES = {
        "RMSEA": "RMSEA (≤0.08)",
        "SRMR":  "SRMR (≤0.08)",
        "CFI":   "CFI (≥0.90)",
        "TLI":   "TLI (≥0.90)",
        "AIC":   "AIC (↓)",
    }

    LATENT_R2 = ["SMP", "SMS"]

    def __init__(
        self,
        loader,
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

        self.section_dir = out_dir / "tables" / "007_global_fit_measures"
        self.appendix_dir = out_dir / "appendix" / "007_global_fit_measures"

        self.section_dir.mkdir(parents=True, exist_ok=True)
        self.appendix_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # PUBLIC ENTRY POINT
    # =====================================================
    def run(self):
        logger.info("Running CB-SEM Global Fit evaluator")

        fit_df = self._load_global_fit()
        r2_df = self._load_r2()

        self._export_global_fit_tables(fit_df)
        self._export_global_fit_stability()
        self._export_r2_tables(r2_df)


    # =====================================================
    # DATA LOADING
    # =====================================================
    def _load_global_fit(self) -> pd.DataFrame:
        """
        Loads CB-SEM global fit MEAN sheet and reshapes to long format.

        Expected sheet structure (wide):
        Metric | REAL | TECH1 | TECH2 | ...
        """
        df = self.loader.cb_global_fit(self.cnt)

        # First column assumed to be metric name
        metric_col = df.columns[0]

        long_df = (
            df
            .melt(
                id_vars=metric_col,
                var_name="Technique",
                value_name="Value"
            )
            .rename(columns={metric_col: "Metric"})
        )

        wide_df = (
            long_df
            .pivot_table(
                index="Technique",
                columns="Metric",
                values="Value"
            )
            .reset_index()
        )

        return wide_df


    def _load_r2(self) -> pd.DataFrame:
        """
        Loads CB-SEM latent R² MEAN sheet and reshapes.

        Expected sheet structure (wide):
        Latent | REAL | TECH1 | TECH2 | ...
        """
        df = self.loader.r_squared(self.cnt)

        latent_col = df.columns[0]

        long_df = (
            df
            .melt(
                id_vars=latent_col,
                var_name="Technique",
                value_name="R2"
            )
            .rename(columns={latent_col: "Latent"})
        )

        # -------------------------------------------------
        # NORMALIZE technique names
        # -------------------------------------------------
        long_df["Technique"] = (
            long_df["Technique"]
            .str.replace("__R2", "", regex=False)
        )

        long_df = long_df[long_df["Latent"].isin(self.LATENT_R2)]

        return long_df



    # =====================================================
    # GLOBAL FIT
    # =====================================================
    def _export_global_fit_tables(self, df: pd.DataFrame):

        # -------------------------------------------------
        # 1. Normalize structure
        # -------------------------------------------------
        tech_col = df.columns[0]
        df = df.rename(columns={tech_col: "Technique"}).copy()
        df["Technique"] = df["Technique"].str.replace("__value", "", regex=False)

        # -------------------------------------------------
        # 2. Select + rename metrics
        # -------------------------------------------------
        METRIC_MAP = {
            "rmsea": "RMSEA",
            "srmr_mplus": "SRMR",
            "cfi": "CFI",
            "tli": "TLI",
            "aic": "AIC",
        }

        mean_df = (
            df[["Technique"] + list(METRIC_MAP.keys())]
            .rename(columns=METRIC_MAP)
            .set_index("Technique")
        )

        # -------------------------------------------------
        # 3. Order rows (REAL first)
        # -------------------------------------------------
        order = ["REAL"] + [t for t in self.techniques if t in mean_df.index]
        mean_df = mean_df.loc[order]

        # -------------------------------------------------
        # 4. Rename columns to include thresholds
        # -------------------------------------------------
        mean_df = mean_df[self.GLOBAL_FIT_METRICS]
        mean_df = mean_df.rename(columns=self.DISPLAY_NAMES)

        # -------------------------------------------------
        # 5. Round to 2 decimals
        # -------------------------------------------------
        mean_df = mean_df.round(2)

        # Remove axis names that show up as weird headers
        mean_df.index.name = "Technique"
        mean_df.columns.name = None  # <- removes "Metric" in top-left

        # -------------------------------------------------
        # 6. Export
        # -------------------------------------------------
        mean_out = mean_df.reset_index()  # Technique becomes a real column

        export_table(
            mean_out,
            path=self.section_dir / "cbsem_global_fit_mean",
            config=self.export_config,
            index=False,
        )


    def _export_global_fit_stability(self):
        """
        CB-SEM Global Fit stability using RANGE (max - min),
        restricted to GLOBAL_FIT_METRICS only.
        """

        range_df = self.loader.cb_global_fit_range(self.cnt)
        if range_df is None:
            logger.info("No CB-SEM global fit RANGE sheet found; skipping stability")
            return

        # -------------------------------------------------
        # Metric name mapping (lavaan → paper)
        # -------------------------------------------------
        METRIC_MAP = {
            "rmsea": "RMSEA",
            "srmr_mplus": "SRMR",
            "cfi": "CFI",
            "tli": "TLI",
            "aic": "AIC",
        }

        metric_col = range_df.columns[0]

        # Keep only rows we care about (RAW names)
        range_df = range_df[range_df[metric_col].isin(METRIC_MAP.keys())]

        # -------------------------------------------------
        # Long format
        # -------------------------------------------------
        long_df = (
            range_df
            .melt(
                id_vars=metric_col,
                var_name="TechniqueStat",
                value_name="Value"
            )
            .rename(columns={metric_col: "MetricRaw"})
        )

        # -------------------------------------------------
        # Split technique and stat (_min / _max)
        # -------------------------------------------------
        split = long_df["TechniqueStat"].str.rsplit("_", n=1, expand=True)

        if split.shape[1] != 2:
            raise ValueError(
                "Expected column names ending with _min / _max in CB-SEM RANGE sheet"
            )

        long_df["Technique"] = split[0].str.replace("__value", "", regex=False)
        long_df["Stat"] = split[1]

        # -------------------------------------------------
        # Pivot and compute range
        # -------------------------------------------------
        pivot = (
            long_df
            .pivot_table(
                index=["MetricRaw", "Technique"],
                columns="Stat",
                values="Value"
            )
            .reset_index()
        )

        if not {"min", "max"}.issubset(pivot.columns):
            raise ValueError(
                "CB-SEM global fit RANGE sheet must contain *_min and *_max columns"
            )

        pivot["Range"] = (pivot["max"] - pivot["min"]).round(2)
        
        

        # -------------------------------------------------
        # Final tidy table
        # -------------------------------------------------
        stability_df = (
            pivot
            .assign(Metric=lambda d: d["MetricRaw"].map(METRIC_MAP))
            [["Metric", "Technique", "Range"]]
        )

        # Drop incomplete rows (e.g., missing min/max or technique)
        stability_df = stability_df.dropna(subset=["Technique", "Range"])

        # Order REAL first within each metric
        stability_df["Technique"] = pd.Categorical(
            stability_df["Technique"],
            categories=["REAL"] + self.techniques,
            ordered=True
        )

        stability_df = stability_df.sort_values(["Metric", "Technique"])
        # Remove axis name left over from pivot
        stability_df.columns.name = None

        export_table(
            stability_df,
            path=self.section_dir / "cbsem_global_fit_stability",
            config=self.export_config,
            index=False,
        )





    # =====================================================
    # R²
    # =====================================================
    def _export_r2_tables(self, df: pd.DataFrame):
        mean_df = (
            df.pivot_table(
                index="Technique",
                columns="Latent",
                values="R2",
                aggfunc="mean",
            )
            .round(3)
        )


        std_df = (
            df.pivot_table(
                index="Technique",
                columns="Latent",
                values="R2",
                aggfunc="std",
            )
            .round(3)
        )

        # Clean axis names (prevents 'Latent' / corner headers)
        mean_df.index.name = "Technique"
        mean_df.columns.name = None

        # -------------------------------------------------
        # Order rows: REAL first (baseline), then techniques
        # -------------------------------------------------
        order = ["REAL"] + [t for t in self.techniques if t in mean_df.index]
        mean_df = mean_df.loc[order]

        mean_out = mean_df.reset_index()

        export_table(
            mean_out,
            path=self.section_dir / "cbsem_r2_mean",
            config=self.export_config,
            index=False,
        )


        range_df = self.loader.r_squared_range(self.cnt) if hasattr(self.loader, "r_squared_range") else None

        if range_df is not None:
            range_df.to_excel(
                self.appendix_dir / "cbsem_r2_ranges.xlsx",
                merge_cells=False,
            )



    
