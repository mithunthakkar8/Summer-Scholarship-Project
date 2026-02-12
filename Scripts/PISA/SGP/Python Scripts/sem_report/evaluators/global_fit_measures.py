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

        # Normalize
        tech_col = df.columns[0]
        df = df.rename(columns={tech_col: "Technique"}).copy()
        df["Technique"] = df["Technique"].str.replace("__value", "", regex=False)

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
            .round(2)
        )

        # ---- pivot: metrics as rows ----
        out = (
            mean_df
            .T
            .reset_index()
            .rename(columns={"index": "Metric"})
        )

        # Apply display names with thresholds
        out["Metric"] = out["Metric"].map(self.DISPLAY_NAMES)

        # SAFETY: ensure no stray Technique column survives
        out = out.drop(columns=["Technique"], errors="ignore")

        tech_cols = sorted(c for c in out.columns if c != "Metric")
        out = out[["Metric"] + tech_cols]

        # IMPORTANT: remove pandas metadata that HTML exporter renders
        out.index.name = None
        out.columns.name = None


        export_table(
            df=out,
            path=self.section_dir / "cbsem_global_fit_mean",
            config=self.export_config,
            table_number=17,
            title="CB-SEM Global Fit Indices (Mean Across Replications)",
            note="Values report mean global fit indices for each technique. Recommended thresholds are shown in parentheses.",
            index=False,
        )


    def _export_global_fit_stability(self):

        range_df = self.loader.cb_global_fit_range(self.cnt)
        if range_df is None:
            logger.info("No CB-SEM global fit RANGE sheet found; skipping stability")
            return

        METRIC_MAP = {
            "rmsea": "RMSEA",
            "srmr_mplus": "SRMR",
            "cfi": "CFI",
            "tli": "TLI",
            "aic": "AIC",
        }

        metric_col = range_df.columns[0]
        range_df = range_df[range_df[metric_col].isin(METRIC_MAP.keys())]

        long_df = (
            range_df
            .melt(
                id_vars=metric_col,
                var_name="TechStat",
                value_name="Value"
            )
            .rename(columns={metric_col: "MetricRaw"})
        )

        split = long_df["TechStat"].str.rsplit("_", n=1, expand=True)
        long_df["Technique"] = split[0].str.replace("__value", "", regex=False)
        long_df["Stat"] = split[1]

        pivot = (
            long_df
            .pivot_table(
                index=["MetricRaw", "Technique"],
                columns="Stat",
                values="Value"
            )
            .reset_index()
        )

        pivot["Range"] = (pivot["max"] - pivot["min"]).round(2)

        tidy = (
            pivot
            .assign(
                Metric=lambda d: d["MetricRaw"].map(
                    lambda m: f"Range ({METRIC_MAP[m]})"
                )
            )
            [["Metric", "Technique", "Range"]]
        )

        out = (
            tidy
            .pivot(index="Metric", columns="Technique", values="Range")
            .reset_index()
        )

        tech_cols = sorted(c for c in out.columns if c != "Metric")
        out = out[["Metric"] + tech_cols]

        export_table(
            df=out,
            path=self.section_dir / "cbsem_global_fit_stability",
            config=self.export_config,
            table_number=18,
            title="CB-SEM Global Fit Stability Across Replications",
            note="Ranges are computed as the difference between maximum and minimum observed fit indices across replications.",
            index=False,
        )




    # =====================================================
    # R²
    # =====================================================
    def _export_r2_tables(self, df: pd.DataFrame):

        required = {"Latent", "Technique", "R2"}
        if not required.issubset(df.columns):
            raise ValueError(
                "R² table must be in long format with columns "
                "['Latent', 'Technique', 'R2']"
            )

        mean_df = (
            df.pivot_table(
                index="Technique",
                columns="Latent",
                values="R2",
                aggfunc="mean",
            )
        )

        std_df = (
            df.pivot_table(
                index="Technique",
                columns="Latent",
                values="R2",
                aggfunc=lambda x: x.std(ddof=0) if len(x) > 1 else 0.0,
            )
        )


        records = []

        for latent in self.LATENT_R2:
            if latent not in mean_df.columns:
                continue

            row_mu = {"Metric": rf"$\mu(R^2_{{{latent}}})$"}

            for tech in mean_df.index:
                row_mu[tech] = round(mean_df.loc[tech, latent], 3)

            records.extend([row_mu])

        out = pd.DataFrame(records)

        tech_cols = sorted(c for c in out.columns if c != "Metric")
        out = out[["Metric"] + tech_cols]

        export_table(
            df=out,
            path=self.section_dir / "cbsem_r2_summary",
            config=self.export_config,
            table_number=19,
            title="Latent Variable R² for CB-SEM Structural Paths",
            note="Values report mean explained variance (R²) for each latent endogenous construct.",
            index=False,
        )

