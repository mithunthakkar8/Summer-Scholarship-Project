from __future__ import annotations

from pathlib import Path
import logging
import pandas as pd
import numpy as np

from sem_report.loaders.sem_comparison_loader import SEMComparisonLoader
from sem_report.utils.exporter import export_table
from sem_report.utils.export_config import ExportConfig

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


logger = logging.getLogger(__name__)


# =====================================================
# CONSTANTS
# =====================================================

EXCLUDED_INDICATORS = set()

CONSTRUCT_PREFIX = {
    "ACM": "BTBG06",   
    "TCI": "BTBS",
    "LEI": "BTBG",
    "NSC": "BTBG10",
    "SSF": "BT"
}



# =====================================================
# LOADINGS EVALUATOR
# =====================================================

class LoadingsEvaluator:
    """
    Loadings evaluation vs REAL.

    Outputs:
      (a) Mean Absolute Loading Error by construct
      (b) |Δ loading| matrix (indicator × technique)
      (c) Loading range width + sign flip flags
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

        # Output folders
        self.tbl_dir = out_dir / "tables" / "006_loadings"
        self.fig_dir = out_dir / "figures" / "006_loadings"

        self.tbl_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------

    @staticmethod
    def _filter_valid(df: pd.DataFrame) -> pd.DataFrame:
        return df[~df["Indicator"].isin(EXCLUDED_INDICATORS)].copy()

    # -------------------------------------------------
    # (a) MAE by construct
    # -------------------------------------------------
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

    def compute_mae_by_construct(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        df = self._filter_valid(df)

        for tech in self.techniques:
            row = {"Technique": tech}

            for construct, prefix in CONSTRUCT_PREFIX.items():
                if construct == "SSF":
                    sub = df[
                        (df["Construct"] == "SSF")
                        & (df["Indicator"].str.endswith("GSOS"))
                    ]
                else:
                    sub = df[df["Construct"] == construct]


                if sub.empty:
                    row[construct] = np.nan
                else:
                    row[construct] = (
                        sub[f"{self.tech_to_col(tech)}__Loading"] - sub["REAL__Loading"]
                    ).abs().mean()

            rows.append(row)

        return pd.DataFrame(rows).set_index("Technique")

    # -------------------------------------------------
    # (b) |Δ loading| matrix
    # -------------------------------------------------

    def compute_loading_delta_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._filter_valid(df)

        mat = pd.DataFrame(
            index=df["Indicator"],
            columns=self.techniques,
            dtype=float
        )

        for tech in self.techniques:
            mat[tech] = (
                df[f"{self.tech_to_col(tech)}__Loading"] - df["REAL__Loading"]
            ).abs().values

        return mat

    # -------------------------------------------------
    # (c) Loading stability (range + sign flip)
    # -------------------------------------------------

    def compute_loading_ranges(
        self, range_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        range_df = self._filter_valid(range_df).set_index("Indicator")

        width = pd.DataFrame(index=range_df.index)
        flip = pd.DataFrame(index=range_df.index)

        for tech in self.techniques:
            lo = range_df[f"{self.tech_to_col(tech)}__Loading_min"]
            hi = range_df[f"{self.tech_to_col(tech)}__Loading_max"]

            width[tech] = hi - lo
            flip[tech] = (lo < 0) & (hi > 0)


        return width, flip
    
    def plot_loading_delta_heatmap(self, delta: pd.DataFrame):
        """
        Loading preservation heatmap.
        Color = |Δ loading|
        """
        fig, ax = plt.subplots(
            figsize=(0.6 * len(delta.columns) + 4, 0.25 * len(delta) + 4)
        )

        # Use only numeric columns for plotting
        num_df = delta.select_dtypes(include=[np.number])

        im = ax.imshow(num_df.values, aspect="auto")

        # Annotate values
        for i in range(num_df.shape[0]):
            for j in range(num_df.shape[1]):
                val = num_df.iloc[i, j]
                if pd.notna(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )

        ax.set_xticks(range(len(num_df.columns)))
        ax.set_xticklabels(num_df.columns, rotation=45, ha="right")

        ax.set_yticks(range(len(num_df.index)))
        ax.set_yticklabels(delta["Indicator"])

        ax.set_title("Loading preservation heatmap (absolute error)")
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("|Δ loading|", rotation=90)

        fig.tight_layout()

        out_base = self.out_dir / "figures" / "006_loadings" / "loading_delta_heatmap"
        out_base.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(out_base.with_suffix(".png"), dpi=300)
        fig.savefig(out_base.with_suffix(".pdf"))
        plt.close(fig)

    def plot_loading_range_heatmap(
        self,
        width: pd.DataFrame,
        flip: pd.DataFrame,
    ):
        """
        Loading stability heatmap.
        Color = range width
        Overlay ⚠ where sign flip occurs.
        """
        fig, ax = plt.subplots(
            figsize=(0.6 * len(width.columns) + 4, 0.25 * len(width) + 4)
        )

        # Use numeric-only data for plotting
        num_df = width.select_dtypes(include=[np.number])

        im = ax.imshow(num_df.values, aspect="auto")



        ax.set_xticks(range(len(num_df.columns)))
        ax.set_xticklabels(num_df.columns, rotation=45, ha="right")

        ax.set_yticks(range(len(num_df)))
        ax.set_yticklabels(width["Indicator"])


        ax.set_title("Loading Stability Across Replicates (Range Width)")

        # Overlay sign-flip warning symbols
        for i in range(num_df.shape[0]):
            indicator = width["Indicator"].iloc[i]

            for j, tech in enumerate(num_df.columns):
                flag = flip.loc[
                    flip["Indicator"] == indicator,
                    tech
                ].values

                if len(flag) and flag[0]:
                    ax.text(
                        j,
                        i,
                        "⚠",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="black",
                    )



        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Range width", rotation=90)

        fig.tight_layout()

        out_base = self.out_dir / "figures" / "006_loadings" / "loading_range_heatmap"
        out_base.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(out_base.with_suffix(".png"), dpi=300)
        fig.savefig(out_base.with_suffix(".pdf"))
        plt.close(fig)


    # -------------------------------------------------
    # Public runner
    # -------------------------------------------------

    def run(self):
        logger.info("Running section: Loadings")

        # Defensive: ensure tables_dir exists
        if not hasattr(self, "tables_dir"):
            self.tables_dir = self.out_dir / "tables" / "006_loadings"
            self.tables_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------
        # Load data
        # ------------------------------
        mean_df = self.loader.loadings(self.cnt)

        try:
            range_df = self.loader.load_sheet(
                f"pls_sem_loadings_R2_{self.cnt}",
                stat="range"
            )
        except ValueError:
            logger.warning("No loadings RANGE sheet found")
            range_df = None

        # ------------------------------
        # (a) MAE by construct
        # ------------------------------
        mae = (
            self.compute_mae_by_construct(mean_df)
            .reset_index()
            .rename(columns={"index": "Technique"})
        )

        export_table(
            df=mae,
            path=self.tables_dir / "mean_absolute_loading_error",
            config=self.export_config,
            table_number=20,
            title="Mean Absolute Loading Error by Construct",
            note="Values report the mean absolute difference between synthetic and real factor loadings for each construct.",
            index=False,
        )

        # ------------------------------
        # (b) Δ loading matrix
        # ------------------------------
        delta = (
            self.compute_loading_delta_matrix(mean_df)
            .reset_index()
            .rename(columns={"index": "Indicator"})
        )

        export_table(
            df=delta,
            path=self.tables_dir / "loading_delta_matrix",
            config=self.export_config,
            table_number=21,
            title="Absolute Loading Differences by Indicator and Technique",
            note="Each cell reports the absolute difference between synthetic and real factor loadings.",
            index=False,
        )

        self.plot_loading_delta_heatmap(delta)

        # ------------------------------
        # (c) Range width + sign flip
        # ------------------------------
        if range_df is not None:
            width, flip = self.compute_loading_ranges(range_df)

            width = (
                width
                .reset_index()
                .rename(columns={"index": "Indicator"})
            )

            export_table(
                df=width,
                path=self.tables_dir / "loading_range_width",
                config=self.export_config,
                table_number=22,
                title="Loading Range Width Across Replications",
                note="Range width is computed as the difference between maximum and minimum loading values across replications.",
                index=False,
            )


            flip = (
                flip
                .fillna(False)
                .astype(int)
                .reset_index()
                .rename(columns={"index": "Indicator"})
            )

            export_table(
                df=flip,
                path=self.tables_dir / "loading_sign_flip",
                config=self.export_config,
                table_number=23,
                title="Loading Sign Flip Indicators Across Replications",
                note="Values indicate whether the loading interval spans zero, implying a sign change across replications.",
                index=False,
            )

            self.plot_loading_range_heatmap(width, flip.fillna(False))
