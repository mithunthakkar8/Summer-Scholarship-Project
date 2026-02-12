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

EXCLUDED_INDICATORS = {
    "ESCS",          # single indicator construct
    "AGE",
    "ST004D01T",     # gender
    "ST001D01T",     # grade
    "MCLSIZE",       # classSize
    "SCHSIZE",       # schoolSize
    "IMMIG",
    "MISCED",        # motherEdu
}

CONSTRUCT_PREFIX = {
    "SMP": "PV",
    "SMS": "ST268",
    "SPI": "SC064",
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

    def compute_mae_by_construct(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        df = self._filter_valid(df)

        for tech in self.techniques:
            row = {"Technique": tech}

            for construct, prefix in CONSTRUCT_PREFIX.items():
                sub = df[
                    (df["Construct"] == construct)
                    & (df["Indicator"].str.startswith(prefix))
                ]

                if sub.empty:
                    row[construct] = np.nan
                else:
                    row[construct] = (
                        sub[f"{tech}__Loading"] - sub["REAL__Loading"]
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
                df[f"{tech}__Loading"] - df["REAL__Loading"]
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
            lo = range_df[f"{tech}__Loading_min"]
            hi = range_df[f"{tech}__Loading_max"]

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

        im = ax.imshow(delta.values, aspect="auto")

        # Annotate values
        for i in range(delta.shape[0]):
            for j in range(delta.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{delta.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )


        ax.set_xticks(range(len(delta.columns)))
        ax.set_xticklabels(delta.columns, rotation=45, ha="right")

        ax.set_yticks(range(len(delta.index)))
        ax.set_yticklabels(delta.index)

        ax.set_title("Loading Preservation Heatmap (|Δ loading|)")
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

        im = ax.imshow(width.values, aspect="auto")


        ax.set_xticks(range(len(width.columns)))
        ax.set_xticklabels(width.columns, rotation=45, ha="right")

        ax.set_yticks(range(len(width.index)))
        ax.set_yticklabels(width.index)

        ax.set_title("Loading Stability Across Replicates (Range Width)")

        # Overlay sign-flip warning symbols
        for i in range(width.shape[0]):
            for j in range(width.shape[1]):
                if flip.iloc[i, j]:
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
        mae = self.compute_mae_by_construct(mean_df)

        mae.index.name = None
        export_table(
            mae,
            path=self.tables_dir / "mean_absolute_loading_error",
            config=self.export_config,
            index=True,
        )

        # ------------------------------
        # (b) Δ loading matrix
        # ------------------------------
        delta = self.compute_loading_delta_matrix(mean_df)

        delta.index.name = None
        export_table(
            delta,
            path=self.tables_dir / "loading_delta_matrix",
            config=self.export_config,
            index=True,
        )

        self.plot_loading_delta_heatmap(delta)

        # ------------------------------
        # (c) Range width + sign flip
        # ------------------------------
        if range_df is not None:
            width, flip = self.compute_loading_ranges(range_df)

            width.index.name = None
            export_table(
                width,
                path=self.tables_dir / "loading_range_width",
                config=self.export_config,
                index=True,
            )

            flip.index.name = None
            export_table(
                flip.fillna(False).astype(int),
                path=self.tables_dir / "loading_sign_flip",
                config=self.export_config,
                index=True,
            )

            self.plot_loading_range_heatmap(width, flip.fillna(False))
