from __future__ import annotations

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sem_report.loaders.sem_comparison_loader import SEMComparisonLoader
from sem_report.utils.exporter import export_table
from sem_report.utils.export_config import ExportConfig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

MAX_PLOTS = 20  # safety cap
MAX_ROWS_PER_COL = 6   # tweakable: 5–8 works well
SUBPLOT_HEIGHT = 3.1  # inches per subplot
SUBPLOT_WIDTH  = 4.5


class StructuralPathEvaluator:
    """
    Evaluates standardized structural path fidelity vs REAL.

    Outputs:
      (a) Standardized β table
      (b) Directional consistency + rank preservation
      (c) REAL-in-range overlap table
      (d) Interval plots per key path
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

        # ---------------------------------
        # Output structure (fixed layout)
        # ---------------------------------
        self.appendix_dir = out_dir / "appendix" / "005_paths"
        self.tables_dir   = out_dir / "tables"   / "005_paths"
        self.figures_dir  = out_dir / "figures"  / "005_paths"

        for d in (self.appendix_dir, self.tables_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)


    # ==========================================================
    # Public API
    # ==========================================================

    def run(self):
        logger.info("Running StructuralPathEvaluator")

        # --------------------------------------------------
        # Correct loader access (DO NOT use raw prefixes)
        # --------------------------------------------------
        cnt = self.loader.path.stem.split("_")[-1] if hasattr(self.loader, "cnt") is False else self.loader.cnt

        mean_df = self.loader.standardized_paths(self.cnt)
        range_df = self.loader.standardized_paths_range(self.cnt)

        if range_df is None:
            logger.warning("No RANGE sheet found for standardized paths")

        beta_table = self._build_beta_table(mean_df, range_df)
        export_table(
            df=beta_table,
            path=self.tables_dir / "standardized_path_betas",
            config=self.export_config,
            table_number=24,
            title="Standardized Structural Path Coefficients",
            note="Entries report standardized β coefficients; asterisks denote statistical significance (* p < .05, ** p < .01).",
            index=False,
        )

        summary_df = self._direction_and_rank_summary(mean_df)

        export_table(
            df=summary_df,
            path=self.tables_dir / "path_direction_rank_summary",
            config=self.export_config,
            table_number=25,
            title="Directional Consistency and Rank Preservation of Structural Paths",
            note="Directional consistency reports the proportion of paths with matching signs relative to real data; rank preservation is measured using Spearman’s ρ.",
            index=False,
        )


        if range_df is not None:
            overlap_table = self._range_overlap(mean_df, range_df)
            export_table(
                df=overlap_table,
                path=self.tables_dir / "path_range_overlap",
                config=self.export_config,
                table_number=26,
                title="Overlap of Synthetic Path Intervals with Real Coefficients",
                note="Intervals are defined as mean ± bootstrap standard deviation; check marks indicate containment of the real coefficient.",
                index=False,
            )


        paths = mean_df["Path"].tolist()

        if len(paths) > MAX_PLOTS:
            logger.warning(
                f"Too many paths ({len(paths)}). "
                f"Limiting plots to first {MAX_PLOTS}."
            )

        paths = paths[:MAX_PLOTS]

        self.plot_bar_grid(mean_df)

        if range_df is not None:
            self.plot_interval_grid(mean_df, range_df)

        logger.info("StructuralPathEvaluator completed")

    @staticmethod
    def _compute_grid(n_items: int, max_rows: int):
        n_cols = int(np.ceil(n_items / max_rows))
        n_rows = min(n_items, max_rows)
        return n_rows, n_cols

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

    def plot_bar_grid(self, df: pd.DataFrame):
        paths = df["Path"].tolist()
        n_paths = len(paths)

        n_rows, n_cols = self._compute_grid(n_paths, MAX_ROWS_PER_COL)

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(
                n_cols * SUBPLOT_WIDTH,
                n_rows * SUBPLOT_HEIGHT * 1.25  # ← height boost
            ),
            squeeze=False
        )

        labels = ["REAL"] + self.techniques

        for i, (_, r) in enumerate(df.iterrows()):
            row = i % n_rows
            col = i // n_rows
            ax = axes[row][col]

            values = [r["REAL__Std_B"]] + [
                r[f"{self.tech_to_col(tech)}__Std_B"] for tech in self.techniques
            ]

            # Bar plot
            ax.bar(range(len(labels)), values)
            ax.axhline(0, linewidth=0.8, color="black")

            # X-axis ticks: explicit positions + smaller font + right alignment
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(
                labels,
                rotation=90,
                ha="right",      # ← CRITICAL
                fontsize=7       # ← smaller than before
            )

            # Title
            ax.set_title(r["Path"], fontsize=10, pad=10)

        # Hide unused axes
        for j in range(i + 1, n_rows * n_cols):
            axes[j % n_rows][j // n_rows].axis("off")

        fig.suptitle("Standardized Path Coefficients (β)", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        fig.savefig(self.figures_dir / "paths_bar_grid.png")
        fig.savefig(self.figures_dir / "paths_bar_grid.pdf")
        plt.close(fig)


    def plot_interval_grid(self, mean_df: pd.DataFrame, range_df: pd.DataFrame):
        paths = mean_df["Path"].tolist()
        n_paths = len(paths)

        n_rows, n_cols = self._compute_grid(n_paths, MAX_ROWS_PER_COL)
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(
                n_cols * SUBPLOT_WIDTH,
                n_rows * SUBPLOT_HEIGHT * 1.25
            ),
            squeeze=False
        )

        for i, (_, r_mean) in enumerate(mean_df.iterrows()):
            row = i % n_rows
            col = i // n_rows
            ax = axes[row][col]

            r_range = range_df.loc[
                range_df.Path == r_mean["Path"]
            ].iloc[0]

            real = r_mean["REAL__Std_B"]
            ys = np.arange(len(self.techniques))

            for j, tech in enumerate(self.techniques):
                mean = r_mean[f"{self.tech_to_col(tech)}__Std_B"]
                sd = r_range[f"{self.tech_to_col(tech)}__Bootstrap_SD_max"]

                ax.hlines(j, mean - sd, mean + sd)
                ax.plot(mean, j, "o")

            ax.axvline(real, linestyle="--", linewidth=1)
            ax.set_yticks(ys)
            ax.set_yticklabels(self.techniques, fontsize=8)
            ax.set_title(r_mean["Path"], fontsize=9)

        # Hide unused axes
        for j in range(i + 1, n_rows * n_cols):
            axes[j % n_rows][j // n_rows].axis("off")

        fig.suptitle("Path Coefficient Intervals vs REAL", fontsize=14)
        fig.subplots_adjust(
            hspace=1.0,   # ← MORE vertical gap
            wspace=0.3    # unchanged / optional
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        fig.savefig(self.figures_dir / "paths_interval_grid.png")
        fig.savefig(self.figures_dir / "paths_interval_grid.pdf")
        plt.close(fig)

    # ==========================================================
    # (a) Standardized β table
    # ==========================================================

    def _build_beta_table(
        self,
        mean_df: pd.DataFrame,
        range_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        rows = []

        for _, r in mean_df.iterrows():

            row = {
                "Path": r["Path"],
                "REAL β": self._format_beta_with_sig(
                    r["REAL__Std_B"],
                    r["REAL__Std_p"],
                ),
            }

            # lookup row for SDs (if available)
            r_range = None
            if range_df is not None:
                r_range = range_df.loc[
                    range_df.Path == r["Path"]
                ].iloc[0]

            for tech in self.techniques:
                mean = r[f"{self.tech_to_col(tech)}__Std_B"]
                pval = r[f"{self.tech_to_col(tech)}__Std_p"]

                if r_range is not None:
                    sd = r_range.get(f"{self.tech_to_col(tech)}__Bootstrap_SD_max", np.nan)
                else:
                    sd = np.nan

                row[tech] = self._format_beta_mean_sd_with_sig(
                    mean=mean,
                    sd=sd,
                    pval=pval,
                )

            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def _format_beta_with_sig(beta: float, pval: float) -> str:
        """
        Format standardized beta with significance annotation.
        """
        if pd.isna(beta):
            return ""

        if pd.isna(pval):
            sig = ""
        elif pval < 0.01:
            sig = "**"
        elif pval < 0.05:
            sig = "*"
        else:
            sig = "ns"

        return f"{beta:.3f} {sig}"


    @staticmethod
    def _format_beta_mean_sd_with_sig(
        mean: float,
        sd: float,
        pval: float,
    ) -> str:
        """
        Format standardized beta as: mean ± sd with significance.
        """
        if pd.isna(mean):
            return ""

        # significance
        if pd.isna(pval):
            sig = ""
        elif pval < 0.01:
            sig = "**"
        elif pval < 0.05:
            sig = "*"
        else:
            sig = "ns"

        # SD optional (REAL will not have it)
        if pd.isna(sd):
            return f"{mean:.3f} {sig}"

        return f"{mean:.3f} ± {sd:.3f} {sig}"


    def _direction_and_rank_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Directional consistency and rank preservation summary.

        Output:
            rows = metrics
            columns = techniques
        """
        real_beta = df["REAL__Std_B"]
        real_rank = real_beta.abs().rank(ascending=False)
        n_paths = len(df)

        metrics = {
            "Directional Consistency": {},
            "Rank Preservation (Spearman ρ)": {},
        }

        for tech in self.techniques:
            synth_beta = df[f"{self.tech_to_col(tech)}__Std_B"]

            # ----------------------------
            # Directional consistency
            # ----------------------------
            correct = (np.sign(synth_beta) == np.sign(real_beta)).sum()
            directional_consistency = correct / n_paths

            # ----------------------------
            # Rank preservation (Spearman)
            # ----------------------------
            synth_rank = synth_beta.abs().rank(ascending=False)
            rho = real_rank.corr(synth_rank, method="spearman")

            metrics["Directional Consistency"][tech] = round(directional_consistency, 3)
            metrics["Rank Preservation (Spearman ρ)"][tech] = (
                round(rho, 3) if pd.notna(rho) else np.nan
            )

        out = (
            pd.DataFrame(metrics)
            .T
            .reset_index()
            .rename(columns={"index": "Metric"})
        )

        # deterministic column order
        out = out[["Metric"] + self.techniques]

        return out


    # ==========================================================
    # (c) Range overlap with REAL
    # ==========================================================

    def _range_overlap(
        self,
        mean_df: pd.DataFrame,
        range_df: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []

        for _, r in mean_df.iterrows():

            row = {
                "Path": r["Path"],
                "REAL β": round(r["REAL__Std_B"], 2) if pd.notna(r["REAL__Std_B"]) else np.nan,
            }

            r_range = range_df.loc[range_df.Path == r["Path"]].iloc[0]

            for tech in self.techniques:
                mean = r[f"{self.tech_to_col(tech)}__Std_B"]
                sd = r_range[f"{self.tech_to_col(tech)}__Bootstrap_SD_max"]

                lo, hi = mean - sd, mean + sd
                ok = lo <= row["REAL β"] <= hi

                row[tech] = f"[{lo:.2f}, {hi:.2f}] {'✓' if ok else '✗'}"

            rows.append(row)

        return pd.DataFrame(rows)

    # ==========================================================
    # (d) Interval plot
    # ==========================================================

    def _plot_interval(
        self,
        mean_df: pd.DataFrame,
        range_df: pd.DataFrame,
        path: str,
    ):
        r_mean = mean_df.loc[mean_df.Path == path].iloc[0]
        r_range = range_df.loc[range_df.Path == path].iloc[0]
        real = r_mean["REAL__Std_B"]

        ys = np.arange(len(self.techniques))

        plt.figure(figsize=(7, 4))

        for i, tech in enumerate(self.techniques):
            mean = r_mean[f"{self.tech_to_col(tech)}__Std_B"]
            sd = r_range[f"{self.tech_to_col(tech)}__Bootstrap_SD_max"]

            plt.hlines(i, mean - sd, mean + sd)
            plt.plot(mean, i, "o")

        plt.axvline(real, linestyle="--", label="REAL β")
        plt.yticks(ys, self.techniques)
        plt.title(f"Standardized β Interval: {path}")
        plt.legend()
        plt.tight_layout()

        fname = f"interval_{path.replace(' ', '').replace('->', '_')}.png"
        plt.savefig(self.figures_dir / fname)
        plt.close()

    # ==========================================================
    # Bar plot
    # ==========================================================

    def _plot_bar(self, mean_df: pd.DataFrame, path: str):
        r = mean_df.loc[mean_df.Path == path].iloc[0]

        labels = ["REAL"]
        values = [r["REAL__Std_B"]]

        for tech in self.techniques:
            labels.append(tech)
            values.append(r[f"{self.tech_to_col(tech)}__Std_B"])

        plt.figure(figsize=(9, 4))
        plt.bar(labels, values)
        plt.axhline(0, linewidth=0.8)
        plt.title(f"Standardized β Comparison: {path}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        fname = f"bar_{path.replace(' ', '').replace('->', '_')}.png"
        plt.savefig(self.figures_dir / fname)
        plt.close()
