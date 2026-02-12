from __future__ import annotations

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sem_report.utils.export_config import ExportConfig
from sem_report.utils.exporter import export_table

from sem_report.loaders.sem_comparison_loader import SEMComparisonLoader

logger = logging.getLogger(__name__)


class CorrelationEvaluator:
    """
    Correlation preservation evaluator.

    Outputs:
      - Latent correlation Δr (signed) table + heatmap (PLS and CB-SEM)
      - Latent correlation stability summary per technique (mean |Δr|)
      - Covariate correlation error summary (mean/median/max |Δr|)
      - Covariate Δr heatmap (mean |Δr| across SMP/SMS/SPI)

    Notes:
      - No ranges used (per your instruction).
      - Expects REAL and techniques columns in merged workbook: REAL__*, TECH__*
    """

    # Default latent pairs required by your report
    DEFAULT_LATENT_PAIRS = [("SMP", "SMS"), ("SMP", "SPI"), ("SMS", "SPI")]

    def __init__(
        self,
        loader: SEMComparisonLoader,
        cnt: str,
        techniques: list[str],
        out_dir: Path,
        export_config: ExportConfig,
        latent_pairs: list[tuple[str, str]] | None = None
    ):
        self.loader = loader
        self.cnt = cnt
        self.techniques = techniques
        self.out_dir = Path(out_dir)
        self.export_config = export_config

        self.latent_pairs = latent_pairs or self.DEFAULT_LATENT_PAIRS

        # Output dirs aligned with your numbered structure
        self.appendix_dir = self.out_dir / "appendix" / "002_correlations"
        self.tables_dir = self.out_dir / "tables" / "002_correlations"
        self.figures_dir = self.out_dir / "figures" / "002_correlations"

        for d in [self.appendix_dir, self.tables_dir, self.figures_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ======================================================
    # PUBLIC ENTRY POINT
    # ======================================================
    def run(self):
        logger.info("Running CorrelationEvaluator")

        # ---------- Latent correlations ----------
        pls_latent = self._latent_delta_table(kind="pls")
        cb_latent = self._latent_delta_table(kind="cb")

        pls_stability = self._latent_stability_summary(pls_latent, prefix="PLS")
        cb_stability = self._latent_stability_summary(cb_latent, prefix="CB")

        # Save latent tables
        export_table(
            pls_latent,
            self.appendix_dir / "latent_delta_pls.csv",
            config=self.export_config,
            index=False,
        )

        export_table(
            cb_latent,
            self.appendix_dir / "latent_delta_cb.csv",
            config=self.export_config,
            index=False,
        )

        export_table(pd.concat([pls_stability, cb_stability], ignore_index=True),
            self.tables_dir / "table_latent_correlation_stability.csv",
            config=self.export_config,
            index=False
        )

        # Heatmaps (signed Δr)
        self._heatmap_matrix(
            df=pls_latent,
            title=f"PLS Latent Correlations: Signed Δr vs REAL ({self.cnt})",
            out_path=self.figures_dir / "heatmap_latent_delta_pls.png",
        )
        self._heatmap_matrix(
            df=cb_latent,
            title=f"CB-SEM Latent Correlations: Signed Δr vs REAL ({self.cnt})",
            out_path=self.figures_dir / "heatmap_latent_delta_cb.png",
        )

        logger.info("CorrelationEvaluator complete")

    # ======================================================
    # LATENT CORRELATIONS (PLS + CB)
    # ======================================================
    def _latent_delta_table(self, kind: str) -> pd.DataFrame:
        """
        Returns a wide table:
            Technique, SMP–SMS, SMP–SPI, SMS–SPI
        where each entry is signed Δr = r_synth - r_real.
        """
        if kind not in {"pls", "cb"}:
            raise ValueError("kind must be 'pls' or 'cb'")

        logger.info(f"Computing latent Δr table for kind={kind}")

        if kind == "pls":
            df = self.loader.latent_correlations_pls(self.cnt)
        else:
            df = self.loader.latent_correlations_cb(self.cnt)

        # Build REAL correlation matrix for quick lookup
        real_mat = self._extract_square_matrix(df, tech="REAL")

        rows = []
        for tech in self.techniques:
            if tech == "REAL":
                continue

            tech_mat = self._extract_square_matrix(df, tech=tech)

            row = {"Technique": tech}
            for a, b in self.latent_pairs:
                r_real = self._safe_get(real_mat, a, b)
                r_tech = self._safe_get(tech_mat, a, b)
                row[f"{a}–{b}"] = (r_tech - r_real) if pd.notna(r_real) and pd.notna(r_tech) else np.nan

            rows.append(row)

        out = pd.DataFrame(rows)
        # deterministic column order
        col_order = ["Technique"] + [f"{a}–{b}" for a, b in self.latent_pairs]
        out = out[col_order]
        return out

    def _latent_stability_summary(self, latent_delta_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """
        Summarize mean |Δr| per technique across the specified latent pairs.
        """
        cols = [c for c in latent_delta_df.columns if c != "Technique"]
        tmp = latent_delta_df.copy()

        tmp["MeanAbsDelta_r"] = tmp[cols].abs().mean(axis=1, skipna=True)
        tmp["MedianAbsDelta_r"] = tmp[cols].abs().median(axis=1, skipna=True)
        tmp["MaxAbsDelta_r"] = tmp[cols].abs().max(axis=1, skipna=True)
        tmp["Model"] = prefix

        return tmp[["Model", "Technique", "MeanAbsDelta_r", "MedianAbsDelta_r", "MaxAbsDelta_r"]]

    # ======================================================
    # MATRIX EXTRACTION HELPERS
    # ======================================================
    def _extract_square_matrix(self, df: pd.DataFrame, tech: str) -> pd.DataFrame:
        """
        Convert the wide comparison sheet into a square matrix for one technique:
          index = row construct names from df["Construct"]
          columns = construct names (stripped from TECH__ prefix)
        """
        if "Construct" not in df.columns and "Latent" not in df.columns:
            raise ValueError("Expected 'Construct' column in correlation matrix sheet.")

        if "Construct" in df.columns:
            constructs = df["Construct"].astype(str).tolist()
        else:
            constructs = df["Latent"].astype(str).tolist()
        tech_cols = [c for c in df.columns if c.startswith(f"{tech}__")]

        if not tech_cols:
            logger.debug(f"No columns found for tech={tech} in correlation matrix.")
            # return empty matrix with correct shape
            return pd.DataFrame(index=constructs, columns=constructs, dtype=float)

        mat = df[tech_cols].copy()
        mat.columns = [c.split("__", 1)[1] for c in tech_cols]
        mat.index = constructs

        # ensure numeric
        mat = mat.apply(pd.to_numeric, errors="coerce")

        return mat

    @staticmethod
    def _safe_get(mat: pd.DataFrame, a: str, b: str) -> float:
        """
        Symmetry-aware lookup: tries (a,b) then (b,a).
        """
        if a in mat.index and b in mat.columns:
            return mat.loc[a, b]
        if b in mat.index and a in mat.columns:
            return mat.loc[b, a]
        return np.nan

    # ======================================================
    # PLOTTING HELPERS (self-contained)
    # ======================================================
    def _heatmap_matrix(self, df: pd.DataFrame, title: str, out_path: Path):
        """
        Heatmap where rows=Technique, cols=paths, values=signed Δr.
        """
        if df.empty:
            logger.warning(f"Heatmap skipped (empty df): {out_path.name}")
            return

        data = df.set_index("Technique")
        arr = data.values.astype(float)

        fig, ax = plt.subplots(figsize=(max(6, 0.9 * data.shape[1]), max(3, 0.5 * data.shape[0] + 1)))
        im = ax.imshow(arr, aspect="auto")
        # ---- annotate cells with values ----
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if not np.isnan(arr[i, j]):
                    ax.text(
                        j, i,
                        f"{arr[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="black"
                    )

        ax.set_title(title)
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(list(data.columns), rotation=45, ha="right")
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels(list(data.index))

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Signed Δr (synth − real)")

        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        # also write PDF alongside PNG
        pdf_path = out_path.with_suffix(".pdf")
        fig2, ax2 = plt.subplots(figsize=(max(6, 0.9 * data.shape[1]), max(3, 0.5 * data.shape[0] + 1)))
        im2 = ax2.imshow(arr, aspect="auto")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if not np.isnan(arr[i, j]):
                    ax2.text(
                        j, i,
                        f"{arr[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="black"
                    )
        ax2.set_title(title)
        ax2.set_xticks(range(data.shape[1]))
        ax2.set_xticklabels(list(data.columns), rotation=45, ha="right")
        ax2.set_yticks(range(data.shape[0]))
        ax2.set_yticklabels(list(data.index))
        cbar2 = fig2.colorbar(im2, ax=ax2)
        cbar2.set_label("Signed Δr (synth − real)")
        fig2.tight_layout()
        fig2.savefig(pdf_path)
        plt.close(fig2)

        logger.info(f"Saved heatmap: {out_path.name}")

    