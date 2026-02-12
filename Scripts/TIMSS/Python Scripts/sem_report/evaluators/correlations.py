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
      - Covariate Δr heatmap (mean |Δr| across all constructs)

    Notes:
      - No ranges used (per your instruction).
      - Expects REAL and techniques columns in merged workbook: REAL__*, TECH__*
    """

    # Default latent pairs required by your report
    DEFAULT_LATENT_PAIRS = [
        ("ACM", "SSF"),
        ("ACM", "TCI"),
        ("ACM", "LEI"),
        ("SSF", "TCI"),
        ("SSF", "LEI"),
        ("TCI", "LEI"),
    ]


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
            self.appendix_dir / "latent_delta_pls",
            config=self.export_config,
            table_number=1,
            title="PLS Latent Correlation Differences Relative to Real Data",
            note="Entries are signed Δr values computed as synthetic minus real correlations.",
            index=False,
        )
        export_table(
            cb_latent,
            self.appendix_dir / "latent_delta_cb",
            config=self.export_config,
            table_number=2,
            title="CB-SEM Latent Correlation Differences Relative to Real Data",
            note="Entries are signed Δr values computed as synthetic minus real correlations.",
            index=False,
        )


        latent_stability = (
            pd.concat([pls_stability, cb_stability], ignore_index=True)
        )

        export_table(
            latent_stability,
            self.tables_dir / "table_latent_correlation_stability",
            config=self.export_config,
            table_number=3,
            title="Latent Correlation Stability Across Synthetic Data Techniques",
            note="Values represent mean, median, and maximum absolute Δr across latent construct pairs.",
            index=False,
        )


        # Heatmaps (signed Δr)
        self._heatmap_matrix_paths_rows(
            df=pls_latent,
            title=f"PLS Latent Correlations: Signed Δr vs REAL ({self.cnt})",
            out_path=self.figures_dir / "heatmap_latent_delta_pls.png",
        )

        self._heatmap_matrix_paths_rows(
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
            Path | TECH_1 | TECH_2 | ...
        where each entry is signed Δr = r_synth - r_real.
        """
        if kind not in {"pls", "cb"}:
            raise ValueError("kind must be 'pls' or 'cb'")

        logger.info(f"Computing latent Δr table for kind={kind}")

        if kind == "pls":
            df = self.loader.latent_correlations_pls(self.cnt)
        else:
            df = self.loader.latent_correlations_cb(self.cnt)

        real_mat = self._extract_square_matrix(df, tech="REAL")

        rows = []
        for a, b in self.latent_pairs:
            row = {"Path": f"{a}–{b}"}

            if a not in real_mat.index or b not in real_mat.columns:
                logger.warning(
                    f"Latent pair ({a}, {b}) not found in REAL correlation matrix — skipping"
                )
                continue

            r_real = self._safe_get(real_mat, a, b)

            for tech in self.techniques:
                if tech == "REAL":
                    continue

                tech_mat = self._extract_square_matrix(df, tech=tech)
                r_tech = self._safe_get(tech_mat, a, b)

                row[tech] = (
                    r_tech - r_real
                    if pd.notna(r_real) and pd.notna(r_tech)
                    else np.nan
                )

            rows.append(row)

        out = pd.DataFrame(rows)

        # deterministic column order
        col_order = ["Path"] + [t for t in self.techniques if t != "REAL"]
        out = out[col_order]

        return out


    def _latent_stability_summary(
        self,
        latent_delta_df: pd.DataFrame,
        prefix: str
    ) -> pd.DataFrame:
        """
        Returns a table with:
        rows    = metrics (Mean |Δr|, Median |Δr|, Max |Δr|)
        columns = techniques
        """

        tech_cols = [c for c in latent_delta_df.columns if c != "Path"]

        stats = {
            "Mean |Δr|": [],
            "Median |Δr|": [],
            "Max |Δr|": [],
        }

        for tech in tech_cols:
            vals = pd.to_numeric(latent_delta_df[tech], errors="coerce").abs()
            stats["Mean |Δr|"].append(vals.mean(skipna=True))
            stats["Median |Δr|"].append(vals.median(skipna=True))
            stats["Max |Δr|"].append(vals.max(skipna=True))

        out = pd.DataFrame(
            stats,
            index=tech_cols
        ).T.reset_index().rename(columns={"index": "Metric"})

        # Optional: add model label for downstream joins
        out.insert(0, "Model", prefix)

        return out

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
        tech_cols = [c for c in df.columns if c.startswith(f"{self.tech_to_col(tech)}__")]

        if not tech_cols:
            logger.debug(f"No columns found for tech={self.tech_to_col(tech)} in correlation matrix.")
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
    def _heatmap_matrix_paths_rows(self, df: pd.DataFrame, title: str, out_path: Path):
        """
        Heatmap with:
        - rows = Paths
        - columns = Techniques
        - values = signed Δr
        """

        if df.empty:
            logger.warning(f"Heatmap skipped (empty df): {out_path.name}")
            return

        data = df.set_index("Path")

        arr = data.values.astype(float)

        fig, ax = plt.subplots(
            figsize=(max(6, 0.9 * arr.shape[1]), max(4, 0.6 * arr.shape[0]))
        )

        im = ax.imshow(arr, aspect="auto")

        # annotate cells
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if not np.isnan(arr[i, j]):
                    ax.text(
                        j, i,
                        f"{arr[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9
                    )

        ax.set_title(title)

        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(data.columns, rotation=45, ha="right")

        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels(data.index)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Signed Δr (synthetic − real)")

        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        logger.info(f"Saved heatmap: {out_path.name}")


    