from pathlib import Path
import pandas as pd
import numpy as np
import logging
from sem_report.utils.export_config import ExportConfig
from sem_report.utils.exporter import export_table

from sem_report.loaders.sem_comparison_loader import SEMComparisonLoader
from sem_report.utils.thresholds import HTMT_THRESHOLD

logger = logging.getLogger(__name__)


class DiscriminantValidityEvaluator:
    """
    Computes discriminant validity diagnostics:
      - Fornell–Larcker diagonal (sqrt AVE)
      - Minimum discriminant margin (AVE − max|r|)
      - Max HTMT + pass/fail
      - Fornell–Larcker & HTMT absolute-error grids + RMSE scalars

    No plotting. No interpretation.
    """

    def __init__(
        self,
        loader: SEMComparisonLoader,
        cnt: str,
        techniques: list[str],
        out_dir: Path,
        export_config: ExportConfig
    ):
        self.loader = loader
        self.cnt = cnt
        self.techniques = techniques
        self.export_config = export_config
        self.out_dir = Path(out_dir)

        self.appendix_dir = (
            self.out_dir / "appendix" / "004_discriminant_validity"
        )
        self.tables_dir = (
            self.out_dir / "tables" / "004_discriminant_validity"
        )

        self.appendix_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    # ======================================================
    # PUBLIC ENTRY POINT
    # ======================================================
    def run(self):
        logger.info("Running DiscriminantValidityEvaluator")

        fl_diag = self._fornell_larcker_diagonal()
        margin = self._discriminant_margin()
        htmt = self._htmt_max()
        htmt_stability = self._htmt_stability()

        grid_summary = self._error_grids()

        # ---------- Appendix ----------
        export_table(
            config=self.export_config,
            df = fl_diag,
            path = self.appendix_dir / "fornell_larcker_diagonal.csv",
            index=False
        )
        export_table(
            config=self.export_config,
            df = margin,
            path = self.appendix_dir / "discriminant_margin.csv",
            index=False
        )
        export_table(
            config=self.export_config,
            df = htmt,
            path = self.appendix_dir / "htmt_max_full.csv",
            index=False
        )

        export_table(
            config=self.export_config,
            df=htmt_stability,
            path=self.appendix_dir / "htmt_stability.csv",
            index=False
        )


        # ---------- Report tables ----------
        export_table(
            config=self.export_config,
            df = fl_diag,
            path = self.tables_dir / "table_fornell_larcker_diag.csv",
            index=False
        )
        export_table(
            config=self.export_config,
            df= margin[["Technique", "Min_Margin"]],
            path = self.tables_dir / "table_discriminant_margin.csv",
            index=False
        )
        export_table(
            config=self.export_config,
            df = htmt[["Technique", "Max_HTMT", "HTMT_Status"]],
            path = self.tables_dir / "table_htmt_summary.csv",
            index=False
        )
        export_table(
            config=self.export_config,
            df = grid_summary,
            path = self.tables_dir / "table_fl_htmt_rmse.csv",
            index=False
        )

        export_table(
            config=self.export_config,
            df=htmt_stability,
            path=self.tables_dir / "table_htmt_stability.csv",
            index=False
        )

        htmt = self._htmt_max()

        # CSV (raw)
        htmt.to_csv(
            self.appendix_dir / "htmt_max_full.csv",
            index=False
        )

        # HTML (styled)
        htmt_html = self._style_htmt_for_html(htmt)
        htmt_html.to_html(
            self.tables_dir / "table_htmt_summary.html",
            index=False,
            escape=False,
            classes="sem-table"
        )


        logger.info("Discriminant validity evaluation complete")

    # ======================================================
    # INTERNAL COMPUTATIONS
    # ======================================================
    def _fornell_larcker_diagonal(self) -> pd.DataFrame:
        logger.info("Computing Fornell–Larcker diagonal (√AVE)")

        df = self.loader.fornell_larcker(self.cnt)
        constructs = df["Construct"].astype(str).tolist()

        rows = []
        for tech in self.techniques:
            row = {"Technique": tech}
            for c in constructs:
                col = f"{tech}__{c}"
                row[c] = (
                    float(df.loc[df["Construct"] == c, col].values[0])
                    if col in df.columns else np.nan
                )
            rows.append(row)

        return pd.DataFrame(rows)

    def _discriminant_margin(self) -> pd.DataFrame:
        logger.info("Computing discriminant validity margins")

        df = self.loader.fornell_larcker(self.cnt)
        constructs = df["Construct"].astype(str).tolist()

        rows = []
        for tech in self.techniques:
            margins = []

            for ci in constructs:
                ave_col = f"{tech}__{ci}"
                if ave_col not in df.columns:
                    continue

                ave = df.loc[df["Construct"] == ci, ave_col].values[0]

                corrs = []
                for cj in constructs:
                    if ci == cj:
                        continue
                    col = f"{tech}__{cj}"
                    if col in df.columns:
                        v = df.loc[df["Construct"] == ci, col].values[0]
                        if pd.notna(v):
                            corrs.append(abs(v))

                if corrs:
                    margins.append(ave - max(corrs))

            rows.append({
                "Technique": tech,
                "Min_Margin": round(min(margins),2) if margins else np.nan
            })

        return pd.DataFrame(rows)

    # ======================================================
    # HTMT STABILITY BANDS
    # ======================================================

    def _htmt_stability(self) -> pd.DataFrame:
        """
        Computes HTMT stability as max–min range across construct pairs.
        """
        logger.info("Computing HTMT stability (range width)")

        df = self.loader.htmt(self.cnt)
        constructs = df["Construct"].astype(str).tolist()

        rows = []

        for tech in self.techniques:
            if tech == "REAL":
                continue
            
            vals = []

            for i, ci in enumerate(constructs):
                for j, cj in enumerate(constructs):
                    if i >= j:
                        continue

                    v = None

                    col_1 = f"{tech}__{cj}"
                    if col_1 in df.columns:
                        s = df.loc[df["Construct"] == ci, col_1]
                        if not s.empty and pd.notna(s.values[0]):
                            v = s.values[0]

                    if v is None:
                        col_2 = f"{tech}__{ci}"
                        if col_2 in df.columns:
                            s = df.loc[df["Construct"] == cj, col_2]
                            if not s.empty and pd.notna(s.values[0]):
                                v = s.values[0]

                    if v is not None:
                        vals.append(float(v))

            if vals:
                width = max(vals) - min(vals)
                std = np.std(vals, ddof=1)
            else:
                width = np.nan
                std = np.nan

            rows.append({
                "Technique": tech,
                "HTMT_Range_Width": round(width,2),
                "HTMT_StdDev": round(std,2)
            })

        return pd.DataFrame(rows)


    def _htmt_max(self) -> pd.DataFrame:
        logger.info("Computing max HTMT")

        df = self.loader.htmt(self.cnt)
        constructs = df["Construct"].astype(str).tolist()

        rows = []

        for tech in self.techniques:
            vals = []

            for i, ci in enumerate(constructs):
                for j, cj in enumerate(constructs):
                    if i >= j:
                        continue

                    # Try (ci, cj)
                    v = None
                    col_1 = f"{tech}__{cj}"
                    if col_1 in df.columns:
                        s = df.loc[df["Construct"] == ci, col_1]
                        if not s.empty and pd.notna(s.values[0]):
                            v = s.values[0]

                    # Try (cj, ci) if needed
                    if v is None:
                        col_2 = f"{tech}__{ci}"
                        if col_2 in df.columns:
                            s = df.loc[df["Construct"] == cj, col_2]
                            if not s.empty and pd.notna(s.values[0]):
                                v = s.values[0]

                    if v is not None:
                        vals.append(float(v))

            max_htmt = max(vals) if vals else np.nan

            rows.append({
                "Technique": tech,
                "Max_HTMT": max_htmt,
                "HTMT_Status": (
                    "pass"
                    if pd.notna(max_htmt) and max_htmt < HTMT_THRESHOLD
                    else "fail"
                )
            })

        return pd.DataFrame(rows)

    def _style_htmt_for_html(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply clean numeric + badge formatting for HTMT (no bars).
        """
        out = df.copy()

        # Format number to 2 decimals
        out["Max_HTMT"] = out["Max_HTMT"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else ""
        )

        # Badge column
        def badge(row):
            if row["HTMT_Status"] == "pass":
                return "<span class='htmt-pass'>✅ Pass</span>"
            return "<span class='htmt-fail'>⚠ Borderline</span>"

        out["HTMT_Status"] = out.apply(badge, axis=1)

        return out[["Technique", "Max_HTMT", "HTMT_Status"]]


    # ======================================================
    # ERROR GRIDS (AE + RMSE)
    # ======================================================
    def _error_grids(self) -> pd.DataFrame:
        logger.info("Computing FL / HTMT absolute-error grids")

        fl = self.loader.fornell_larcker(self.cnt)
        htmt = self.loader.htmt(self.cnt)

        summary_rows = []

        for tech in self.techniques:
            if tech == "REAL":
                continue

            fl_rmse = self._grid_error_and_save(
                fl, tech, "FL", drop_diagonal=False
            )
            htmt_rmse = self._grid_error_and_save(
                htmt, tech, "HTMT", drop_diagonal=True
            )

            summary_rows.append({
                "Technique": tech,
                "FL_RMSE": fl_rmse,
                "HTMT_RMSE": htmt_rmse
            })

        return pd.DataFrame(summary_rows)

    def _grid_error_and_save(
        self,
        df: pd.DataFrame,
        tech: str,
        label: str,
        drop_diagonal: bool,
    ) -> float:

        constructs = df["Construct"].astype(str)
        real_cols = [c.replace("REAL__", "") for c in df.columns if c.startswith("REAL__")]
        tech_cols = [c.replace(f"{tech}__", "") for c in df.columns if c.startswith(f"{tech}__")]

        if not real_cols or not tech_cols:
            return np.nan

        real_mat = df[["Construct"] + [f"REAL__{c}" for c in real_cols]].copy()
        tech_mat = df[["Construct"] + [f"{tech}__{c}" for c in tech_cols]].copy()

        real_mat.columns = ["Construct"] + real_cols
        tech_mat.columns = ["Construct"] + tech_cols

        ae, rmse = self._grid_error(
            real_mat,
            tech_mat,
            drop_diagonal=drop_diagonal
        )

        export_table(
            config=self.export_config,
            df = ae,
            path = self.appendix_dir / f"ae_{label.lower()}_{tech}.csv",
            index=False
        )

        return rmse

    @staticmethod
    def _grid_error(
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        construct_col: str = "Construct",
        drop_diagonal: bool = False,
    ):
        real = real_df.set_index(construct_col)
        synth = synth_df.set_index(construct_col)

        common = real.index.intersection(synth.index)
        real = real.loc[common, common].apply(pd.to_numeric, errors="coerce")
        synth = synth.loc[common, common].apply(pd.to_numeric, errors="coerce")

        diff = real - synth
        ae = diff.abs()

        sq = diff.pow(2)
        if drop_diagonal:
            np.fill_diagonal(sq.values, np.nan)

        rmse = np.sqrt(np.nanmean(sq.values))

        ae_out = ae.copy()
        ae_out.insert(0, construct_col, ae_out.index)
        ae_out.reset_index(drop=True, inplace=True)

        return ae_out, rmse
