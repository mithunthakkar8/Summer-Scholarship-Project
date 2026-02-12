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

    Notes:
    - Only multi-indicator latent constructs are evaluated.
    - Single-indicator controls are excluded by design.

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
            df=fl_diag,
            path=self.appendix_dir / "fornell_larcker_diagonal",
            config=self.export_config,
            table_number=9,
            title="Fornell–Larcker Diagonal Values by Technique",
            note="Diagonal entries report the square root of AVE for each latent construct.",
            index=False,
        )

        export_table(
            df=margin,
            path=self.appendix_dir / "discriminant_margin",
            config=self.export_config,
            table_number=10,
            title="Minimum Discriminant Margin Across Constructs",
            note="Margins are computed as the minimum of √AVE minus the maximum absolute inter-construct correlation.",
            index=False,
        )

        export_table(
            df=htmt,
            path=self.appendix_dir / "htmt_max_full",
            config=self.export_config,
            table_number=11,
            title="Maximum HTMT Values and Threshold Compliance",
            note=f"Check marks indicate whether maximum HTMT is below the threshold of {HTMT_THRESHOLD}.",
            index=False,
        )


        export_table(
            df=htmt_stability,
            path=self.appendix_dir / "htmt_stability",
            config=self.export_config,
            table_number=12,
            title="HTMT Stability Across Construct Pairs",
            note="Rows report the HTMT range and standard deviation across all construct pairs for each technique.",
            index=False,
        )


        # ---------- Report tables ----------
        # CSV / BOTH
        export_table(
            df=fl_diag,
            path=self.tables_dir / "table_fornell_larcker_diag",
            config=self.export_config,
            table_number=13,
            title="Fornell–Larcker Discriminant Validity Criterion",
            note="Diagonal values correspond to √AVE; off-diagonal correlations are omitted.",
            index=False,
        )


        


        # HTML-only: bold Metric column
        if self.export_config.export_html:
            fl_diag_html = fl_diag.copy()
            fl_diag_html.index.name = None
            fl_diag_html.columns.name = None

            from sem_report.utils.export_config import ExportFormat

            export_table(
                df=fl_diag,
                path=self.tables_dir / "table_fornell_larcker_diag",
                config=self.export_config,
                table_number=13,
                title="Fornell–Larcker Discriminant Validity Criterion",
                note="Diagonal values correspond to √AVE; off-diagonal correlations are omitted.",
                index=False,
            )

        # CSV / BOTH
        export_table(
            df=margin,
            path=self.tables_dir / "table_discriminant_margin",
            config=self.export_config,
            table_number=14,
            title="Minimum Discriminant Validity Margin by Technique",
            note="Higher values indicate stronger separation between latent constructs.",
            index=False,
        )


        # HTML-only: bold Metric column
        if self.export_config.export_html:
            margin_html = margin.copy()
            margin_html.index.name = None
            margin_html.columns.name = None

            from sem_report.utils.export_config import ExportFormat

            export_table(
                df=margin,
                path=self.tables_dir / "table_discriminant_margin",
                config=self.export_config,
                table_number=14,
                title="Minimum Discriminant Validity Margin by Technique",
                note="Higher values indicate stronger separation between latent constructs.",
                index=False,
            )


        export_table(
            df=htmt,
            path=self.tables_dir / "table_htmt_summary",
            config=self.export_config,
            table_number=15,
            title="HTMT Discriminant Validity Summary",
            note=f"Threshold for acceptable discriminant validity is HTMT < {HTMT_THRESHOLD}.",
            index=False,
        )


        # CSV / BOTH
        export_table(
            df=grid_summary,
            path=self.tables_dir / "table_fl_htmt_rmse",
            config=self.export_config,
            table_number=16,
            title="RMSE of Fornell–Larcker and HTMT Error Grids",
            note="RMSE values summarize absolute error between synthetic and real discriminant validity matrices.",
            index=False,
        )


        # HTML-only: bold Metric column
        if self.export_config.export_html:
            grid_html = grid_summary.copy()
            grid_html.index.name = None
            grid_html.columns.name = None

            from sem_report.utils.export_config import ExportFormat

            export_table(
                df=grid_html,
                path=self.tables_dir / "table_fl_htmt_rmse",
                config=ExportConfig(fmt=ExportFormat.HTML),
                table_number=16,
                title="RMSE of Fornell–Larcker and HTMT Error Grids",
                note="RMSE values summarize absolute error between synthetic and real discriminant validity matrices.",
                index=False,
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




        logger.info("Discriminant validity evaluation complete")

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
    # INTERNAL COMPUTATIONS
    # ======================================================

    def _filter_latent_constructs(self, constructs: list[str]) -> list[str]:
        """
        Remove single-indicator or control constructs that should not
        be part of discriminant validity diagnostics.
        """
        EXCLUDE = {"NSC"}   # TIMSS single-indicator construct
        return [c for c in constructs if c not in EXCLUDE]


    def _fornell_larcker_diagonal(self) -> pd.DataFrame:
        df = self.loader.fornell_larcker(self.cnt)
        constructs = self._filter_latent_constructs(
            df["Construct"].astype(str).tolist()
        )

        records = []

        for c in constructs:
            row = {"Metric": f"√AVE ({c})"}

            for tech in self.techniques:
                col = f"{self.tech_to_col(tech)}__{c}"
                row[tech] = (
                    round(float(df.loc[df["Construct"] == c, col].values[0]), 3)
                    if col in df.columns else np.nan
                )

            records.append(row)

        out = pd.DataFrame(records)

        tech_cols = sorted(c for c in out.columns if c != "Metric")
        return out[["Metric"] + tech_cols]


    def _discriminant_margin(self) -> pd.DataFrame:
        df = self.loader.fornell_larcker(self.cnt)
        constructs = df["Construct"].astype(str).tolist()

        row = {"Metric": "Minimum discriminant margin"}

        for tech in self.techniques:
            margins = []

            for ci in constructs:
                ave_col = f"{self.tech_to_col(tech)}__{ci}"
                if ave_col not in df.columns:
                    continue

                ave = df.loc[df["Construct"] == ci, ave_col].values[0]
                corrs = []

                for cj in constructs:
                    if ci == cj:
                        continue
                    col = f"{self.tech_to_col(tech)}__{cj}"
                    if col in df.columns:
                        v = df.loc[df["Construct"] == ci, col].values[0]
                        if pd.notna(v):
                            corrs.append(abs(v))

                if corrs:
                    margins.append(ave - max(corrs))

            row[tech] = round(min(margins), 3) if margins else np.nan

        return pd.DataFrame([row])




    # ======================================================
    # HTMT STABILITY BANDS
    # ======================================================

    def _htmt_stability(self) -> pd.DataFrame:
        df = self.loader.htmt(self.cnt)
        constructs = self._filter_latent_constructs(
            df["Construct"].astype(str).tolist()
        )

        row_rng = {"Metric": r"$\max(\mathrm{HTMT}) - \min(\mathrm{HTMT})$"}
        row_sd  = {"Metric": r"$\sigma(\mathrm{HTMT})$"}

        for tech in self.techniques:
            if tech == "REAL":
                continue

            vals = []

            for i, ci in enumerate(constructs):
                for j, cj in enumerate(constructs):
                    if i >= j:
                        continue

                    for col in (f"{tech}__{cj}", f"{tech}__{ci}"):
                        if col in df.columns:
                            s = df.loc[df["Construct"] == ci, col]
                            if not s.empty and pd.notna(s.values[0]):
                                vals.append(float(s.values[0]))
                                break

            if vals:
                row_rng[tech] = round(max(vals) - min(vals), 3)
                row_sd[tech]  = round(np.std(vals, ddof=1), 3)
            else:
                row_rng[tech] = np.nan
                row_sd[tech]  = np.nan

        return pd.DataFrame([row_rng, row_sd])



    def _htmt_max(self) -> pd.DataFrame:
        df = self.loader.htmt(self.cnt)

        row_val = {"Metric": "Max HTMT"}
        row_ok  = {"Metric": f"HTMT < {HTMT_THRESHOLD}"}

        techs = ["REAL"] + [t for t in self.techniques if t != "REAL"]
        for tech in techs:

            prefix = f"{self.tech_to_col(tech)}__"
            tech_cols = [c for c in df.columns if c.startswith(prefix)]

            if not tech_cols:
                row_val[tech] = np.nan
                row_ok[tech]  = "✗"
                continue

            vals = (
                df[tech_cols]
                .stack()
                .dropna()
                .astype(float)
            )

            m = vals.max() if len(vals) else np.nan

            row_val[tech] = round(m, 3) if pd.notna(m) else np.nan
            row_ok[tech]  = "✓" if pd.notna(m) and m < HTMT_THRESHOLD else "✗"

        return pd.DataFrame([row_val, row_ok])




    # ======================================================
    # ERROR GRIDS (AE + RMSE)
    # ======================================================
    def _error_grids(self) -> pd.DataFrame:
        fl = self.loader.fornell_larcker(self.cnt)
        htmt = self.loader.htmt(self.cnt)

        row_fl   = {"Metric": "RMSE (Fornell–Larcker)"}
        row_htmt = {"Metric": "RMSE (HTMT)"}

        for tech in self.techniques:
            if tech == "REAL":
                continue

            fl_rmse = self._grid_error_and_save(fl, tech, "FL", drop_diagonal=False)
            htmt_rmse = self._grid_error_and_save(htmt, tech, "HTMT", drop_diagonal=True)

            row_fl[tech]   = round(fl_rmse, 4)
            row_htmt[tech] = round(htmt_rmse, 4)

        return pd.DataFrame([row_fl, row_htmt])


    def _grid_error_and_save(
        self,
        df: pd.DataFrame,
        tech: str,
        label: str,
        drop_diagonal: bool,
    ) -> float:

        constructs = df["Construct"].astype(str)
        real_cols = [c.replace("REAL__", "") for c in df.columns if c.startswith("REAL__")]
        tech_cols = [c.replace(f"{self.tech_to_col(tech)}__", "") for c in df.columns if c.startswith(f"{self.tech_to_col(tech)}__")]

        if not real_cols or not tech_cols:
            return np.nan

        real_mat = df[["Construct"] + [f"REAL__{c}" for c in real_cols]].copy()
        tech_mat = df[["Construct"] + [f"{self.tech_to_col(tech)}__{c}" for c in tech_cols]].copy()

        real_mat.columns = ["Construct"] + real_cols
        tech_mat.columns = ["Construct"] + tech_cols

        ae, rmse = self._grid_error(
            real_mat,
            tech_mat,
            drop_diagonal=drop_diagonal
        )

        export_table(
            df=ae,
            path=self.appendix_dir / f"ae_{label.lower()}_{self.tech_to_col(tech)}",
            config=self.export_config,
            table_number=None,  # appendix micro-table, no numbering
            title=None,
            note=None,
            index=False,
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
