from pathlib import Path
import pandas as pd
import numpy as np
import logging

from sem_report.utils.exporter import export_table
from sem_report.utils.export_config import ExportConfig
from sem_report.loaders.privacy_loader import PrivacyLoader

logger = logging.getLogger(__name__)


def _minmax(s: pd.Series) -> pd.Series:
    if s.max() == s.min():
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


class SummaryRankingEvaluator:
    """
    Builds the final consolidated ranking of synthetic data generators.

    Uses ONLY evaluator outputs already written to disk.
    REAL is excluded from ranking (baseline only).
    """

    def __init__(
        self,
        out_dir: Path,
        techniques: list[str],
        export_config: ExportConfig,
    ):
        self.out_dir = Path(out_dir)
        self.techniques = techniques
        self.export_config = export_config

        self.tables_dir = self.out_dir / "tables" / "000_summary"
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        # =====================================================
        # OVERALL SCORE WEIGHTS (DOCUMENTED)
        # =====================================================
        self.OVERALL_WEIGHTS = {
            "Distribution_Score": 0.13,
            "Path_Score": 0.13,
            "LatentCorr_Score": 0.08,
            "Covariate_Score": 0.08,
            "Loading_Score": 0.12,
            "Reliability_Score": 0.10,
            "HTMT_Score": 0.12,
            "Fit_Score": 0.09,
            "Privacy_Score": 0.15,
            "Indirect_Score": 0.08,
        }


    def _overall_weight_note(self) -> str:
        """
        Generate a readable note describing overall score weight allocation.
        """
        parts = [
            f"{k.replace('_Score','').replace('Corr',' corr').replace('HTMT','HTMT')}: {v:.2f}"
            for k, v in self.OVERALL_WEIGHTS.items()
        ]

        return (
            "Overall composite score is computed as a weighted sum of normalized "
            "component scores with the following weights: "
            + "; ".join(parts)
            + ". All component scores are min–max normalized to [0,1] prior to aggregation."
        )


    # =====================================================
    # PUBLIC ENTRY POINT
    # =====================================================
    def run(self):
        logger.info("Running SummaryRankingEvaluator")

        final = self._build_ranking()

        # ---- Table 00: Composite scores (appendix-style) ----
        export_table(
            df=final,
            path=self.tables_dir / "table_00_final_ranking",
            config=self.export_config,
            table_number=32,
            title="Composite Performance Scores and Overall Ranking of Synthetic Data Techniques",
            note="Scores are min–max normalized within each evaluation dimension and combined using predefined weights; higher scores indicate better overall performance. REAL data are excluded from ranking.",
            index=False,
        )


        # ---- Table 00a: Component-wise ranks (MAIN TABLE) ----
        table_a = self._build_component_ranks(final)

        export_table(
            df=table_a,
            path=self.tables_dir / "table_00a_component_ranks",
            config=self.export_config,
            table_number=33,
            title="Component-wise Rankings Across Evaluation Dimensions",
            note="Ranks are computed independently for each evaluation dimension, with Rank 1 indicating the best-performing technique.",
            index=False,
        )


        logger.info("Summary ranking tables generated")

        table_b = self._build_detailed_metric_table(final)

        export_table(
            df=table_b,
            path=self.tables_dir / "table_00b_detailed_metric_breakdown",
            config=self.export_config,
            table_number=34,
            title="Detailed Metric Breakdown by Technique",
            note=(
                "Rows represent individual evaluation metrics. Errors are shown as positive magnitudes. "
                "Normalized scores are scaled to [0,1], where higher is better. "
                + self._overall_weight_note()
            ),
            index=False,
        )


    # =====================================================
    # CORE LOGIC
    # =====================================================

    def _build_detailed_metric_table(self, final: pd.DataFrame) -> pd.DataFrame:

        techs = final["Technique"].tolist()

        # ----------------------------
        # helpers
        # ----------------------------
        def fmt(v, is_rank=False):
            if pd.isna(v):
                return ""
            if is_rank:
                return str(int(round(v)))
            return f"{float(v):.3f}"

        def best_mask(values, higher_better):
            s = pd.Series(values).astype(float)
            if higher_better:
                return s == s.max()
            return s == s.min()

        def add_row(name, values=None, higher_better=False, is_rank=False):
            row = {"Metric": name}

            if values is None:
                for t in techs:
                    row[t] = ""
                return row

            mask = best_mask(values, higher_better) if not is_rank else best_mask(values, False)

            for t, v, m in zip(techs, values, mask):
                cell = fmt(v, is_rank)

                if m and cell != "":
                    cell = f"<b>{cell}</b>"

                row[t] = cell

            return row


        rows = []

        # =================================================
        # 1. DISTRIBUTION
        # =================================================
        cont = pd.read_csv(
            self.out_dir / "tables/001_distribution/table_distribution_summary_continuous.csv"
        ).query("Technique != 'REAL'")

        cat = pd.read_csv(
            self.out_dir / "tables/001_distribution/table_distribution_summary_categorical.csv"
        ).query("Technique != 'REAL'")

        cont = cont.set_index("Technique").loc[techs]
        cat = cat.set_index("Technique").loc[techs]

        rows.append(add_row("— DISTRIBUTION FIDELITY —"))

        rows.append(add_row("Mean |Δμ| ↓", cont["Mean_|Δμ|"], higher_better=False))
        rows.append(add_row("Mean |Δσ| ↓", cont["Mean_|Δσ|"], higher_better=False))
        rows.append(add_row("Continuous adherence ↑", cont["Mean_Continuous_Adherence"], higher_better=True))
        rows.append(add_row("Category adherence ↑", cat["Mean_Category_Adherence"], higher_better=True))
        rows.append(add_row("Composite error ↓", final["Distribution_Error_Raw"], higher_better=False))

        dist_comp = _minmax(final["Distribution_Score"])
        rows.append(add_row("Distribution composite ↑", dist_comp, higher_better=True))
        rows.append(add_row("Distribution rank", final["Distribution_Rank"], is_rank=True))

        # =================================================
        # 2. STRUCTURAL
        # =================================================
        rows.append(add_row(""))
        rows.append(add_row("— STRUCTURAL FIDELITY —"))

        paths_raw = pd.read_csv(
            self.out_dir / "tables/005_paths/path_direction_rank_summary.csv"
        )

        paths_long = (
            paths_raw
            .melt(id_vars="Metric", var_name="Technique", value_name="Value")
            .query("Technique != 'REAL'")
        )

        paths_wide = (
            paths_long
            .pivot(index="Metric", columns="Technique", values="Value")
            .loc[:, techs]
        )

        rows.append(add_row("Directional consistency ↑", paths_wide.loc["Directional Consistency"], higher_better=True))
        rows.append(add_row("Rank preservation (Spearman ρ) ↑", paths_wide.loc["Rank Preservation (Spearman ρ)"], higher_better=True))

        rows.append(add_row("Latent corr |Δr| ↓", final["LatentCorr_Score"].abs(), higher_better=False))
        rows.append(add_row("Covariate |Δr| ↓", final["Covariate_Score"].abs(), higher_better=False))
        rows.append(add_row("Indirect |Δβ| ↓", final["Indirect_Score"].abs(), higher_better=False))

        struct_comp = (
            _minmax(final["Path_Score"]) +
            _minmax(final["LatentCorr_Score"]) +
            _minmax(final["Covariate_Score"]) +
            _minmax(final["Indirect_Score"])
        ) / 4

        struct_rank = struct_comp.rank(ascending=False)

        rows.append(add_row("Structural composite ↑", struct_comp, higher_better=True))

        rows.append(add_row("Structural rank", struct_rank, is_rank=True))

        # =================================================
        # 3. MEASUREMENT
        # =================================================
        rows.append(add_row(""))
        rows.append(add_row("— MEASUREMENT FIDELITY —"))

        rows.append(add_row("Loading MAD ↓", final["Loading_Score"].abs()))
        rows.append(add_row("Reliability MAD ↓", final["Reliability_Score"].abs()))

        meas_comp = 1 - (
            _minmax(final["Loading_Score"].abs())
            + _minmax(final["Reliability_Score"].abs())
        ) / 2

        rows.append(add_row("Measurement composite ↑", meas_comp, higher_better=True))

        meas_rank = (meas_comp).rank(ascending=False)
        rows.append(add_row("Measurement rank", meas_rank, is_rank=True))

        

        # =================================================
        # 4. VALIDITY
        # =================================================
        rows.append(add_row(""))
        rows.append(add_row("— VALIDITY —"))

        rows.append(add_row("HTMT RMSE ↓", final["HTMT_Score"].abs()))
        fit_raw = pd.read_csv(
            self.out_dir / "tables/007_global_fit_measures/cbsem_global_fit_mean.csv"
        )

        fit_wide = (
            fit_raw.set_index("Metric").T.reset_index().rename(columns={"index": "Technique"})
            .set_index("Technique").loc[techs]
        )

        rows.append(add_row("CFI ↑", fit_wide["CFI (≥0.90)"], higher_better=True))
        rows.append(add_row("TLI ↑", fit_wide["TLI (≥0.90)"], higher_better=True))
        rows.append(add_row("RMSEA ↓", fit_wide["RMSEA (≤0.08)"], higher_better=False))
        rows.append(add_row("SRMR ↓", fit_wide["SRMR (≤0.08)"], higher_better=False))
        


        htmt_norm = _minmax(final["HTMT_Score"])
        fit_norm  = _minmax(final["Fit_Score"])

        val_comp = (htmt_norm + fit_norm) / 2

        rows.append(add_row("Validity composite ↑", val_comp, higher_better=True))

        val_rank = val_comp.rank(ascending=False, method="min")
        rows.append(add_row("Validity rank", val_rank, is_rank=True))


        # =================================================
        # 5. PRIVACY
        # =================================================
        rows.append(add_row(""))
        rows.append(add_row("— PRIVACY —"))

        privacy_loader = PrivacyLoader(self.out_dir)
        privacy_raw = privacy_loader.mean()

        privacy_long = (
            privacy_raw
            .melt(id_vars="Metric", var_name="Technique", value_name="Value")
            .query("Technique != 'REAL'")
        )

        privacy = (
            privacy_long
            .pivot(index="Metric", columns="Technique", values="Value")
            .loc[:, techs]
        )

        rows.append(add_row("Exact Match Rate ↓", privacy.loc["Exact Match Rate"], higher_better=False))
        rows.append(add_row("NNDR ↑", privacy.loc["NNDR"], higher_better=True))
        rows.append(add_row("Membership inference risk ↓", privacy.loc["Membership Inference risk"], higher_better=False))
        rows.append(add_row("DCR p05 ↑", privacy.loc["DCR_p05"], higher_better=True))
        priv_comp = _minmax(final["Privacy_Score"])
        rows.append(add_row("Privacy composite ↑", priv_comp, higher_better=True))
        rows.append(add_row("Privacy rank", final["Privacy_Rank"], is_rank=True))

        


        # =================================================
        # 6. OVERALL
        # =================================================
        rows.append(add_row(""))
        rows.append(add_row("— OVERALL —"))

        rows.append(add_row("Composite score ↑", _minmax(final["Overall_Score"]), higher_better=True))
        rows.append(add_row("Overall rank", final["Overall_Rank"], is_rank=True))

        return pd.DataFrame(rows)


    @staticmethod
    def _tech_to_col(tech: str) -> str:
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

    def _rank_desc(self, s: pd.Series) -> pd.Series:
        """
        Rank where higher is better.
        Rank 1 = best.
        """
        return s.rank(ascending=False, method="min").astype(int)

    def _build_ranking(self) -> pd.DataFrame:

        # --------------------------------------------------
        # 1. DISTRIBUTION FIDELITY
        # --------------------------------------------------
        cont = pd.read_csv(
            self.out_dir / "tables/001_distribution/table_distribution_summary_continuous.csv"
        )
        cat = pd.read_csv(
            self.out_dir / "tables/001_distribution/table_distribution_summary_categorical.csv"
        )

        dist = (
            cont.merge(cat, on="Technique")
            .query("Technique != 'REAL'")
            .assign(
                # raw error terms (kept for transparency / Table 34)
                e_mu=lambda d: pd.to_numeric(d["Mean_|Δμ|"], errors="coerce"),
                e_sig=lambda d: pd.to_numeric(d["Mean_|Δσ|"], errors="coerce"),
                e_cont=lambda d: 1 - pd.to_numeric(d["Mean_Continuous_Adherence"], errors="coerce"),
                e_cat=lambda d: 1 - pd.to_numeric(d["Mean_Category_Adherence"], errors="coerce"),
            )
        )

        # normalize each error term to [0,1] as a "goodness" score (higher better)
        dist["Distribution_Score"] = (
            _minmax(-dist["e_mu"])
            + _minmax(-dist["e_sig"])
            + _minmax(-dist["e_cont"])
            + _minmax(-dist["e_cat"])
        ) / 4

        # optional: keep a raw composite error for reporting only (not used for ranking)
        dist["Distribution_Error_Raw"] = dist["e_mu"] + dist["e_sig"] + dist["e_cont"] + dist["e_cat"]

        dist = dist[["Technique", "Distribution_Score", "Distribution_Error_Raw"]]


        # --------------------------------------------------
        # 2. STRUCTURAL FIDELITY
        # --------------------------------------------------
        # (a) Paths
        paths_raw = pd.read_csv(
            self.out_dir / "tables/005_paths/path_direction_rank_summary.csv"
        )

        # paths_raw format:
        # Metric | CTGAN | GReaT | Tabula | ...

        paths = (
            paths_raw
            .melt(
                id_vars="Metric",
                var_name="Technique",
                value_name="Value"
            )
        )

        # Split metrics into columns
        paths = (
            paths
            .pivot(
                index="Technique",
                columns="Metric",
                values="Value"
            )
            .reset_index()
        )

        # Drop REAL
        paths = paths[paths["Technique"] != "REAL"]

        # Path score
        dirc = pd.to_numeric(paths["Directional Consistency"], errors="coerce")
        rho  = pd.to_numeric(paths["Rank Preservation (Spearman ρ)"], errors="coerce")

        paths["Path_Score"] = (_minmax(dirc) + _minmax(rho)) / 2

        # (b) Latent correlations (PLS + CB)
        latent = pd.read_csv(
            self.out_dir / "tables/002_correlations/table_latent_correlation_stability.csv"
        )

        # Keep only Mean |Δr|
        latent = latent[latent["Metric"] == "Mean |Δr|"]

        # Average PLS + CB per technique
        latent_long = (
            latent
            .melt(
                id_vars=["Model", "Metric"],
                var_name="Technique",
                value_name="MeanAbsDelta_r"
            )
        )

        latent_long = latent_long.query("Technique != 'REAL'")

        latent_mean = (
            latent_long
            .groupby("Technique")["MeanAbsDelta_r"]
            .mean()
            .reset_index(name="LatentCorr_Error")
        )

        latent_mean["LatentCorr_Score"] = -latent_mean["LatentCorr_Error"]


        # (c) Covariate correlations
        cov_long = pd.read_csv(
            self.out_dir / "appendix/003_covariate_correlations/covariate_delta_long.csv"
        )

        cov = (
            cov_long
            .groupby("Technique")["DeltaAbs"]
            .mean()
            .reset_index(name="MeanAbsDelta_r")
            .query("Technique != 'REAL'")
        )

        cov["Covariate_Score"] = -cov["MeanAbsDelta_r"]

        # (d) Indirect effects (mediation fidelity)
        med_long = pd.read_csv(
            self.out_dir / "tables/007_indirect_effects/table_indirect_effects_comparison.csv"
        )

        med = (
            med_long
            .groupby("Technique")["|Δβ|"]     # ← aggregate raw errors
            .mean()
            .reset_index(name="Indirect_Error")
            .query("Technique != 'REAL'")
        )

        med["Indirect_Score"] = -med["Indirect_Error"]



        # --------------------------------------------------
        # 3. MEASUREMENT FIDELITY
        # --------------------------------------------------
        load = pd.read_csv(
            self.out_dir / "tables/006_loadings/mean_absolute_loading_error.csv",
            index_col=0
        ).reset_index().rename(columns={"index": "Technique"})

        load["Loading_Error"] = load.drop(columns="Technique").mean(axis=1)

        # Reliability MAD vs REAL (table is wide: techniques as columns)
        rel_wide = pd.read_csv(
            self.out_dir / "tables/003_reliability/reliability_mad_vs_real.csv"
        )

        # Melt techniques into rows
        rel_long = (
            rel_wide
            .melt(
                id_vars=["Construct", "Metric"],
                var_name="Technique",
                value_name="Delta"
            )
            .dropna(subset=["Delta"])
        )

        rel_err = (
            rel_long
            .groupby("Technique")["Delta"]
            .mean()
            .reset_index(name="Reliability_Error")
            .query("Technique != 'REAL'")
        )


        meas = (
            load.merge(rel_err, on="Technique")
            .query("Technique != 'REAL'")
        )

        meas["Loading_Score"] = -meas["Loading_Error"]
        meas["Reliability_Score"] = -meas["Reliability_Error"]

        # --------------------------------------------------
        # Measurement composite (single dimension)
        # --------------------------------------------------
        meas["Measurement_Score"] = 1 - (
            _minmax(meas["Loading_Error"])
            + _minmax(meas["Reliability_Error"])
        ) / 2

        # --------------------------------------------------
        # 4. VALIDITY / GLOBAL FIT
        # HTMT error-based fidelity (continuous, not pass/fail)
        # --------------------------------------------------

        htmt_rmse = pd.read_csv(
            self.out_dir / "tables/004_discriminant_validity/table_fl_htmt_rmse.csv"
        )

        # Keep only HTMT row
        htmt_rmse = htmt_rmse[htmt_rmse["Metric"] == "RMSE (HTMT)"]

        htmt_long = (
            htmt_rmse
            .melt(id_vars="Metric", var_name="Technique", value_name="HTMT_Error")
            .query("Technique != 'REAL'")
        )

        htmt_score = (
            htmt_long
            .assign(HTMT_Score=lambda d: -pd.to_numeric(d["HTMT_Error"], errors="coerce"))
            [["Technique", "HTMT_Score"]]
        )

        # --------------------------------------------------
        # Global fit (CB-SEM) — input is metrics-as-rows, techniques-as-columns
        # --------------------------------------------------
        fit_raw = pd.read_csv(
            self.out_dir / "tables/007_global_fit_measures/cbsem_global_fit_mean.csv"
        )

        # Expect: first column is "Metric"
        if "Metric" not in fit_raw.columns:
            # safety: if exporter wrote unnamed first column
            fit_raw = fit_raw.rename(columns={fit_raw.columns[0]: "Metric"})

        # Keep only the four thresholded metrics (AIC is not part of Fit_Score)
        needed_metrics = ["CFI (≥0.90)", "TLI (≥0.90)", "RMSEA (≤0.08)", "SRMR (≤0.08)"]
        fit_sub = fit_raw[fit_raw["Metric"].isin(needed_metrics)].copy()

        # Convert to: rows=Technique, cols=Metric
        fit_wide = fit_sub.set_index("Metric").T.reset_index().rename(columns={"index": "Technique"})

        # Ensure numeric
        for m in needed_metrics:
            fit_wide[m] = pd.to_numeric(fit_wide.get(m, np.nan), errors="coerce")

        fit_score = (
            fit_wide
            .assign(
                Fit_Score=lambda d:
                    (
                        _minmax(d["CFI (≥0.90)"])
                        + _minmax(d["TLI (≥0.90)"])
                        + _minmax(-d["RMSEA (≤0.08)"])
                        + _minmax(-d["SRMR (≤0.08)"])
                    ) / 4
            )
            [["Technique", "Fit_Score"]]
            .query("Technique != 'REAL'")
        )


        # --------------------------------------------------
        # Combine validity components (HTMT + Global fit)
        # --------------------------------------------------
        validity = (
            htmt_score
            .merge(fit_score, on="Technique", how="inner")
        )

        privacy_loader = PrivacyLoader(self.out_dir)
        privacy_raw = privacy_loader.mean()


        # privacy_raw format:
        # Metric | CTGAN | GReaT | Tabula | ...

        privacy_long = (
            privacy_raw
            .melt(
                id_vars="Metric",
                var_name="Technique",
                value_name="Value"
            )
            .query("Technique != 'REAL'")
        )

        # Convert to wide: one row per Technique
        privacy = (
            privacy_long
            .pivot(
                index="Technique",
                columns="Metric",
                values="Value"
            )
            .reset_index()
        )

        # Ensure numeric
        for col in ["ExactMatchRate", "NNDR", "MembershipInference", "DCR_p05"]:
            if col in privacy.columns:
                privacy[col] = pd.to_numeric(privacy[col], errors="coerce")

        # ---------------------------
        # Balanced privacy composite
        # normalize each metric first
        # ---------------------------

        emr = _minmax(-privacy["Exact Match Rate"])
        mi  = _minmax(-privacy["Membership Inference risk"])
        nndr = _minmax(privacy["NNDR"])
        dcr  = _minmax(privacy["DCR_p05"])

        privacy["Privacy_Score"] = (emr + mi + nndr + dcr) / 4



        # --------------------------------------------------
        # 3.5 STRUCTURAL COMPONENT COMBINATION
        # --------------------------------------------------
        structural = (
            paths[["Technique", "Path_Score"]]
            .merge(latent_mean[["Technique", "LatentCorr_Score"]], on="Technique")
            .merge(cov[["Technique", "Covariate_Score"]], on="Technique")
            .merge(med[["Technique", "Indirect_Score"]], on="Technique")
        )


        # --------------------------------------------------
        # 5. MERGE ALL COMPONENTS
        # --------------------------------------------------
        final = (
            dist[["Technique", "Distribution_Score", "Distribution_Error_Raw"]]

            .merge(
                structural[
                    [
                        "Technique",
                        "Path_Score",
                        "LatentCorr_Score",
                        "Covariate_Score",
                        "Indirect_Score", 
                    ]
                ],
                on="Technique",
                how="inner",
            )

            .merge(
                meas[
                    [
                        "Technique",
                        "Loading_Score",
                        "Reliability_Score",
                        "Measurement_Score",
                    ]
                ],
                on="Technique",
                how="inner",
            )

            .merge(
                validity[
                    [
                        "Technique",
                        "HTMT_Score",
                        "Fit_Score",
                    ]
                ],
                on="Technique",
                how="inner",
            )

            .merge(
                privacy[
                    [
                        "Technique",
                        "Privacy_Score",
                    ]
                ],
                on="Technique",
                how="inner",
            )


        )


        # --------------------------------------------------
        # 6. OVERALL SCORE + RANK
        # --------------------------------------------------
        final["Overall_Score"] = sum(
            w * _minmax(final[k]) if k not in ["Distribution_Score", "Path_Score", "Fit_Score", "Privacy_Score"]
            else w * final[k]
            for k, w in self.OVERALL_WEIGHTS.items()
        )


        RANK_COLS = [
            "Distribution_Score",
            "Path_Score",
            "LatentCorr_Score",
            "Covariate_Score",
            "Loading_Score",
            "Reliability_Score",
            "HTMT_Score",
            "Fit_Score",
            "Privacy_Score",
            "Indirect_Score"
        ]

        for col in RANK_COLS:
            final[col.replace("_Score", "_Rank")] = self._rank_desc(final[col])


        final = (
            final.sort_values("Overall_Score", ascending=False)
            .reset_index(drop=True)
        )
        final["Overall_Rank"] = final.index + 1

        return final[
        [
            "Overall_Rank",
            "Technique",

            "Distribution_Error_Raw",
            "Distribution_Score", "Distribution_Rank",
            "Path_Score", "Path_Rank",
            "LatentCorr_Score", "LatentCorr_Rank",
            "Covariate_Score", "Covariate_Rank",
            "Indirect_Score", "Indirect_Rank",
            "Loading_Score", "Loading_Rank",
            "Reliability_Score", "Reliability_Rank",
            "HTMT_Score", "HTMT_Rank",
            "Fit_Score", "Fit_Rank",
            "Privacy_Score", "Privacy_Rank",

            "Overall_Score",
        ]
    ]



    # =====================================================
    # TABLE A: COMPONENT-WISE RANKS (INTERPRETABLE)
    # =====================================================
    def _build_component_ranks(self, final: pd.DataFrame) -> pd.DataFrame:
        """
        Build interpretable atomic component-wise ranking table (Table 00a).
        """

        cols = [
            "Technique",

            "Distribution_Rank",
            "Path_Rank",
            "LatentCorr_Rank",
            "Covariate_Rank",
            "Loading_Rank",
            "Reliability_Rank",
            "HTMT_Rank",
            "Fit_Rank",
            "Privacy_Rank",

            "Overall_Rank",
        ]


        return (
            final[cols]
            .sort_values("Overall_Rank")
            .reset_index(drop=True)
        )


