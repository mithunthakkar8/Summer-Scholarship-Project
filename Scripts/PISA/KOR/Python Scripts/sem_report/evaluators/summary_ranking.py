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


    # =====================================================
    # CORE LOGIC
    # =====================================================

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
                Distribution_Error=lambda d:
                    d["Mean_|Δμ|"]
                    + d["Mean_|Δσ|"]
                    + (1 - d["Mean_Continuous_Adherence"])
                    + (1 - d["Mean_Category_Adherence"])
            )
            [["Technique", "Distribution_Error"]]
        )

        dist["Distribution_Score"] = -dist["Distribution_Error"]

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
        paths["Path_Score"] = (
            paths["Directional Consistency"].astype(float)
            + paths["Rank Preservation (Spearman ρ)"].astype(float)
        )

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
        # 4. VALIDITY / GLOBAL FIT
        # --------------------------------------------------
        htmt = pd.read_csv(
            self.out_dir / "tables/004_discriminant_validity/table_htmt_summary.csv"
        )

        # Extract the HTMT pass/fail row
        htmt = htmt[htmt["Metric"].str.contains("HTMT", regex=False)]

        # Convert ✓ / ✗ to score
        htmt_long = (
            htmt
            .melt(id_vars="Metric", var_name="Technique", value_name="Flag")
            .query("Technique != 'REAL'")
        )

        # Collapse HTMT to ONE row per technique
        htmt_score = (
            htmt_long
            .assign(HTMT_Score=lambda d: (d["Flag"] == "✓").astype(int))
            .groupby("Technique", as_index=False)["HTMT_Score"]
            .max()   # pass if ANY HTMT row passes
        )

        # Combine structural components
        structural = (
            paths[["Technique", "Path_Score"]]
            .merge(latent_mean[["Technique", "LatentCorr_Score"]], on="Technique")
            .merge(cov[["Technique", "Covariate_Score"]], on="Technique")
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
                    (d["CFI (≥0.90)"] >= 0.90).astype(int)
                    + (d["TLI (≥0.90)"] >= 0.90).astype(int)
                    + (d["RMSEA (≤0.08)"] <= 0.08).astype(int)
                    + (d["SRMR (≤0.08)"] <= 0.08).astype(int)
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

        # Build Privacy_Score
        #   Lower is better: ExactMatchRate, MembershipInference
        #   Higher is better: NNDR, DCR_p05
        privacy["Privacy_Score"] = (
            - privacy["Exact Match Rate"]
            - privacy["Membership Inference risk"]
            + privacy["NNDR"]
            + privacy["DCR_p05"]
        )


        # --------------------------------------------------
        # 5. MERGE ALL COMPONENTS
        # --------------------------------------------------
        final = (
            dist[["Technique", "Distribution_Score"]]

            .merge(
                structural[
                    [
                        "Technique",
                        "Path_Score",
                        "LatentCorr_Score",
                        "Covariate_Score",
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
        final["Overall_Score"] = (
            0.13 * _minmax(final["Distribution_Score"])
            + 0.13 * _minmax(final["Path_Score"])
            + 0.08 * _minmax(final["LatentCorr_Score"])
            + 0.08 * _minmax(final["Covariate_Score"])
            + 0.12 * _minmax(final["Loading_Score"])
            + 0.10 * _minmax(final["Reliability_Score"])
            + 0.12 * _minmax(final["HTMT_Score"])
            + 0.09 * _minmax(final["Fit_Score"])
            + 0.15 * _minmax(final["Privacy_Score"])
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

            "Distribution_Score", "Distribution_Rank",
            "Path_Score", "Path_Rank",
            "LatentCorr_Score", "LatentCorr_Rank",
            "Covariate_Score", "Covariate_Rank",
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


