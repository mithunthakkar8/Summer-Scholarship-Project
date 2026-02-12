from pathlib import Path
import logging
import glob
import pandas as pd
import numpy as np

import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sem_report.utils.exporter import export_table
from sem_report.utils.export_config import ExportConfig

from sdmetrics.single_column import BoundaryAdherence, CategoryAdherence



logger = logging.getLogger(__name__)


class DistributionEvaluator:
    """
    Compares marginal distributions between REAL and synthetic datasets.

    Aggregates statistics across multiple synthetic CSVs per technique
    and compares them against REAL.
    """
    def __init__(
        self,
        technique_paths: dict[str, str | Path],
        real_path: str | Path,
        cnt: str,
        out_dir: Path,
        export_config: ExportConfig,
        categorical_cols: list[str],
        ordinal_cols: list[str],
    ):
        self.technique_paths = {
            k: Path(v) for k, v in technique_paths.items()
        }
        self.real_path = real_path
        self.cnt = cnt
        self.out_dir = Path(out_dir)
        self.export_config = export_config

        self.categorical_cols = set(categorical_cols)
        self.ordinal_cols = set(ordinal_cols)
        self.numeric_cols = None  # auto-detected

        self.appendix_dir = self.out_dir / "appendix" / "001_distribution"
        self.tables_dir   = self.out_dir / "tables" / "001_distribution"
        self.figures_dir  = self.out_dir / "figures" / "001_distribution"

        self.appendix_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.plots_dir = self.appendix_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_csv = self.appendix_dir / "marginal_distances.csv"

        

        logger.info("Initialized DistributionEvaluator")
        logger.info(f"Country: {self.cnt}")
        logger.info(f"Techniques: {list(self.technique_paths.keys())}")
        logger.debug(f"Appendix dir: {self.appendix_dir}")
        logger.debug(f"Tables dir:   {self.tables_dir}")

    # ======================================================
    # PUBLIC ENTRY POINT
    # ======================================================
    def run(self):
        logger.info("===== Running DistributionEvaluator =====")

        rows = []
        n_rows = []
        metrics_rows = []
        boundary_rows = []
        category_rows = []

        # --- Load REAL once ---
        real_csv = self._load_real_csv(self.real_path)
        df_real = pd.read_csv(real_csv)
        n_rows.append(self._n_row(df_real, "REAL", n_reps=1))
        logger.info(f"[Distribution] Loaded REAL CSV: {real_csv} ({len(df_real)} rows)")

        self._detect_numeric_cols(df_real)

        dist_rows = []
        missing_rows = []
        range_rows = []
        continuous_rows = []

        dist_rows += self._build_distribution_rows(df_real, df_real, "REAL")

        # REAL rows for tables where relevant
        continuous_rows += self._build_continuous_table(df_real, df_real, "REAL")
        missing_rows += self._build_missingness_rows(df_real, "REAL")

        # Boundary adherence: include REAL with NA adherence but with Range and Mean±SD
        boundary_rows += self._build_boundary_adherence_table(df_real, df_real, "REAL")

        # Category adherence: include REAL (adherence NA), proportions used only for appendix
        category_rows += self._build_category_table(df_real, df_real, "REAL")

        # Range sanity: include REAL baseline row (see change in section 4)
        range_rows += self._build_range_sanity_rows(df_real, df_real, "REAL")

        # --- Add REAL to distribution rows --

        summary_rows = []

        for tech, path in self.technique_paths.items():

            logger.info(f"[Distribution] Processing technique: {self.tech_to_col(tech)}")
            csvs = self._load_csvs(tech, path)
            logger.info(f"[Distribution] {self.tech_to_col(tech)}: {len(csvs)} CSV(s)")

            # --- Skip diagnostics for REAL ---
            if tech == "REAL":
                continue
            
            df_syn = self._load_and_concat(csvs)

            common = sorted(set(df_real.columns) & set(df_syn.columns))
            df_real_use = df_real[common]
            df_syn_use  = df_syn[common]


            summary_rows += self._aggregate_distribution_summary(
                df_real_use,
                df_syn_use,
                tech
            )

            n_rows.append(self._n_row(df_syn_use, tech, n_reps=len(csvs)))

            continuous_rows += self._build_continuous_table(
                df_real_use, df_syn_use, tech
            )

            boundary_rows += self._build_boundary_adherence_table(
                df_real_use, df_syn_use, tech
            )
            
            category_rows += self._build_category_table(
                df_real_use, df_syn_use, tech
            )


            dist_rows += self._build_distribution_rows(
                df_real_use, df_syn_use, tech
            )

            missing_rows += self._build_missingness_rows(
                df_syn_use, tech
            )

            range_rows += self._build_range_sanity_rows(
                df_real_use, df_syn_use, tech
            )

            # self._plot_hist_grid(df_real_use, df_syn_use, tech, metrics_rows)
            # self._plot_corr_diff_heatmap(df_real_use, df_syn_use, tech)
            logger.info(f"[Distribution] {self.tech_to_col(tech)}: Plots generated")

        df_n = pd.DataFrame(n_rows).sort_values("Technique").reset_index(drop=True)
        df_cont = self._sort_by_variable_then_technique(pd.DataFrame(continuous_rows))
        df_bound = self._sort_by_variable_then_technique(pd.DataFrame(boundary_rows))
        df_miss = self._sort_by_variable_then_technique(pd.DataFrame(missing_rows))
        df_range = self._sort_by_variable_then_technique(pd.DataFrame(range_rows))
        df_cat = self._sort_by_variable_then_technique(pd.DataFrame(category_rows))
        df_bound_diff = self._build_boundary_diff_vs_real(df_bound)

        export_table(
            df=df_bound_diff,
            path=self.tables_dir / "table_boundary_adherence_diff_vs_real",
            config=self.export_config,
            table_number=1,
            title="Boundary Adherence Differences Relative to Real Data",
            note="Rows report differences in mean and standard deviation between synthetic and real distributions.",
            index=False,
        )


        # Sample size
        export_table(
            df=df_n,
            path=self.tables_dir / "table_sample_size",
            config=self.export_config,
            table_number=2,
            title="Effective Sample Size per Technique",
            note="Sample size is reported per replicate and averaged across synthetic datasets.",
            index=False,
        )


        # Continuous summary (pivoted, publication-ready)
        df_cont_pivot = self._pivot_continuous_summary(df_cont)

        export_table(
            df=df_cont_pivot,
            path=self.tables_dir / "table_continuous_summary",
            config=self.export_config,
            table_number=3,
            title="Continuous Variable Distribution Summary",
            note="Mean ± SD and range are reported for each continuous variable by technique.",
            index=False,
        )



        # Boundary adherence (pivoted, publication-ready)
        df_bound_pivot = self._pivot_boundary_adherence(df_bound)

        export_table(
            df=df_bound_pivot,
            path=self.tables_dir / "table_boundary_adherence",
            config=self.export_config,
            table_number=4,
            title="Boundary Adherence Statistics by Variable",
            note="Mean, standard deviation, and observed range are reported for real and synthetic data.",
            index=False,
        )

        # Missingness diagnostics
        export_table(
            df=df_miss,
            path=self.tables_dir / "table_missingness_diagnostics",
            config=self.export_config,
            table_number=5,
            title="Missingness Diagnostics by Variable and Technique",
            note="Missingness is reported as the percentage of missing values per variable.",
            index=False,
        )


        # Range sanity vs REAL
        export_table(
            df=df_range,
            path=self.tables_dir / "table_range_sanity_vs_real",
            config=self.export_config,
            table_number=6,
            title="Range Sanity Checks Relative to Real Data",
            note="Check marks indicate whether the synthetic value range exactly matches the real range after rounding.",
            index=False,
        )


        export_table(
            df=pd.DataFrame(dist_rows),
            path=self.appendix_dir / "distribution_comparison_full",
            config=self.export_config,
            table_number=7,
            title="Full Distribution Comparison Metrics",
            note="Table reports per-variable distribution statistics for real and synthetic data.",
            index=False,
        )


        # Appendix version (with proportions)
        export_table(
            df=df_cat,
            path=self.appendix_dir / "table_category_adherence",
            config=self.export_config,
            table_number=8,
            title="Category Adherence and Proportions by Variable",
            note="Category proportions are shown for descriptive purposes; adherence is undefined for real data.",
            index=False,
        )


        # Tables version (pivoted, publication-ready)
        df_cat_main = df_cat.drop(columns=["Category_Proportions"], errors="ignore")
        df_cat_pivot = self._pivot_category_adherence(df_cat_main)

        export_table(
            df=df_cat_pivot,
            path=self.tables_dir / "table_category_adherence",
            config=self.export_config,
            table_number=9,
            title="Category Adherence Summary by Variable",
            note="Values report category adherence for synthetic techniques only.",
            index=False,
        )


        df_summary = pd.DataFrame(summary_rows)

        df_continuous = df_summary[df_summary["Variable_Type"] == "continuous"]
        df_continuous = df_summary[
            df_summary["Variable_Type"] == "continuous"
        ][
            ["Technique", "Mean_|Δμ|", "Mean_|Δσ|", "Mean_Adherence"]
        ].sort_values("Technique")
        df_categorical = df_summary[
            df_summary["Variable_Type"] == "ordinal/categorical"
        ][
            ["Technique", "Mean_Adherence"]
        ].sort_values("Technique")

        df_continuous = df_continuous.rename(
            columns={"Mean_Adherence": "Mean_Continuous_Adherence"}
        )

        df_categorical = df_categorical.rename(
            columns={"Mean_Adherence": "Mean_Category_Adherence"}
        )

        export_table(
            df=df_continuous,
            path=self.tables_dir / "table_distribution_summary_continuous",
            config=self.export_config,
            table_number=10,
            title="Continuous Distribution Fidelity Summary",
            note="Metrics summarize standardized deviations in mean and standard deviation, and boundary adherence.",
            index=False,
        )

        export_table(
            df=df_categorical,
            path=self.tables_dir / "table_distribution_summary_categorical",
            config=self.export_config,
            table_number=11,
            title="Categorical and Ordinal Distribution Fidelity Summary",
            note="Metrics report mean category adherence aggregated across variables.",
            index=False,
        )

        logger.info("===== DistributionEvaluator complete =====")

        metrics_csv = self.appendix_dir / "marginal_distances.csv"
        pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False)
        logger.info(f"Saved marginal distance metrics: {metrics_csv}")

    def _sort_by_variable_then_technique(self, out: pd.DataFrame) -> pd.DataFrame:
        if "Technique" in out.columns:
            techs = ["REAL"] + sorted([t for t in out["Technique"].unique() if t != "REAL"])
            out["Technique"] = pd.Categorical(out["Technique"], categories=techs, ordered=True)

        if "Variable" in out.columns and "Technique" in out.columns:
            out = out.sort_values(["Variable", "Technique"])
        elif "Technique" in out.columns:
            out = out.sort_values(["Technique"])

        return out.reset_index(drop=True)

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

    @staticmethod
    def _fmt_mean_sd(mean, sd, digits=2):
        if pd.isna(mean) or pd.isna(sd):
            return ""
        return f"{mean:.{digits}f} ± {sd:.{digits}f}"


    @staticmethod
    def _n_row(df: pd.DataFrame, technique: str, n_reps: int) -> dict:
        n_reps = max(int(n_reps), 1)
        n_per_rep = len(df) / n_reps
        return {
            "Metric": "N",
            "Technique": technique,
            "Value": int(round(n_per_rep))
        }

    def _build_continuous_table(self, df_real, df_syn, technique):
        rows = []

        for col in self.numeric_cols:
            if col in self.categorical_cols or col in self.ordinal_cols:
                continue

            s = pd.to_numeric(df_syn[col], errors="coerce")

            rows.append({
                "Variable": col,
                "Technique": technique,
                "Mean ± SD": f"{s.mean():.3f} ± {s.std():.3f}",
                "Range": f"{int(s.min())} to {int(s.max())}",
            })

        return rows

    def _build_category_table(self, df_real, df_syn, technique):
        rows = []

        for col in self.categorical_cols | self.ordinal_cols:
            s = df_syn[col]

            props = (
                s.value_counts(normalize=True)
                .sort_index()
                .apply(lambda x: f"{x*100:.1f}%")
                .to_dict()
            )

            rows.append({
                "Variable": col,
                "Technique": technique,
                "Type": "categorical" if col in self.categorical_cols else "ordinal",
                "Category_Adherence": (
                    np.nan if technique == "REAL"
                    else self._category_adherence(df_real[col], df_syn[col])
                ),
                "Category_Proportions": ", ".join(
                    f"{k}: {v}" for k, v in props.items()
                ),
            })

        return rows


    # ======================================================
    # CSV LOADING
    # ======================================================
    def _load_real_csv(self, real_path: Path) -> Path:
        csv = Path(real_path, f"df_core{self.cnt}.csv")

        if not csv.exists():
            logger.error(f"REAL CSV not found: {csv}")
            raise FileNotFoundError(f"REAL CSV not found: {csv}")

        logger.debug(f"Using REAL CSV: {csv}")
        return csv

    def _load_csvs(self, tech: str, path: Path) -> list[Path]:
        logger.debug(f"Loading corrected CSVs for {self.tech_to_col(tech)} from {path}")

        if tech == "REAL":
            csv = path / f"df_core{self.cnt}.csv"
            if not csv.exists():
                logger.error(f"REAL CSV not found: {csv}")
                raise FileNotFoundError(f"REAL CSV not found: {csv}")
            return [csv]

        # 🔒 ONLY corrected synthetic files
        csvs = sorted(path.glob("*_corrected.csv"))

        if not csvs:
            logger.error(f"No corrected CSVs found for {self.tech_to_col(tech)} in {path}")
            raise FileNotFoundError(
                f"No *_corrected.csv files found for {self.tech_to_col(tech)} in {path}"
            )

        logger.debug(f"{self.tech_to_col(tech)}: corrected CSVs = {[c.name for c in csvs]}")
        return csvs


    def _build_boundary_adherence_table(self, df_real, df_syn, technique):

        rows = []

        for col in self.numeric_cols:

            if col in self.categorical_cols or col in self.ordinal_cols:
                continue
            s = pd.to_numeric(df_syn[col], errors="coerce")

            # REAL baseline stats per variable (computed from df_real)
            r = pd.to_numeric(df_real[col], errors="coerce")
            real_range = f"{int(r.min())} to {int(r.max())}" if r.notna().any() else ""
            syn_range = f"{int(s.min())} to {int(s.max())}" if s.notna().any() else ""


            real_mean = r.mean()
            real_sd   = r.std()
            syn_mean  = s.mean()
            syn_sd    = s.std()


            if technique == "REAL":
                rows.append({
                    "Variable": col,
                    "Technique": "REAL",
                    "Mean": real_mean,
                    "SD": real_sd,
                    "Range": real_range
                })
            else:
                rows.append({
                    "Variable": col,
                    "Technique": technique,
                    "Mean": syn_mean,
                    "SD": syn_sd,
                    "Range": syn_range
                })


        return rows

    @staticmethod
    def _load_and_concat(csvs: list[Path]) -> pd.DataFrame:
        dfs = [pd.read_csv(c) for c in csvs]
        return pd.concat(dfs, ignore_index=True)

    

    @staticmethod
    def _boundary_adherence(real: pd.Series, syn: pd.Series) -> float:
        metric = BoundaryAdherence()
        return metric.compute(
            real_data=real.dropna(),
            synthetic_data=syn.dropna()
        )


    @staticmethod
    def _category_adherence(real: pd.Series, syn: pd.Series) -> float:
        metric = CategoryAdherence()
        return metric.compute(
            real_data=real.dropna(),
            synthetic_data=syn.dropna()
        )


    def _build_distribution_rows(self, df_real, df_syn, technique):
        rows = []

        # ----- Continuous / numeric -----
        for col in self.numeric_cols:
            r = pd.to_numeric(df_real[col], errors="coerce")
            s = pd.to_numeric(df_syn[col], errors="coerce")

            rows.append({
                "Variable": col,
                "Technique": technique,
                "Type": "continuous",
                "Mean ± SD": f"{s.mean():.3f} ± {s.std():.3f}",
                "Range": f"{int(s.min())} to {int(s.max())}",
                "Boundary_Adherence": self._boundary_adherence(r, s),
            })

        # ----- Categorical / ordinal -----
        for col in self.categorical_cols | self.ordinal_cols:
            r = df_real[col]
            s = df_syn[col]

            props = (
                s.value_counts(normalize=True)
                .sort_index()
                .apply(lambda x: f"{x*100:.1f}%")
                .to_dict()
            )

            rows.append({
                "Variable": col,
                "Technique": technique,
                "Type": "categorical" if col in self.categorical_cols else "ordinal",
                "Category_Proportions": ", ".join(f"{k}: {v}" for k, v in props.items()),
                "Category_Adherence": self._category_adherence(r, s),
            })

        return rows


    def _build_missingness_rows(self, df, technique):
        return [
            {
                "Variable": c,
                "Technique": technique,
                "Missing_%": df[c].isna().mean() * 100,
            }
            for c in df.columns
        ]


    def _build_range_sanity_rows(self, df_real, df_syn, technique):
        rows = []

        for col in self.numeric_cols:
            r = pd.to_numeric(df_real[col], errors="coerce")
            s = pd.to_numeric(df_syn[col], errors="coerce")

            rmin_raw, rmax_raw = r.min(), r.max()
            smin_raw, smax_raw = s.min(), s.max()

            # ---- round to 2 decimals FIRST ----
            rmin = round(rmin_raw, 2)
            rmax = round(rmax_raw, 2)
            smin = round(smin_raw, 2)
            smax = round(smax_raw, 2)

            real_range = f"[{rmin}, {rmax}]"
            syn_range = f"[{smin}, {smax}]"

            if technique == "REAL":
                rows.append({
                    "Variable": col,
                    "Technique": "REAL",
                    "REAL_Range": real_range,
                    "Synthetic_Range": real_range,
                    "Same_As_REAL": "-",
                })
            else:
                rows.append({
                    "Variable": col,
                    "Technique": technique,
                    "REAL_Range": real_range,
                    "Synthetic_Range": syn_range,
                    # ---- compare ROUNDED values ----
                    "Same_As_REAL": "✓" if (smin == rmin and smax == rmax) else "✗",
                })

        return rows


    @staticmethod
    def _wasserstein(a, b):
        if len(a) == 0 or len(b) == 0:
            return np.nan
        return wasserstein_distance(a, b)

    @staticmethod
    def _js_divergence(a, b):
        all_vals = sorted(set(a) | set(b))
        pa = pd.Series(a).value_counts(normalize=True).reindex(all_vals, fill_value=0).values
        pb = pd.Series(b).value_counts(normalize=True).reindex(all_vals, fill_value=0).values
        return jensenshannon(pa, pb)
    
    def _detect_numeric_cols(self, df: pd.DataFrame):
        numeric = df.select_dtypes(include=[np.number]).columns
        self.numeric_cols = [
            c for c in numeric
            if c not in self.categorical_cols
            and c not in self.ordinal_cols
        ]

    # ======================================================
    # STAT COMPUTATION
    # ======================================================

    def _aggregate_distribution_summary(
        self,
        df_real: pd.DataFrame,
        df_syn: pd.DataFrame,
        technique: str,
    ) -> list[dict]:
        """
        Aggregated distribution fidelity summary for Section 4.1.

        Computes per-technique averages:
        - Mean absolute deviation of mean (|Δμ|)
        - Mean absolute deviation of std  (|Δσ|)
        - Mean Boundary Adherence

        Aggregated by variable type:
        - continuous
        - ordinal
        - categorical
        """

        rows = []

        # ---------- CONTINUOUS ----------
        cont_deltas_mu = []
        cont_deltas_sd = []
        cont_boundary  = []

        for col in self.numeric_cols:
            if col in self.categorical_cols or col in self.ordinal_cols:
                continue

            r = pd.to_numeric(df_real[col], errors="coerce")
            s = pd.to_numeric(df_syn[col], errors="coerce")

            if r.notna().any() and s.notna().any():
                r_std = r.std()

                if r_std > 0:
                    cont_deltas_mu.append(abs(s.mean() - r.mean()) / r_std)
                    cont_deltas_sd.append(abs(s.std() - r.std()) / r_std)

                cont_boundary.append(self._boundary_adherence(r, s))

        if cont_deltas_mu:
            rows.append({
                "Technique": technique,
                "Variable_Type": "continuous",
                "Mean_|Δμ|": round(np.mean(cont_deltas_mu), 2),
                "Mean_|Δσ|": round(np.mean(cont_deltas_sd), 2),
                "Mean_Adherence": round(np.mean(cont_boundary), 4),
            })

        # ---------- ORDINAL + CATEGORICAL ----------
        cat_adherence = []

        for col in self.categorical_cols | self.ordinal_cols:
            r = df_real[col]
            s = df_syn[col]

            if r.notna().any() and s.notna().any():
                cat_adherence.append(self._category_adherence(r, s))

        if cat_adherence:
            rows.append({
                "Technique": technique,
                "Variable_Type": "ordinal/categorical",
                "Mean_|Δμ|": "-",
                "Mean_|Δσ|": "-",
                "Mean_Adherence": round(np.mean(cat_adherence), 4),
            })

        return rows


    def _plot_hist_grid(self, df_real, df_syn, technique, metrics_rows):
        cols = df_real.columns.tolist()
        grid_cols = 5
        grid_rows = math.ceil(len(cols) / grid_cols)

        fig, axes = plt.subplots(
            grid_rows, grid_cols,
            figsize=(24, 4.2 * grid_rows),
            sharey=False
        )
        axes = axes.flatten()

        for i, col in enumerate(cols):
            ax = axes[i]

            real_vals = df_real[col].dropna()
            syn_vals  = df_syn[col].dropna()

            ax.hist(real_vals, bins=30, density=True, alpha=0.65,
                    edgecolor="black", linewidth=0.4, label="Real")
            ax.hist(syn_vals, bins=30, density=True, histtype="step",
                    linewidth=1.8, label="Synthetic")

            if col in self.categorical_cols or col in self.ordinal_cols:
                dist = self._js_divergence(real_vals, syn_vals)
                label = f"JS={dist:.2f}"
                vtype = "categorical"
            else:
                dist = self._wasserstein(real_vals, syn_vals)
                label = f"W={dist:.2f}"
                vtype = "continuous"

            ax.set_title(f"{col}\n{label}", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.tick_params(axis="both", labelsize=8)

            metrics_rows.append({
                "technique": technique,
                "variable": col,
                "type": vtype,
                "distance": dist
            })

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"REAL vs {technique} — Marginal Distributions", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")

        out = self.figures_dir / f"hist_grid_{technique}.png"
        plt.savefig(out, dpi=220)
        plt.close()

    def _plot_corr_diff_heatmap(self, df_real, df_syn, technique):
        real_corr = df_real.corr(numeric_only=True)
        syn_corr  = df_syn.corr(numeric_only=True)

        diff = real_corr - syn_corr

        plt.figure(figsize=(14, 12))
        sns.heatmap(
            diff,
            center=0,
            cmap="coolwarm",
            square=True,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            cbar_kws={"label": "Real − Synthetic"}
        )

        plt.title(f"Correlation Difference: {technique}")
        plt.tight_layout()

        out = self.figures_dir / f"corr_diff_{technique}.png"
        plt.savefig(out, dpi=200)
        plt.close()


    # ======================================================
    # HELPERS
    # ======================================================
    @staticmethod
    def _numeric(df, col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)

    @staticmethod
    def _mean(df, col):
        return df[col].mean() if col in df.columns else np.nan

    @staticmethod
    def _std(df, col):
        return df[col].std() if col in df.columns else np.nan

    @staticmethod
    def _binary_stats(mask1, mask2, name1, name2):
        return {
            f"{name1}_n": mask1.sum(),
            f"{name2}_n": mask2.sum(),
            f"{name1}_%": mask1.mean() * 100,
            f"{name2}_%": mask2.mean() * 100,
        }

    def _math_stats(self, df):
        pv_cols = [c for c in df.columns if c.startswith("PV") and "MATH" in c]
        if not pv_cols:
            logger.warning("No PV*MATH columns found")
            return {"Math_mean": np.nan, "Math_sd": np.nan}

        score = df[pv_cols].mean(axis=1)
        return {"Math_mean": score.mean(), "Math_sd": score.std()}

    def _pivot_category_adherence(self, df_cat: pd.DataFrame) -> pd.DataFrame:
        """
        Convert category adherence table into:
        - Techniques as columns (REAL REMOVED)
        - One metric row (Category Adherence) per Variable
        """

        records = []

        for var in df_cat["Variable"].unique():
            sub = df_cat[df_cat["Variable"] == var]

            row = {
                "Variable": var,
                "Metric": "Category Adherence"
            }

            for _, r in sub.iterrows():
                tech = r["Technique"]

                # ---- SKIP REAL entirely ----
                if tech == "REAL":
                    continue

                val = r.get("Category_Adherence", np.nan)
                row[tech] = "-" if pd.isna(val) else f"{val:.4f}"

            records.append(row)

        out = pd.DataFrame(records)

        # ---- Column ordering: alphabetical techniques only ----
        tech_cols = sorted(
            c for c in out.columns if c not in ("Variable", "Metric")
        )

        return out[["Variable", "Metric"] + tech_cols]

    def _pivot_boundary_adherence(self, df_bound: pd.DataFrame) -> pd.DataFrame:
        records = []

        for var in df_bound["Variable"].unique():
            sub = df_bound[df_bound["Variable"] == var]

            # ---- Mean ----
            row_mean = {"Variable": var, "Metric": "Mean"}
            for _, r in sub.iterrows():
                val = r.get("Mean", np.nan)
                row_mean[r["Technique"]] = "-" if pd.isna(val) else f"{val:.2f}"
            records.append(row_mean)

            # ---- SD ----
            row_sd = {"Variable": var, "Metric": "SD"}
            for _, r in sub.iterrows():
                val = r.get("SD", np.nan)
                row_sd[r["Technique"]] = "-" if pd.isna(val) else f"{val:.2f}"
            records.append(row_sd)

            # ---- Range ----
            row_range = {"Variable": var, "Metric": "Range"}
            for _, r in sub.iterrows():
                row_range[r["Technique"]] = r.get("Range", "")
            records.append(row_range)


        out = pd.DataFrame(records)

        tech_cols = ["REAL"] + sorted(
            c for c in out.columns if c not in ("Variable", "Metric", "REAL")
        )

        return out[["Variable", "Metric"] + tech_cols]



    def _build_boundary_diff_vs_real(self, df_bound: pd.DataFrame) -> pd.DataFrame:
        records = []

        for var in df_bound["Variable"].unique():
            sub = df_bound[df_bound["Variable"] == var]
            real_row = sub[sub["Technique"] == "REAL"].iloc[0]

            real_mean = real_row["Mean"]
            real_sd   = real_row["SD"]

            # ---- Mean difference ----
            row_mean = {"Variable": var, "Metric": "Δ Mean (Synthetic − Real)"}
            # ---- SD difference ----
            row_sd   = {"Variable": var, "Metric": "Δ SD (Synthetic − Real)"}

            for _, r in sub.iterrows():
                tech = r["Technique"]
                if tech == "REAL":
                    continue

                m = r["Mean"]
                s = r["SD"]

                row_mean[tech] = "-" if pd.isna(m) else f"{(m - real_mean):.2f}"
                row_sd[tech]   = "-" if pd.isna(s) else f"{(s - real_sd):.2f}"

            records.extend([row_mean, row_sd])

        out = pd.DataFrame(records)
        tech_cols = sorted(c for c in out.columns if c not in ("Variable", "Metric"))
        return out[["Variable", "Metric"] + tech_cols]



    def _pivot_continuous_summary(self, df_cont: pd.DataFrame) -> pd.DataFrame:
        """
        Convert long-format continuous summary into:
        - Techniques as columns
        - Metrics (Mean ± SD, Range) as rows per Variable
        """

        records = []

        for var in df_cont["Variable"].unique():
            sub = df_cont[df_cont["Variable"] == var]

            # ---- Mean ± SD row ----
            row_mean = {"Variable": var, "Metric": "Mean ± SD"}
            for _, r in sub.iterrows():
                row_mean[r["Technique"]] = r["Mean ± SD"]
            records.append(row_mean)

            # ---- Range row ----
            row_range = {"Variable": var, "Metric": "Range"}
            for _, r in sub.iterrows():
                row_range[r["Technique"]] = r["Range"]
            records.append(row_range)

        out = pd.DataFrame(records)

        # ---- Column ordering: Variable | Metric | REAL | others ----
        tech_cols = [c for c in out.columns if c not in ("Variable", "Metric")]
        tech_cols = ["REAL"] + sorted([c for c in tech_cols if c != "REAL"])

        return out[["Variable", "Metric"] + tech_cols]
