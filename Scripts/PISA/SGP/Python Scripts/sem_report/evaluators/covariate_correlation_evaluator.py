from pathlib import Path
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

from sem_report.loaders.sem_comparison_loader import SEMComparisonLoader
from sem_report.utils.exporter import export_table
from sem_report.utils.export_config import ExportConfig

logger = logging.getLogger(__name__)


class CovariateCorrelationEvaluator:
    """
    Covariate correlation preservation vs REAL.

    Computes:
      - Absolute deviation |Δr| per variable–target
      - Summary stats per technique (mean / median / max |Δr|)
      - Heatmap of mean |Δr| over SMP/SPI/SMS
      - Interval vs REAL-dot table from range sheets
    """

    TARGETS = ["SMP", "SPI", "SMS"]

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
        self.export_config = export_config
        self.out_dir = Path(out_dir)

        self.appendix_dir = self.out_dir / "appendix" / "003_covariate_correlations"
        self.tables_dir = self.out_dir / "tables" / "003_covariate_correlations"
        self.figures_dir = self.out_dir / "figures" / "003_covariate_correlations"

        for d in [self.appendix_dir, self.tables_dir, self.figures_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _absolute_delta_long(self) -> pd.DataFrame:
        df = self.loader.covariate_corr_mean(self.cnt)

        records = []

        for _, row in df.iterrows():
            var = row["Variable"]

            for tgt in self.TARGETS:
                real = row.get(f"REAL__{tgt}", np.nan)
                if pd.isna(real):
                    continue

                tech_cols = [
                    c for c in df.columns
                    if c.endswith(f"__{tgt}") and not c.startswith("REAL__")
                ]

                for col in tech_cols:
                    tech = col.split("__", 1)[0]
                    synth = row[col]

                    if pd.notna(synth):
                        records.append({
                            "Technique": tech,
                            "Variable": var,
                            "Target": tgt,
                            "DeltaAbs": abs(synth - real),
                            "DeltaSigned": synth - real,
                            "r_REAL": real,
                            "r_SYNTH": synth,
                        })

        out = pd.DataFrame(records)

        if out.empty:
            raise RuntimeError(
                "No covariate correlation deltas computed. "
                "Check column prefixes in comparison workbook."
            )

        return out


    def _summary_table(self, long_df: pd.DataFrame) -> pd.DataFrame:
        """
        Canonical summary table.

        Rows    = techniques
        Columns = summary metrics
        """

        return (
            long_df
            .groupby("Technique")["DeltaAbs"]
            .agg(
                MeanAbsDelta_r="mean",
                MedianAbsDelta_r="median",
                MaxAbsDelta_r="max",
            )
            .round(3)
            .reset_index()
        )



    def _heatmap_matrix(self, long_df: pd.DataFrame) -> pd.DataFrame:
        return (
            long_df
            .groupby(["Variable", "Technique"])["DeltaAbs"]
            .mean()
            .reset_index()
            .pivot(index="Variable", columns="Technique", values="DeltaAbs")
            .sort_index()
        )

    def _plot_delta_heatmap(self, heat_df: pd.DataFrame):
        """ 
        Heatmap:
        rows = covariates
        columns = techniques
        values = mean |Δr| over SMP, SPI, SMS
        """
        fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(heat_df))))

        im = ax.imshow(heat_df.values, aspect="auto")

        ax.set_xticks(np.arange(len(heat_df.columns)))
        ax.set_yticks(np.arange(len(heat_df.index)))

        ax.set_xticklabels(heat_df.columns, rotation=45, ha="right")
        ax.set_yticklabels(heat_df.index)

        for i in range(len(heat_df.index)):
            for j in range(len(heat_df.columns)):
                val = heat_df.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        ax.set_title("Mean |Δr| over SMP, SPI, SMS (vs REAL)")
        fig.colorbar(im, ax=ax, label="Mean |Δr|")

        plt.tight_layout()
        plt.savefig(self.figures_dir / "heatmap_mean_abs_delta.png", dpi=300)
        plt.close()
    

    def _format_range_summary(self, ranges: pd.DataFrame) -> pd.DataFrame:
        """
        Rows    = variable → target
        Metrics = interval containment
        Columns = techniques
        """

        records = []

        for (var, tgt), g in ranges.groupby(["Variable", "Target"]):
            row = {
                "Variable": f"{var} → {tgt}",
                "Metric": r"$r_{\mathrm{REAL}} \in [\min(r_{\mathrm{SYN}}), \max(r_{\mathrm{SYN}})]$"
            }

            for _, r in g.iterrows():
                tech = r["Technique"]
                lo, hi = r["Synth_Lo"], r["Synth_Hi"]
                ok = r["Contains_REAL"]

                if pd.notna(lo) and pd.notna(hi):
                    CHECK = "✓"
                    CROSS = "✗"

                    row[tech] = (
                        f"[{lo:.2f}, {hi:.2f}] {CHECK}"
                        if ok else
                        f"[{lo:.2f}, {hi:.2f}] {CROSS}"
                    )
                else:
                    row[tech] = ""

            records.append(row)

        out = pd.DataFrame(records)
        tech_cols = sorted(c for c in out.columns if c not in ("Variable", "Metric"))
        return out[["Variable", "Metric"] + tech_cols]


    def _plot_interval_dot(self, ranges: pd.DataFrame):
        """
        Interval–dot plot:
        - horizontal bar = synthetic range
        - dot = REAL
        - grouped by Technique
        """
        for (var, tgt), g in ranges.groupby(["Variable", "Target"]):
            fig, ax = plt.subplots(figsize=(8, 0.4 * len(g)))

            y = np.arange(len(g))

            ax.hlines(
                y=y,
                xmin=g["Synth_Lo"],
                xmax=g["Synth_Hi"],
                color="gray",
                alpha=0.7
            )

            ax.scatter(
                g["REAL"],
                y,
                color="red",
                zorder=3,
                label="REAL"
            )

            ax.set_yticks(y)
            ax.set_yticklabels(g["Technique"])
            ax.set_xlabel("Correlation")

            ax.set_title(f"{var} → {tgt}: REAL vs Synthetic Ranges")

            plt.tight_layout()
            fname = f"interval_dot_{var}_{tgt}.png".replace(" ", "_")
            plt.savefig(self.figures_dir / fname, dpi=300)
            plt.close()


    def _range_table(self) -> pd.DataFrame:
        df_range = self.loader.covariate_corr_range(self.cnt)
        df_mean  = self.loader.covariate_corr_mean(self.cnt)

        rows = []

        # REAL correlations lookup
        real_map = (
            df_mean
            .set_index("Variable")[[f"REAL__{t}" for t in self.TARGETS]]
            .to_dict(orient="index")
        )

        for _, row in df_range.iterrows():
            var = row["Variable"]

            if var in self.TARGETS:
                continue

            for tgt in self.TARGETS:
                if var == tgt:
                    continue

                real = real_map.get(var, {}).get(f"REAL__{tgt}", np.nan)

                # discover techniques from *_min columns
                techs = {
                    c.split("__", 1)[0]
                    for c in df_range.columns
                    if c.endswith(f"__{tgt}_min")
                }

                for tech in techs:
                    lo = row.get(f"{tech}__{tgt}_min", np.nan)
                    hi = row.get(f"{tech}__{tgt}_max", np.nan)

                    rows.append({
                        "Variable": var,
                        "Target": tgt,
                        "Technique": tech,
                        "REAL": real,
                        "Synth_Lo": lo,
                        "Synth_Hi": hi,
                        "Contains_REAL": (
                            lo <= real <= hi
                            if pd.notna(lo) and pd.notna(hi) and pd.notna(real)
                            else np.nan
                        )
                    })

        return pd.DataFrame(rows)

    def run(self):
        logger.info("Running CovariateCorrelationEvaluator")

        long_df = self._absolute_delta_long()
        summary = self._summary_table(long_df)
        heat = self._heatmap_matrix(long_df).round(2)
        self._plot_delta_heatmap(heat)
        ranges = self._range_table()
        self._plot_interval_dot(ranges)

        range_summary = self._format_range_summary(ranges)

        export_table(
            range_summary,
            self.tables_dir / "table_covariate_corr_ranges",
            config=self.export_config,
            table_number=4,
            title="Covariate Correlation Ranges Relative to Real Data",
            note="Intervals represent the minimum and maximum synthetic correlations; check marks indicate containment of the real correlation.",
            index=False,
        )

        export_table(
            long_df,
            self.appendix_dir / "covariate_delta_long",
            config=self.export_config,
            table_number=5,
            title="Absolute and Signed Covariate Correlation Differences",
            note="Δr values are computed as synthetic minus real correlations for each covariate–target pair.",
            index=False,
        )
        
        # -------------------------------------------------
        # HTML: paper layout (metrics as rows)
        # -------------------------------------------------
        if self.export_config.export_html:
            html_df = (
                summary
                .set_index("Technique")
                .T
                .reset_index()
                .rename(columns={"index": "Metric"})
            )

            # ---------------------------------------------
            # Pretty metric labels (math + bold)
            # ---------------------------------------------
            METRIC_LABELS = {
                "MeanAbsDelta_r": "Mean |Δr|",
                "MedianAbsDelta_r": "Median |Δr|",
                "MaxAbsDelta_r": "Max |Δr|",
            }

            html_df["Metric"] = html_df["Metric"].map(METRIC_LABELS)

            # ---------------------------------------------
            # IMPORTANT: remove pandas metadata that leaks
            # into HTML headers
            # ---------------------------------------------
            html_df.index.name = None
            html_df.columns.name = None

            from sem_report.utils.export_config import ExportFormat

            export_table(
                html_df,
                self.tables_dir / "table_covariate_corr_error",
                config=ExportConfig(fmt=ExportFormat.HTML),
                table_number=6,
                title="Covariate Correlation Error Summary Across Techniques",
                note="Values report mean, median, and maximum absolute correlation deviations aggregated across SMP, SPI, and SMS.",
                index=False,
            )


        export_table(
            heat,
            self.appendix_dir / "covariate_delta_heatmap_matrix",
            config=self.export_config,
            table_number=7,
            title="Mean Absolute Covariate Correlation Differences by Technique",
            note="Cells report mean |Δr| values averaged across SMP, SPI, and SMS.",
            index=True,
        )


        export_table(
            ranges,
            self.appendix_dir / "covariate_corr_ranges_long",
            config=self.export_config,
            table_number=8,
            title="Synthetic Covariate Correlation Ranges and Real Containment",
            note="For each covariate–target pair, the table reports the synthetic correlation range and whether it contains the real correlation.",
            index=False,
        )

        logger.info("CovariateCorrelationEvaluator complete")
