from pathlib import Path
import logging
import pandas as pd

from sem_report.evaluators.privacy import PrivacyEvaluator
from sem_report.utils.exporter import export_table
from sem_report.utils.export_config import ExportConfig

logger = logging.getLogger(__name__)


class PrivacyRunner:
    """
    Orchestrates privacy evaluation across techniques + replicates.
    Writes mean privacy metrics to disk.
    """

    def __init__(
        self,
        real_path: Path,
        technique_paths: dict[str, Path],
        out_dir: Path,
        export_config: ExportConfig,
        quasi_identifiers=None,
        sensitive_attributes=None,
        metadata=None, 
    ):
        self.real_path = Path(real_path)
        self.technique_paths = technique_paths
        self.out_dir = Path(out_dir)
        self.export_config = export_config

        self.qi = quasi_identifiers or []
        self.sa = sensitive_attributes or []

        self.tables_dir = self.out_dir / "tables" / "008_privacy"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata

    # --------------------------------------------------
    def run(self):
        logger.info("Running Privacy Evaluation")

        real_df = pd.read_csv(Path(self.real_path, f"df_coreSGP.csv"))

        rows = []

        for technique, tech_dir in self.technique_paths.items():
            logger.info("Privacy: %s", technique)

            for csv in sorted(Path(tech_dir).glob("*.csv")):
                if "log" in csv.name.lower():
                    continue
                
                synth_df = pd.read_csv(csv)

                evaluator = PrivacyEvaluator(
                    real_df=real_df,
                    synthetic_df=synth_df,
                    technique=technique,
                    quasi_identifiers=self.qi,
                    sensitive_attributes=self.sa,
                )

                res = evaluator.run()

                rows.append({
                    "Technique": technique,
                    "Exact Match Rate": res["Tier1"]["ExactMatchRate"],
                    "NNDR": res["Tier1"]["NNDR"],
                    "Membership Inference risk": res["Tier1"]["MembershipInference"],
                    "DCR_p05": res["Tier1"]["DCR"]["p05"],

                    # Tier-2 (appendix only)
                    "KAnonymity_mean_k_distance": (
                        res["Tier2"]["KAnonymity"]["mean_k_distance"]
                        if res["Tier2"]["KAnonymity"] is not None else None
                    ),
                })

        df = pd.DataFrame(rows)

        # ---- Mean over replicates ----
        mean_df = (
            df
            .groupby("Technique", as_index=False)
            .mean(numeric_only=True)
        )

        # ---- Wide format (Metric × Technique) ----
        wide = (
            mean_df
            .set_index("Technique")
            .T
            .reset_index()
            .rename(columns={"index": "Metric"})
        )

        export_table(
            df=wide,
            path=self.tables_dir / "privacy_mean",
            config=self.export_config,
            table_number=27,
            title="Mean Privacy Risk Metrics Across Synthetic Data Techniques",
            note="Reported values are averaged across synthetic replicates. Lower values indicate better privacy protection except for NNDR, where higher values indicate lower disclosure risk.",
            index=False,
        )


        kanon = (
            df.groupby("Technique", as_index=False)["KAnonymity_mean_k_distance"]
            .mean()
        )

        export_table(
            df=kanon,
            path=self.tables_dir / "privacy_k_anonymity_mean",
            config=self.export_config,
            table_number=28,
            title="Mean k-Anonymity Distance Across Techniques",
            note="Mean k-distance reflects average distance to the k-th nearest neighbor in the real dataset; larger values indicate stronger anonymity.",
            index=False,
        )

        logger.info("Privacy tables written")
