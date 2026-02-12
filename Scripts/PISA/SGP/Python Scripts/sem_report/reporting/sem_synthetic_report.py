from pathlib import Path
import logging
from datetime import datetime
from sem_report.evaluators.correlations import CorrelationEvaluator
from sem_report.evaluators.distribution import DistributionEvaluator
from sem_report.loaders.sem_comparison_loader import SEMComparisonLoader
from sem_report.evaluators.covariate_correlation_evaluator import CovariateCorrelationEvaluator
from sem_report.evaluators.discriminant_validity import (
    DiscriminantValidityEvaluator
)
from sem_report.evaluators.loadings import LoadingsEvaluator
from sem_report.utils.export_config import ExportConfig
from sem_report.evaluators.reliability import ReliabilityEvaluator
from sem_report.evaluators.paths import StructuralPathEvaluator
from sem_report.evaluators.global_fit_measures import GlobalFitEvaluator
from sem_report.evaluators.summary_ranking import SummaryRankingEvaluator
from sem_report.evaluators.privacy_runner import PrivacyRunner
from sem_report.evaluators.indirect_effects import IndirectEffectsEvaluator



logger = logging.getLogger(__name__)


class SEMSyntheticReport:
    """
    Orchestrates the full SEM synthetic-data evaluation pipeline.

    This class:
      - owns the loader
      - instantiates evaluators
      - controls output folders
    """

    def __init__(
        self,
        comparison_xlsx: str | Path,
        cnt: str,
        techniques: list[str],
        real_path: str | Path,
        technique_paths,
        base_out_dir: str | Path,
        export_config: ExportConfig,
    ):
        self.cnt = cnt
        self.techniques = techniques
        self.technique_paths = technique_paths
        self.real_path = real_path
        self.export_config = export_config

        self.metadata = {
            "columns": {
                # Quasi-Identifiers
                "ST004D01T": {"sdtype": "categorical"}, # Gender
                "AGE": {"sdtype": "numerical"},
                "IMMIG": {"sdtype": "categorical"},
                "MISCED": {"sdtype": "categorical"},
                "ST001D01T": {"sdtype": "categorical"},
                
                # Sensitive Attributes (Scores)
                "PV1MATH": {"sdtype": "numerical"},
                "PV2MATH": {"sdtype": "numerical"},
                "ESCS": {"sdtype": "numerical"},
                
                # Attitudes (Likert scales 1-4 are usually categorical)
                "ST268Q01JA": {"sdtype": "categorical"},
                "ST268Q04JA": {"sdtype": "categorical"},
                
                # School context
                "SCHSIZE": {"sdtype": "numerical"},
                "MCLSIZE": {"sdtype": "numerical"},
            }
        }

        # -------------------------------------------------
        # Run-specific output directory
        # -------------------------------------------------
        run_id = f"{cnt}_{datetime.now():%Y%m%d_%H%M%S}"
        self.out_dir = Path(base_out_dir) / run_id
        self.out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing SEMSyntheticReport: {run_id}")
        logger.info(f"Output directory: {self.out_dir}")

        # -------------------------------------------------
        # Shared loader (single source of truth)
        # -------------------------------------------------
        self.loader = SEMComparisonLoader(comparison_xlsx)

        # -------------------------------------------------
        # Create output subfolders (defensive)
        # -------------------------------------------------
        for sub in ["appendix", "figures", "tables"]:
            (self.out_dir / sub).mkdir(exist_ok=True)

        

    def run_global_fit(self):
        logger.info("Running section: CB-SEM Global Fit")

        evaluator = GlobalFitEvaluator(
            loader=self.loader,
            cnt=self.cnt,
            techniques=self.techniques,
            out_dir=self.out_dir,
            export_config=self.export_config,
        )
        evaluator.run()


    def run_distribution(self):
        evaluator = DistributionEvaluator(
            real_path=self.real_path,
            technique_paths=self.technique_paths,
            cnt=self.cnt,
            out_dir=self.out_dir,
            categorical_cols = [
                "IMMIG", "ST004D01T", "ST001D01T"
            ],
            ordinal_cols = [
                "MISCED",
                "ST268Q01JA", "ST268Q04JA", "ST268Q07JA", "ST268Q01JA", "ST268Q04JA", "ST268Q07JA",
                "SC064Q05WA", "SC064Q06WA", "SC064Q01TA",
                "SC064Q02TA", "SC064Q04NA", "SC064Q03TA",
                "SC064Q07WA"
            ],
            export_config=self.export_config,
        )
        evaluator.run()

    def run_reliability(self):
        logger.info("Running section: Reliability")

        evaluator = ReliabilityEvaluator(
            loader=self.loader,
            cnt=self.cnt,
            techniques=self.techniques,
            out_dir=self.out_dir,
            export_config=self.export_config,
        )
        evaluator.run()

    def run_indirect_effects(self):
        logger.info("Running section: Indirect Effects (Mediation)")

        evaluator = IndirectEffectsEvaluator(
            loader=self.loader,
            cnt=self.cnt,
            techniques=self.techniques,
            out_dir=self.out_dir,
            export_config=self.export_config
        )
        evaluator.run()


    # =====================================================
    # PUBLIC ENTRY POINT
    # =====================================================
    def run_correlations(self):
        logger.info("Running section: Correlations")

        evaluator = CorrelationEvaluator(
            loader=self.loader,
            cnt=self.cnt,
            techniques=self.techniques,
            out_dir=self.out_dir,
            export_config=self.export_config
        )
        evaluator.run()

        ccevaluator = CovariateCorrelationEvaluator(
            loader=self.loader,
            cnt=self.cnt,
            techniques=self.techniques,
            out_dir=self.out_dir,
            export_config=self.export_config
        )
        ccevaluator.run()

    def run_loadings(self):
        logger.info("Running section: Loadings")

        evaluator = LoadingsEvaluator(
            loader=self.loader,
            cnt=self.cnt,
            techniques=self.techniques,
            out_dir=self.out_dir,
            export_config=self.export_config
        )
        evaluator.run()

    def run_paths(self):
        evaluator = StructuralPathEvaluator(
            loader=self.loader,
            cnt=self.cnt,          # ← THIS IS THE KEY FIX
            techniques=self.techniques,
            out_dir=self.out_dir,
            export_config=self.export_config,
        )
        evaluator.run()

    def run_privacy(self):
        logger.info("Running section: Privacy")

        runner = PrivacyRunner(
            real_path=self.real_path,
            technique_paths=self.technique_paths,
            out_dir=self.out_dir,
            export_config=self.export_config,
            # Updated to include more "clues" for better privacy testing
            quasi_identifiers=[
                "IMMIG", "ST004D01T", "ST001D01T", "AGE", "SCHSIZE",     # School size
                "MCLSIZE",     # Class size
            ],
            # Ensure these match the actual column names in your PISA CSV
            sensitive_attributes=[
                "PV1MATH", "ESCS"
            ]
        )

        runner.run()


    def run_all(self):
        """
        Run the full evaluation pipeline in paper order.
        """
        logger.info("===== STARTING SEM SYNTHETIC REPORT =====")

        self.run_distribution()
        self.run_discriminant_validity()
        self.run_correlations()
        self.run_loadings()
        self.run_reliability()
        self.run_paths()        
        self.run_indirect_effects()
        self.run_global_fit()
        self.run_privacy()
        self.run_summary()

        logger.info("===== SEM SYNTHETIC REPORT COMPLETE =====")

    # =====================================================
    # SECTION RUNNERS
    # =====================================================
    def run_discriminant_validity(self):
        logger.info("Running section: Discriminant Validity")

        evaluator = DiscriminantValidityEvaluator(
            loader=self.loader,
            cnt=self.cnt,
            techniques=self.techniques,
            out_dir=self.out_dir,
            export_config=self.export_config
        )

        evaluator.run()




    def run_summary(self):
        logger.info("Running section: Final Summary Ranking")

        evaluator = SummaryRankingEvaluator(
            out_dir=self.out_dir,
            techniques=self.techniques,
            export_config=self.export_config,
        )
        evaluator.run()


