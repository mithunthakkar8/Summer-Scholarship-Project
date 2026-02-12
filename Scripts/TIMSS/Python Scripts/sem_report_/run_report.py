import logging
from sem_report.reporting.sem_synthetic_report import SEMSyntheticReport
from sem_report.loaders.sem_replication_aggregator import SEMReplicationAggregator
from sem_report.loaders.sem_comparison_builder import SEMComparisonBuilder
from sem_report.utils.export_config import ExportConfig, ExportFormat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

EXPORT_CONFIG = ExportConfig(
    fmt=ExportFormat.BOTH   # CSV | HTML | BOTH
)


if __name__ == "__main__":
    real_path = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Datasets\TIMSS 2023"

    technique_paths= {
            
            "GReaT_DistilGPT2": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\TIMSS-SEM\SGP\GReaT\DistilGPT2",
            "GReaT_GPT2":       r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\TIMSS-SEM\SGP\GReaT\GPT2",

            "Tabula_DistilGPT2": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\TIMSS-SEM\SGP\Tabula\DistilGPT2",
            "Tabula_GPT2":       r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\TIMSS-SEM\SGP\Tabula\GPT2",
            "TapTap_DistilGPT2": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\TIMSS-SEM\SGP\TapTap\DistilGPT2",
            "TapTap_GPT2":       r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\TIMSS-SEM\SGP\TapTap\GPT2",

            "PredLLM_DistilGPT2": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\TIMSS-SEM\SGP\PredLLM\DistilGPT2",
            "PredLLM_GPT2":       r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\TIMSS-SEM\SGP\PredLLM\GPT2",

            "TabDiff": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\TIMSS-SEM\SGP\TabDiff",
            "REaLTabFormer": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\TIMSS-SEM\SGP\REaLTabFormer",
        }
    
    SHEETS = [
        "covariate_corr_SGP",
        "pls_sem_full_indirect_SGP",
        "sem_full_std_paths_SGP",
        "sem_full_rsquared_SGP",
        "sem_full_total_effects_SGP",
        "pls_sem_reliability_SGP",
        "pls_sem_fornell_larcker_SGP",
        "pls_sem_htmt_SGP",
        "sem_cb_fit_measures_SGP",
        "sem_cb_rsquare_SGP",
        "sem_cb_correlations_SGP",
        "pls_sem_correlations_SGP",
        "pls_sem_loadings_R2_SGP",
        "pls_sem_mediation_SGP"
    ]
    # 1️⃣ Aggregate replicates
    aggregator = SEMReplicationAggregator(
        technique_paths=technique_paths,
        cnt="SGP",
        sheets=SHEETS
    )
    aggregator.run()

    builder = SEMComparisonBuilder(
        real_path = real_path,
        technique_paths=technique_paths,
        cnt="SGP",
        out_file=r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\TIMSS\SEM_TECHNIQUE_COMPARISON.xlsx"
    )
    builder.run()

    report = SEMSyntheticReport(
        comparison_xlsx=r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\TIMSS\SEM_TECHNIQUE_COMPARISON.xlsx",
        cnt="SGP",
        techniques=list(technique_paths.keys()),
        real_path= real_path,
        technique_paths= technique_paths,
        base_out_dir="sem_report/outputs",
        export_config=EXPORT_CONFIG
    )

    report.run_all()

