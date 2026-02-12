import logging
import os
import glob
import pandas as pd
from sem_report.reporting.sem_synthetic_report import SEMSyntheticReport
from sem_report.loaders.sem_replication_aggregator import SEMReplicationAggregator
from sem_report.loaders.sem_comparison_builder import SEMComparisonBuilder
from sem_report.utils.export_config import ExportConfig, ExportFormat
from sem_report.utils.synthetic_corrector import SyntheticDataCorrector
from sem_report.utils.run_script import run_r_sem_script



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
            "CTGAN": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\TIMSS-SEM\SGP\CTGAN",
            "GReaT": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\TIMSS-SEM\SGP\GReaT\DistilGPT2",
            "GReaTLarge": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\TIMSS-SEM\SGP\GReaT\GPT2Large",

            "Tabula": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\TIMSS-SEM\SGP\Tabula\DistilGPT2",
            "TapTap": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\TIMSS-SEM\SGP\TapTap\DistilGPT2",

            "PredLLM": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\TIMSS-SEM\SGP\PredLLM\DistilGPT2",
            "TabSyn": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\TIMSS-SEM\SGP\TabSyn",
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

    # ------------------------------------------------------------
    # Determine REAL dataset row count (upper bound for synthetic)
    # ------------------------------------------------------------
    CNT = 'SGP'

    # Assumption: all real CSVs have identical row counts
    real_df = pd.read_csv(os.path.join(real_path, f"df_core{CNT}.csv"))
    REAL_N_ROWS = len(real_df)

    logging.info(f"Real dataset row count detected: {REAL_N_ROWS}")


    # ============================================================
    # 0️⃣ Synthetic correction (NEW, FIRST)
    # ============================================================

    corrector = SyntheticDataCorrector(
        technique_paths=technique_paths,
        max_rows=REAL_N_ROWS,
        real_df = real_df
    )

    # corrector.run()

    # ------------------------------------------------
    # 1️⃣ SEM estimation (R)  ← BLOCKING & GUARDED
    # ------------------------------------------------
    # run_r_sem_script(
    #     r_script_path=r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Code Files\Scripts\TIMSS\R Scripts\timss_sem.R",
    #     r_executable="Rscript"   # or full path if needed
    # )

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


