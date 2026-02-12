import os
import glob
import pandas as pd
import numpy as np

# =====================================================
# USER INPUT — RUN ALL DATA SOURCES
# =====================================================

BASE_DIR = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project"
CNT = "SGP"

EXPERIMENTS_DIR = os.path.join(BASE_DIR, "Experiments", "PISA-SEM", CNT)

DATA_SOURCE_PATHS = {

    "synthetic_great_distilgpt2": os.path.join(EXPERIMENTS_DIR, "GReaT", "DistilGPT2"),
    "synthetic_great_gpt2":        os.path.join(EXPERIMENTS_DIR, "GReaT", "GPT2"),

    "synthetic_tabula_distilgpt2": os.path.join(EXPERIMENTS_DIR, "Tabula", "DistilGPT2"),
    "synthetic_tabula_gpt2":        os.path.join(EXPERIMENTS_DIR, "Tabula", "GPT2"),

    "synthetic_taptap_distilgpt2": os.path.join(EXPERIMENTS_DIR, "TapTap", "DistilGPT2"),
    "synthetic_taptap_gpt2":        os.path.join(EXPERIMENTS_DIR, "TapTap", "GPT2"),

    "synthetic_predllm_distilgpt2": os.path.join(EXPERIMENTS_DIR, "PredLLM", "DistilGPT2"),
    "synthetic_predllm_gpt2":        os.path.join(EXPERIMENTS_DIR, "PredLLM", "GPT2"),

    "synthetic_tabdiff":           os.path.join(EXPERIMENTS_DIR, "TabDiff"),
    "synthetic_realtabformer":     os.path.join(EXPERIMENTS_DIR, "REaLTabFormer"),
}


# =====================================================
# SHEET CONFIGURATION
# =====================================================
SHEETS = [
    "table4_covariate_corr_SGP",
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

# =====================================================
# HELPERS
# =====================================================
def numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def id_cols(df):
    return [c for c in df.columns if c not in numeric_cols(df)]

def aggregate_numeric(df, group_cols):
    agg = df.groupby(group_cols, sort=False).agg(
        mean = pd.NamedAgg(column=None, aggfunc="mean")
    )
    return agg

def full_aggregate(df, group_cols):
    num_cols = numeric_cols(df)

    original_num_order = [c for c in df.columns if c in num_cols]

    agg = (
        df
        .groupby(group_cols, sort=False)[num_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    agg.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in agg.columns
    ]

    ordered_cols = []
    for c in original_num_order:
        for suffix in ["mean", "std"]:
            col_name = f"{c}_{suffix}"
            if col_name in agg.columns:
                ordered_cols.append(col_name)

    return agg[group_cols + ordered_cols]


def split_and_write_aggregates(
    df, group_cols, writer, sheet, num_cols
):
    # Preserve original column order
    original_num_order = [c for c in df.columns if c in num_cols]

    grouped = df.groupby(group_cols, sort=False)[num_cols]

    # ---- MEAN ----
    mean_df = grouped.mean().reset_index()
    mean_df = mean_df[group_cols + original_num_order]
    mean_df.to_excel(
        writer,
        sheet_name=f"{sheet}_mean"[:31],
        index=False
    )

    # ---- STD ----
    std_df = grouped.std().reset_index()
    std_df = std_df[group_cols + original_num_order]
    std_df.to_excel(
        writer,
        sheet_name=f"{sheet}_std"[:31],
        index=False
    )

    # ---- RANGE (min-max string) ----
    min_df = grouped.min()
    max_df = grouped.max()

    range_df = min_df.copy()
    for c in original_num_order:
        range_df[c] = (
            min_df[c].round(2).astype(str)
            + "-"
            + max_df[c].round(2).astype(str)
        )

    range_df = range_df.reset_index()
    range_df = range_df[group_cols + original_num_order]
    range_df.to_excel(
        writer,
        sheet_name=f"{sheet}_range"[:31],
        index=False
    )


# =====================================================
# RUN AGGREGATION FOR ALL DATA SOURCES
# =====================================================

for data_source, INPUT_DIR in DATA_SOURCE_PATHS.items():

    print("\n===================================================")
    print(f"Aggregating results for: {data_source}")
    print("Directory:", INPUT_DIR)
    print("===================================================")

    if not os.path.isdir(INPUT_DIR):
        print(f"[SKIP] Directory not found: {INPUT_DIR}")
        continue

    OUTPUT_FILE = os.path.join(INPUT_DIR, "SEM_AGGREGATED_RESULTS.xlsx")

    xlsx_files = [
        f for f in glob.glob(os.path.join(INPUT_DIR, "*.xlsx"))
        if not os.path.basename(f).startswith("SEM_AGGREGATED_RESULTS")
    ]

    if len(xlsx_files) < 2:
        print(f"[SKIP] Not enough Excel files to aggregate in {INPUT_DIR}")
        continue



    # =====================================================
    # MAIN AGGREGATION
    # =====================================================
    writer = pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter")

    for sheet in SHEETS:

        frames = []

        for f in xlsx_files:
            try:
                xl = pd.ExcelFile(f, engine="openpyxl")
                if sheet not in xl.sheet_names:
                    print(f"[WARN] {sheet} missing in {os.path.basename(f)} — skipped")
                    continue

                df = xl.parse(sheet)
                df["_source_file"] = os.path.basename(f)
                frames.append(df)

            except Exception as e:
                raise RuntimeError(f"Error reading {f}: {e}")

        if not frames:
            print(f"[SKIP] Sheet '{sheet}' not found in any replication — not aggregated")
            continue


        full_df = pd.concat(frames, ignore_index=True)

        # Identify grouping (non-numeric) columns
        group_cols = id_cols(full_df)
        group_cols = [c for c in group_cols if c != "_source_file"]

        num_cols = numeric_cols(full_df)

        # =================================================
        # 1. Fornell–Larcker & HTMT → mean / std / range
        # =================================================
        if sheet in ["pls_sem_fornell_larcker_SGP", "pls_sem_htmt_SGP"]:
            split_and_write_aggregates(
                full_df,
                group_cols,
                writer,
                sheet,
                num_cols
            )
            continue




        # =================================================
        # 3. ALL OTHER SHEETS → mean / std / range
        # =================================================
        else:
            split_and_write_aggregates(
                full_df,
                group_cols,
                writer,
                sheet,
                num_cols
            )
            continue



        # -------------------------------------------------
        # CB-SEM fit measures (KEEP AS SINGLE SHEET)
        # -------------------------------------------------
        if sheet == "sem_cb_fit_measures_SGP":

            original_num_order = [c for c in full_df.columns if c in num_cols]

            agg = (
                full_df
                .groupby(group_cols, sort=False)[num_cols]
                .mean()
                .reset_index()
            )

            agg = agg[group_cols + original_num_order]

            agg.to_excel(
                writer,
                sheet_name=sheet[:31],
                index=False
            )
            continue

        # -------------------------------------------------
        # ALL OTHER SHEETS → split into mean / std / range
        # -------------------------------------------------
        else:
            split_and_write_aggregates(
                full_df,
                group_cols,
                writer,
                sheet,
                num_cols
            )
            continue



    writer.close()

    print(f"Aggregated SEM results written to: {OUTPUT_FILE}")


