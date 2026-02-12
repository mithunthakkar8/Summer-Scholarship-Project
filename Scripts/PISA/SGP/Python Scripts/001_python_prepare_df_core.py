# python_step1_prepare_df_core.py
import os
import numpy as np
import pandas as pd
import pyreadstat
import re

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Path configuration
# DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data/"
DATA_DIR = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Datasets\PISA 2022"
OUTPUT_DIR = DATA_DIR

student_path = os.path.join(DATA_DIR, "CY08MSP_STU_QQQ.SAS7BDAT")
school_path = os.path.join(DATA_DIR, "CY08MSP_SCH_QQQ.SAS7BDAT")

print("=== Step 1: Loading SAS files ===")
print(f"Student: {student_path}")
print(f"School: {school_path}")

# -----------------------------
# VARIABLES SELECTION
# -----------------------------
student_vars = [
    "CNT",           # country code
    "CNTSCHID",      # school ID
    "CNTSTUID",      # student ID
    # Covariates
    "ST001D01T",     # grade
    "ST004D01T",     # gender
    "MISCED",        # mother's education
    "ESCS",          # socioeconomic status
    "AGE",           # age
    "IMMIG",         # immigration status
    # SMS indicators
    "ST268Q01JA", "ST268Q04JA", "ST268Q07JA",
    # PVs
    "PV1MATH", "PV2MATH", "PV3MATH", "PV4MATH", "PV5MATH",
    "PV6MATH", "PV7MATH", "PV8MATH", "PV9MATH", "PV10MATH"
]

school_vars = [
    "CNT",           # country code
    "CNTSCHID",      # school ID
    # SPI indicators
    "SC064Q01TA", "SC064Q02TA", "SC064Q03TA", 
    "SC064Q04NA", "SC064Q05WA", "SC064Q06WA", "SC064Q07WA",
    # School characteristics
    "MCLSIZE",       # Math class size
    "SCHSIZE",       # School size
]

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------



def make_canonical_name(col: str) -> str:
    """
    Convert column name into a safe GPT-2 compatible version
    """
    s = col.lower()
    s = re.sub(r'[^a-z0-9]', '_', s)
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')
    return s if s else "col"

# -----------------------------
# LOAD SAS DATA
# -----------------------------
stu_df, stu_meta = pyreadstat.read_sas7bdat(student_path, usecols=student_vars)
sch_df, sch_meta = pyreadstat.read_sas7bdat(school_path, usecols=school_vars)

print(f"Original student shape: {stu_df.shape}")
print(f"Original school shape: {sch_df.shape}")

# -----------------------------
# FILTER COUNTRIES
# -----------------------------
# target_cnt = ["JPN", "KOR", "SGP", "TAP", "HKG", "MAC"]
CNT = "SGP"
# stu_df_cnt = stu_df[stu_df["CNT"].isin(target_cnt)]
# sch_df_cnt = sch_df[sch_df["CNT"].isin(target_cnt)]

stu_df = stu_df[stu_df["CNT"] == CNT]
sch_df = sch_df[sch_df["CNT"] == CNT]

print("Filtered student:", stu_df.shape)
print("Filtered school:", sch_df.shape)


print(f"\nFiltered student shape: {stu_df.shape}")
print(f"Filtered school shape: {sch_df.shape}")

# -----------------------------
# MERGE DATASETS
# -----------------------------
merged = stu_df.merge(sch_df, on=["CNT", "CNTSCHID"], how="inner")
print(f"Merged shape: {merged.shape}")

# -----------------------------
# DEFINE VARIABLE TYPES
# -----------------------------

categorical_cols = [
    # Identifiers (do NOT impute, but included for completeness)
    "CNT", "CNTSCHID", "CNTSTUID",

    # Student background (categorical / ordinal)
    "ST001D01T",     # grade
    "ST004D01T",     # gender
    "MISCED",        # mother's education
    "IMMIG",         # immigration status

    # SMS (Likert-type)
    "ST268Q01JA", "ST268Q04JA", "ST268Q07JA",

    
]

continuous_cols = [
    "AGE",
    "ESCS",
    "MCLSIZE",
    "SCHSIZE",
    # SPI items (ordinal)
    "SC064Q01TA", "SC064Q02TA", "SC064Q03TA",
    "SC064Q04NA", "SC064Q05WA", "SC064Q06WA", "SC064Q07WA",
] + [f"PV{i}MATH" for i in range(1, 11)]


# Define SEM item groups
spi_items = [
    "SC064Q01TA", "SC064Q02TA", "SC064Q03TA",
    "SC064Q05WA", "SC064Q06WA", "SC064Q04NA", "SC064Q07WA"
]
sms_items = ["ST268Q01JA", "ST268Q04JA", "ST268Q07JA"]
smp_items = [f"PV{i}MATH" for i in range(1, 11)]


# -------------------------------------- 
# REMOVE INCOMPLETE CASES for SEM ITEMS
# ---------------------------------------
for items, name in [
    (spi_items, "SPI"),
    (sms_items, "SMS"), 
    (smp_items, "SMP")
]:
    initial_count = merged.shape[0]
    merged = merged.dropna(subset=items, how='any')
    removed = initial_count - merged.shape[0]
    print(f"Removed {removed} rows with incomplete {name} items")


# --------------------------------------
# IMPUTE MISSING VALUES (POST-SEM-FILTER)
# --------------------------------------

print("\n=== Imputing missing values ===")

# --- Safety checks ---
cat_cols = [c for c in categorical_cols if c in merged.columns]
cont_cols = [c for c in continuous_cols if c in merged.columns]

# Exclude identifiers explicitly (defensive)
id_cols = {"CNT", "CNTSCHID", "CNTSTUID"}
cat_cols = [c for c in cat_cols if c not in id_cols]
cont_cols = [c for c in cont_cols if c not in id_cols]

# --- Impute categorical columns with mode ---
for col in cat_cols:
    n_missing = merged[col].isna().sum()
    if n_missing > 0:
        mode_val = merged[col].mode(dropna=True)
        if not mode_val.empty:
            merged[col] = merged[col].fillna(mode_val.iloc[0])
        print(f"Categorical: {col} | imputed {n_missing} with mode")

# --- Impute continuous columns with median ---
for col in cont_cols:
    n_missing = merged[col].isna().sum()
    if n_missing > 0:
        median_val = merged[col].median()
        merged[col] = merged[col].fillna(median_val)
        print(f"Continuous: {col} | imputed {n_missing} with median")

# --- Final assertion (critical for TabDiff) ---
remaining_na = merged[cat_cols + cont_cols].isna().sum().sum()
if remaining_na > 0:
    raise ValueError(f"Imputation incomplete: {remaining_na} NaNs remain")

print("✔ Imputation completed successfully")

# -----------------------------
# SAVE BASE DATASET
# -----------------------------
df_core = merged.copy()
out_csv_core = os.path.join(OUTPUT_DIR, f"df_core{CNT}_with_keys.csv")
df_core.to_csv(out_csv_core, index=False)
print(f"\n✔ Saved df_core to: {out_csv_core}")

# -----------------------------
# DROP IDENTIFIER COLUMNS
# -----------------------------
cols_to_drop = ["CNT", "CNTSCHID", "CNTSTUID"]
merged = merged.drop(columns=[c for c in cols_to_drop if c in merged.columns])


# -----------------------------
# SAVE BASE DATASET
# -----------------------------
df_core = merged.copy()
out_csv_core = os.path.join(OUTPUT_DIR, f"df_core{CNT}.csv")
df_core.to_csv(out_csv_core, index=False)
print(f"\n✔ Saved df_core to: {out_csv_core}")

# =====================================================================
# CREATE VARIABLE MAPPINGS
# =====================================================================

# Extract variable metadata
stu_map = pd.DataFrame({
    "code": stu_meta.column_names,
    "full_name": stu_meta.column_labels,
    "dataset_level": "student"
})

sch_map = pd.DataFrame({
    "code": sch_meta.column_names,
    "full_name": sch_meta.column_labels,
    "dataset_level": "school"
})

# Combine and clean mappings
mapping = pd.concat([stu_map, sch_map], ignore_index=True)
mapping = mapping.drop_duplicates(subset=["code"])
mapping = mapping[mapping["full_name"].notnull()]
mapping["canonical_name"] = mapping["full_name"].apply(make_canonical_name)

# Save full mapping
mapping_path = os.path.join(OUTPUT_DIR, "pisa_variable_mapping.csv")
mapping.to_csv(mapping_path, index=False)
print(f"✔ Saved variable mapping: {mapping_path}")

# Create lookup dictionary
fullname_lookup = dict(zip(mapping["code"], mapping["full_name"]))


# =====================================================================
# FINAL SUMMARY
# =====================================================================
print("\n" + "="*50)
print("PREPROCESSING COMPLETE")
print("="*50)
print("Generated files:")
print(f"  1. df_core.csv                    (SEM-ready dataset)")
print(f"  2. df_core_fullnames.csv          (GReaT-ready dataset)")
print(f"  3. pisa_variable_mapping.csv      (Variable metadata)")
print("="*50)
