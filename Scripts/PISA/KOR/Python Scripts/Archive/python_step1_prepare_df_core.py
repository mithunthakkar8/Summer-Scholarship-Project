# python_step1_prepare_df_core.py
import os
import numpy as np
import pandas as pd
import pyreadstat

DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data/"
# DATA_DIR = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Datasets\PISA 2022"
OUTPUT_DIR = DATA_DIR

student_path = os.path.join(DATA_DIR, "CY08MSP_STU_QQQ.SAS7BDAT")
school_path  = os.path.join(DATA_DIR, "CY08MSP_SCH_QQQ.SAS7BDAT")

print("=== Step 1: Loading SAS files ===")
print("Student:", student_path)
print("School :", school_path)

# -----------------------------
# VARIABLES SELECTED
# -----------------------------
student_vars = [
    "CNT","CNTSCHID",
    "ST001D01T","ST004D01T","MISCED","ESCS",
    "ST268Q01JA","ST268Q04JA","ST268Q07JA",
    "PV1MATH","PV2MATH","PV3MATH","PV4MATH","PV5MATH",
    "PV6MATH","PV7MATH","PV8MATH","PV9MATH","PV10MATH"
]

school_vars = [
    "CNT","CNTSCHID",
    "SC064Q01TA","SC064Q02TA","SC064Q03TA",
    "SC064Q04NA","SC064Q05WA","SC064Q06WA",
    "SCHSIZE"
]

# -----------------------------
# READ SAS FILES + METADATA (labels!)
# -----------------------------
stu_df, stu_meta = pyreadstat.read_sas7bdat(student_path, usecols=student_vars)
sch_df, sch_meta = pyreadstat.read_sas7bdat(school_path, usecols=school_vars)

print("Original student:", stu_df.shape)
print("Original school :", sch_df.shape)

# -----------------------------
# FILTER SGP
# -----------------------------
stu_df = stu_df[stu_df["CNT"] == "SGP"]
sch_df = sch_df[sch_df["CNT"] == "SGP"]

print("Filtered student (SGP):", stu_df.shape)
print("Filtered school  (SGP):", sch_df.shape)

# -----------------------------
# MERGE STUDENT + SCHOOL
# -----------------------------
merged = stu_df.merge(sch_df, on=["CNT","CNTSCHID"], how="left")
print("Merged shape:", merged.shape)

# -----------------------------
# GENDER + STANDARDIZATION
# -----------------------------
merged["female"] = (merged["ST004D01T"] == 1).astype(int)

for var in ["ESCS", "ST001D01T", "SCHSIZE"]:
    mean_val = merged[var].mean()
    std_val  = merged[var].std()
    merged[var + "_z"] = (merged[var] - mean_val) / std_val
    print(f"Standardized {var} -> {var}_z (mean≈{mean_val:.3f}, sd≈{std_val:.3f})")

# -----------------------------
# CORE MODEL VARIABLES
# -----------------------------
model_vars = [
    "ST268Q01JA","ST268Q04JA","ST268Q07JA",
    "SC064Q01TA","SC064Q02TA","SC064Q03TA",
    "SC064Q04NA","SC064Q05WA","SC064Q06WA",
    "PV1MATH","PV2MATH","PV3MATH","PV4MATH","PV5MATH",
    "PV6MATH","PV7MATH","PV8MATH","PV9MATH","PV10MATH",
    "ST004D01T","MISCED","ESCS","ST001D01T","SCHSIZE"
]

df_core_cols = model_vars + ["female", "ESCS_z", "ST001D01T_z", "SCHSIZE_z"] \
    if "ST001D01T_z" in merged.columns else model_vars + ["female", "ESCS_z", "ST001D01T", "SCHSIZE_z"]

# -----------------------------
# ORIGINAL df_core (for SEM)
# -----------------------------
df_core = merged[df_core_cols].copy()

out_csv_core = os.path.join(OUTPUT_DIR, "df_core.csv")
df_core.to_csv(out_csv_core, index=False)

print("\n✔ Saved df_core to:", out_csv_core)
print("df_core shape:", df_core.shape)

# =====================================================================
#  PART 2: CREATE FULL + SHORT SEMANTIC NAMES FROM METADATA
# =====================================================================

# --- Extract SAS variable mapping ---
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

# Combine mappings
mapping = pd.concat([stu_map, sch_map], ignore_index=True)
mapping = mapping.drop_duplicates(subset=["code"])
mapping = mapping[mapping["full_name"].notnull()]

# Save mapping
mapping_path = os.path.join(OUTPUT_DIR, "pisa_variable_mapping.csv")
mapping.to_csv(mapping_path, index=False)
print("✔ Saved variable → question mapping:", mapping_path)

# Create dictionary: short_name → full SAS label
fullname_lookup = dict(zip(mapping["code"], mapping["full_name"]))

# Latent variable names (these will be added later in R, not here)
LATENT_NAMES = [
    "Latent Factor: Student Math Performance (SMP)",
    "Latent Factor: Student Math self-efficacy (SMS)",
    "Latent Factor: School-level Parental Involvement (SPI)",
]

# =====================================================================
#  BUILD SEMANTIC SAFE COLUMN NAMES (unique + short + meaningful)
# =====================================================================

def make_safe_short_name(full):
    """
    Convert SAS question text into a short semantic name.
    Ensures uniqueness and removes special characters.
    """
    # 1. Keep only part before first ":" (reduces very long questions)
    if ":" in full:
        base = full.split(":", 1)[0].strip()
    else:
        base = full.strip()

    # 2. Clean it
    base = (
        base.replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace(",", "")
            .replace("’", "")
            .replace("'", "")
    )

    # If empty, fallback to generic
    if base == "":
        base = "var"

    return base


# Build new unique names
used = {}
new_names = {}
mapping_rows = []

for col in df_core.columns:

    # Derived columns (female, z-scores)
    if col not in fullname_lookup:
        if col == "female":
            base = "Gender_female_binary"
        elif col.endswith("_z"):
            base = col.replace("_z", "_standardized")
        else:
            base = col
    else:
        base = make_safe_short_name(fullname_lookup[col])

    # Ensure uniqueness
    if base not in used:
        used[base] = 1
        short = base
    else:
        used[base] += 1
        short = f"{base}_{used[base]}"

    new_names[col] = short
    mapping_rows.append({"old": col, "short": short, "full": fullname_lookup.get(col, col)})

# Save short-name → full-name mapping
short_map_df = pd.DataFrame(mapping_rows)
short_map_path = os.path.join(OUTPUT_DIR, "pisa_shortname_mapping.csv")
short_map_df.to_csv(short_map_path, index=False)

print("✔ Saved shortname mapping:", short_map_path)

# Apply renaming
df_core_short = df_core.rename(columns=new_names)

# Save semantic-short CSV
out_csv_full = os.path.join(OUTPUT_DIR, "df_core_fullnames.csv")
df_core_short.to_csv(out_csv_full, index=False)

print("✔ Saved df_core_fullnames (short, unique semantic names) to:", out_csv_full)
print("df_core_fullnames shape:", df_core_short.shape)

print("\n🎉 Step 1 COMPLETE — produced:")
print("  - df_core.csv                    (SEM-ready with original names)")
print("  - df_core_fullnames.csv          (GReaT-ready with short unique names)")
print("  - pisa_variable_mapping.csv      (SAS variable→full question mapping)")
print("  - pisa_shortname_mapping.csv     (short→full semantic mapping)")
