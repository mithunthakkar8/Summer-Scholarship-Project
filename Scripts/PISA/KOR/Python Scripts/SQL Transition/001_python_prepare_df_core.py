# python_step1_prepare_df_core.py
import os
import numpy as np
import pandas as pd
import pyreadstat
import re
import psycopg2
from psycopg2.extras import execute_values


# -----------------------------
# POSTGRES CONNECTION
# -----------------------------
conn = psycopg2.connect(
    dbname="PISA_2022",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)
conn.autocommit = False


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
def make_canonical_namename(col: str) -> str:
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
target_cnt = ["JPN", "KOR", "SGP", "TAP", "HKG", "MAC"]
stu_df_cnt = stu_df[stu_df["CNT"].isin(target_cnt)]
sch_df_cnt = sch_df[sch_df["CNT"].isin(target_cnt)]


print("Filtered student:", stu_df_cnt.shape)
print("Filtered school:", sch_df_cnt.shape)

len(np.unique(sch_df_cnt.CNTSCHID))

sch_df_cnt.describe()

sch_df_cnt.groupby("CNT").describe()



# ---- type alignment ----
sch_df_cnt["CNTSCHID"] = sch_df_cnt["CNTSCHID"].astype("Int64")

smallint_cols = [
    "SC064Q01TA", "SC064Q02TA", "SC064Q03TA",
    "SC064Q04NA", "SC064Q05WA", "SC064Q06WA", "SC064Q07WA",
    "MCLSIZE"
]



for col in smallint_cols:
    sch_df_cnt[col] = sch_df_cnt[col].round().astype("Int64")

sch_df_cnt["SCHSIZE"] = sch_df_cnt["SCHSIZE"].round().astype("Int64")

# ---- CRITICAL FIX (THIS WAS MISSING) ----
sch_df_cnt = sch_df_cnt.replace({pd.NA: None})

# ---- convert numpy scalars → python scalars ----
sch_df_cnt = sch_df_cnt.astype(object)

# ---- tuple creation (safe) ----
school_cols = sch_df_cnt.columns.tolist()
school_tuples = list(
    sch_df_cnt.itertuples(index=False, name=None)
)



insert_schools_sql = f"""
INSERT INTO pisa.schools ({",".join(school_cols)})
VALUES %s
ON CONFLICT (CNT, CNTSCHID) DO NOTHING;
"""

with conn.cursor() as cur:
    execute_values(cur, insert_schools_sql, school_tuples)

stu_df_cnt = stu_df_cnt.copy()

int_cols = [
    "CNTSCHID", "CNTSTUID",
    "ST001D01T", "ST004D01T",
    "IMMIG", "MISCED",
    "ST268Q01JA", "ST268Q04JA", "ST268Q07JA"
]

for col in int_cols:
    stu_df_cnt[col] = stu_df_cnt[col].round().astype("Int64")

stu_df_cnt = stu_df_cnt.replace({pd.NA: None})
stu_df_cnt = stu_df_cnt.astype(object)
stu_df_cnt = stu_df_cnt.where(pd.notnull(stu_df_cnt), None)

student_cols = stu_df_cnt.columns.tolist()

student_tuples = list(
    stu_df_cnt.itertuples(index=False, name=None)
)


with conn.cursor() as cur:
    execute_values(
        cur,
        f"""
        INSERT INTO pisa.students ({",".join(student_cols)})
        VALUES %s
        ON CONFLICT (CNT, CNTSCHID, CNTSTUID) DO NOTHING
        """,
        student_tuples,
        page_size=5000
    )

# =====================================================================
# CREATE + LOAD VARIABLE MAPPINGS INTO POSTGRES
# =====================================================================

# ---- build mapping frames ----
stu_map = pd.DataFrame({
    "variable_code": stu_meta.column_names,
    "full_name": stu_meta.column_labels,
    "dataset_level": "student"
})

sch_map = pd.DataFrame({
    "variable_code": sch_meta.column_names,
    "full_name": sch_meta.column_labels,
    "dataset_level": "school"
})

# ---- combine ----
mapping = pd.concat([stu_map, sch_map], ignore_index=True)

# ---- clean ----
mapping = mapping.drop_duplicates(
    subset=["variable_code", "dataset_level"]
)
mapping = mapping[mapping["full_name"].notnull()]

# ---- canonical_name generation ----
mapping["canonical_name"] = mapping["full_name"].apply(make_canonical_namename)

def infer_value_type(code: str) -> str:
    """
    Infer storage/modeling type for PISA variables
    """

    # ---- identifiers ----
    if code in {"CNT", "CNTSCHID", "CNTSTUID"}:
        return "id"

    # ---- plausible values (achievement) ----
    if code.startswith("PV"):
        return "continuous"

    # ---- continuous indices ----
    if code in {"ESCS"}:
        return "continuous"

    # ---- categorical / ordinal student items ----
    if code.startswith(("ST001", "ST004", "ST268")):
        return "categorical"

    # ---- categorical / ordinal school items ----
    if code.startswith("SC064"):
        return "categorical"

    # ---- sizes / counts ----
    if code in {"AGE", "MCLSIZE", "SCHSIZE"}:
        return "continuous"

    # ---- fallback ----
    return "categorical"


mapping["value_type"] = mapping["variable_code"].apply(infer_value_type)

# ---- convert to python-native types ----
mapping = mapping.replace({pd.NA: None})
mapping = mapping.astype(object)

# ---- insert into Postgres ----
mapping_cols = [
    "variable_code",
    "full_name",
    "dataset_level",
    "canonical_name",
    "value_type"
]

mapping_tuples = list(
    mapping[mapping_cols].itertuples(index=False, name=None)
)

insert_mapping_sql = """
    INSERT INTO pisa.variable_mapping (
        variable_code,
        full_name,
        dataset_level,
        canonical_name,
        value_type
    )
    VALUES %s
    ON CONFLICT (variable_code, dataset_level) DO NOTHING
"""

with conn.cursor() as cur:
    execute_values(
        cur,
        insert_mapping_sql,
        mapping_tuples,
        page_size=1000
    )

conn.commit()

print(f"✔ Loaded {len(mapping_tuples)} rows into pisa.variable_mapping")
