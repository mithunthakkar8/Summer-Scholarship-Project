# python_step1_prepare_df_timss_core.py

import os
import pandas as pd
import pyreadstat

# =============================
# PATH CONFIG
# =============================

CNT = 'LTU'

DATA_DIR = r"C:\Users\mithu\Downloads\T23_Data_SAS_G8\SAS Data"
OUTPUT_DIR = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Datasets\TIMSS 2023"

# ---- UPDATE THESE FILE NAMES TO MATCH YOUR TIMSS FILES ----
teacher_path = os.path.join(DATA_DIR, f"bts{CNT.lower()}m8.sas7bdat")

print("=== Loading TIMSS SAS files ===")
print("Teacher:", teacher_path)

# =============================
# VARIABLES FROM PAPER
# =============================

# ---- School Safety (SSF) – 2 items ----
ssf_items = ["BTBGSOS", "BTDGSOS"]

# ---- Teacher–Student Interaction (TCI) – 4 items ----
tci_items = ["BTBS17A", "BTBS17B", "BTBS17C", "BTBS17D"]

# ---- Learning Environment (LEI) – 9 items ----
lei_items = [f"BTBG13{c}" for c in list("ABCDEFGHI")]

# ---- Academic Mindedness (ACM) – 3 items ----
acm_items = ["BTBG06I", "BTBG06J", "BTBG06K"]

# ---- Number of Students in Class (NSC) – single item ----
nsc_item = ["BTBG10"]  



teacher_vars = ssf_items + tci_items + lei_items + acm_items + nsc_item

# =============================
# LOAD DATA
# =============================

tch_df, tch_meta = pyreadstat.read_sas7bdat(
    teacher_path,
    usecols=teacher_vars
)

print("Teacher shape (raw):", tch_df.shape)


# =============================
# COMPLETE-CASE FILTER (AS IN PAPER)
# =============================
before = tch_df.shape[0]
df = tch_df.dropna(subset=teacher_vars, how="any")
after = df.shape[0]

print(f"Removed {before - after} rows with missing teacher indicators")
print("Final teacher dataset:", df.shape)

# =============================
# SAVE CSV
# =============================

out_csv = os.path.join(OUTPUT_DIR, f"df_core{CNT}.csv")
df.to_csv(out_csv, index=False)

print("\n✔ Saved SEM-ready TIMSS dataset:")
print(out_csv)



