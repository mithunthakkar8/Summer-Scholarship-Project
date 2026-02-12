# python_step2_generate_gc.py

import os
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

# DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data/"
DATA_DIR = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Datasets\PISA 2022"
CNT = "SGP"
INPUT_CSV = os.path.join(DATA_DIR, f"df_core{CNT}.csv")

print("=== Step 2: GaussianCopula Synthetic Data ===")
print("Loading df_core from:", INPUT_CSV)

df_core = pd.read_csv(INPUT_CSV)
print("df_core shape:", df_core.shape)

id_cols = ["CNTSCHID", "CNTSTUID"]
df_core = df_core.drop(columns=[c for c in id_cols if c in df_core.columns])

# Drop constant / non-informative columns
if "CNT" in df_core.columns:
    df_core = df_core.drop(columns=["CNT"])



# Detect metadata
metadata = Metadata.detect_from_dataframe(df_core)

# # Gender is categorical (1 = female, 2 = male in PISA)
# if "ST004D01T" in df_core.columns:
#     metadata.update_column(
#         column_name="ST004D01T",
#         sdtype="categorical"
#     )

# # Ordinal Likert items (1–5 or 1–4 depending on item)
# ordinal_cols = [
#     "ST268Q01JA","ST268Q04JA","ST268Q07JA",
#     "SC064Q01TA","SC064Q02TA","SC064Q03TA",
#     "SC064Q04NA","SC064Q05WA","SC064Q06WA","SC064Q07WA"
# ]

# for col in ordinal_cols:
#     if col in df_core.columns:
#         metadata.update_column(
#             column_name=col,
#             sdtype="numerical"
#         )

# # Education level
# if "MISCED" in df_core.columns:
#     metadata.update_column(
#         column_name="MISCED",
#         sdtype="numerical"
#     )


metadata_json_path = os.path.join(DATA_DIR, "pisa2022_metadata.json")
metadata.save_to_json(metadata_json_path)
print("Saved SDV metadata JSON to:", metadata_json_path)

gc = GaussianCopulaSynthesizer(
    metadata,
    enforce_min_max_values=True
)
gc.fit(df_core)

synth = gc.sample(num_rows=len(df_core)).copy()

# -----------------------------
# Enforce discrete categorical variables
# -----------------------------

# # Gender (1–2)
# if "ST004D01T" in synth.columns:
#     synth["ST004D01T"] = (
#         synth["ST004D01T"]
#         .round()
#         .clip(1, 2)
#         .astype(int)
#     )

# # Immigration status (assumed binary 0/1 or 1/2 depending on coding)
# if "IMMIG" in synth.columns:
#     lo, hi = df_core["IMMIG"].min(), df_core["IMMIG"].max()
#     synth["IMMIG"] = (
#         synth["IMMIG"]
#         .round()
#         .clip(lo, hi)
#         .astype(int)
#     )

# # Grade (integer)
# if "ST001D01T" in synth.columns:
#     synth["ST001D01T"] = (
#         synth["ST001D01T"]
#         .round()
#         .astype(int)
#     )


# -----------------------------
# Match decimal precision
# -----------------------------
# DECIMAL_RULES = {
#     "AGE": 2,
#     "ESCS": 4,
#     "PV1MATH": 3, "PV2MATH": 3, "PV3MATH": 3, "PV4MATH": 3,
#     "PV5MATH": 3, "PV6MATH": 3, "PV7MATH": 3, "PV8MATH": 3,
#     "PV9MATH": 3, "PV10MATH": 3,
# }

# for col, d in DECIMAL_RULES.items():
#     if col in synth.columns:
#         synth[col] = synth[col].round(d)


# def round_to_int(s, lo=None, hi=None):
#     x = s.round().astype("Int64")
#     if lo is not None:
#         x = x.clip(lo, hi)
#     return x.astype(float)

# -----------------------------
# Enforce discrete support
# -----------------------------
# for col in ordinal_cols:
#     if col in synth.columns:
#         synth[col] = round_to_int(synth[col], 1, 5)

# Count variables
# for col in ["SCHSIZE", "MCLSIZE"]:
#     if col in synth.columns:
#         synth[col] = synth[col].round().clip(lower=1).astype(int)



# if "MISCED" in synth:
#     synth["MISCED"] = round_to_int(synth["MISCED"], 0, 5)


print("Synthetic GC shape:", synth.shape)

dropped_cols = set(df_core.columns) - set(synth.columns)
if dropped_cols:
    print("⚠ Dropped columns in synthetic GC:", dropped_cols)

out_csv     = os.path.join(DATA_DIR, f"synthetic_gc{CNT}.csv")

synth.to_csv(out_csv, index=False)

print("\n✅ Saved GaussianCopula synthetic data to:")
print("  -", out_csv)
