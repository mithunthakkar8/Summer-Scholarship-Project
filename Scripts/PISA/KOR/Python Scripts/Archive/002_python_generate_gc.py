# python_step2_generate_gc.py

import os
import numpy as np
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data/"
INPUT_CSV = os.path.join(DATA_DIR, "df_core.csv")

print("=== Step 2: GaussianCopula Synthetic Data ===")
print("Loading df_core from:", INPUT_CSV)

df_core = pd.read_csv(INPUT_CSV)
print("df_core shape:", df_core.shape)

# Detect metadata
metadata = Metadata.detect_from_dataframe(df_core)
metadata_json_path = os.path.join(DATA_DIR, "pisa2022_metadata.json")
metadata.save_to_json(metadata_json_path)
print("Saved SDV metadata JSON to:", metadata_json_path)

gc = GaussianCopulaSynthesizer(
    metadata,
    enforce_min_max_values=True
)
gc.fit(df_core)

synth = gc.sample(num_rows=len(df_core)).copy()

def round_to_int(s, lo=None, hi=None):
    x = s.round().astype("Int64")
    if lo is not None:
        x = x.clip(lo, hi)
    return x.astype(float)

ordinal_cols = [
    "ST268Q01JA","ST268Q04JA","ST268Q07JA",
    "SC064Q01TA","SC064Q02TA","SC064Q03TA",
    "SC064Q04NA","SC064Q05WA","SC064Q06WA"
]

for col in ordinal_cols:
    if col in synth:
        synth[col] = round_to_int(synth[col], 1, 5)

if "MISCED" in synth:
    synth["MISCED"] = round_to_int(synth["MISCED"], 0, 5)
if "female" in synth:
    synth["female"] = round_to_int(synth["female"], 0, 1)

for v in ["ESCS", "ST001D01T", "SCHSIZE"]:
    if v in synth:
        synth[v + "_z"] = (synth[v] - synth[v].mean()) / synth[v].std()

print("Synthetic GC shape:", synth.shape)

dropped_cols = set(df_core.columns) - set(synth.columns)
if dropped_cols:
    print("⚠ Dropped columns in synthetic GC:", dropped_cols)

out_csv     = os.path.join(DATA_DIR, "synthetic_gc.csv")

synth.to_csv(out_csv, index=False)

print("\n✅ Saved GaussianCopula synthetic data to:")
print("  -", out_csv)
