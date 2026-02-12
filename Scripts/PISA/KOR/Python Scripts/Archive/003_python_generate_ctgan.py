# python_step3_generate_ctgan.py

import os
import random
import numpy as np
import pandas as pd
import torch

from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data/"
INPUT_CSV = os.path.join(DATA_DIR, "df_core.csv")

print("=== Step 3: CTGAN multi-run synthetic data ===")
df_core = pd.read_csv(INPUT_CSV)
print("df_core shape:", df_core.shape)

metadata = Metadata.detect_from_dataframe(df_core)

seeds = [42, 101, 202, 303, 404]

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

for i, seed in enumerate(seeds):
    print(f"\n--- CTGAN run {i+1}/{len(seeds)} with seed={seed} ---")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ctgan = CTGANSynthesizer(
        metadata,
        enforce_min_max_values=True,
        epochs=300,
        batch_size=128,
        pac=1
    )
    ctgan.fit(df_core)

    synth_ctgan = ctgan.sample(num_rows=len(df_core)).copy()
    synth_ctgan = synth_ctgan.fillna(synth_ctgan.median(numeric_only=True))

    # Zero variance fix (rare but safer)
    for col in ["SC064Q02TA"]:
        if col in synth_ctgan and synth_ctgan[col].nunique() <= 1:
            real_vals = df_core[col].dropna()
            if len(real_vals) > 0:
                synth_ctgan[col] = np.random.choice(real_vals, size=len(synth_ctgan))
            else:
                synth_ctgan[col] = np.random.randint(1, 6, size=len(synth_ctgan))

    # Low variance "jitter"
    for col in ["SC064Q02TA", "SC064Q06WA"]:
        if col in synth_ctgan and synth_ctgan[col].std() < 0.1:
            synth_ctgan[col] += np.random.uniform(-0.3, 0.3, size=len(synth_ctgan[col]))
            synth_ctgan[col] = synth_ctgan[col].clip(1, 5)

    for col in ordinal_cols:
        if col in synth_ctgan:
            synth_ctgan[col] = (synth_ctgan[col].round().clip(1, 5)).astype(float)

    for v in ["ESCS", "ST001D01T", "SCHSIZE"]:
        if v in synth_ctgan:
            synth_ctgan[v + "_z"] = (synth_ctgan[v] - synth_ctgan[v].mean()) / synth_ctgan[v].std()

    out_csv     = os.path.join(DATA_DIR, f"synthetic_ctgan_seed{seed}.csv")

    synth_ctgan.to_csv(out_csv, index=False)

    print("Saved CTGAN synthetic to:")
    print("  -", out_csv)

print("\n✅ All CTGAN runs complete.")
