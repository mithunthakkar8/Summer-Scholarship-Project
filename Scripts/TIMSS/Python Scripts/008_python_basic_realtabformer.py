import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import torch
assert torch.cuda.is_available(), "CUDA not available – aborting run"
import numpy as np
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])


# -------------------------------------------------
# NumPy compatibility
# -------------------------------------------------
if not hasattr(np, "float"):
    np.float = float

# -------------------------------------------------
# Paths
# -------------------------------------------------
DATA_DIR = "/nesi/project/vuw04485/TIMSS"
INPUT_CSV = os.path.join(DATA_DIR, "df_coreSGP.csv")
OUTPUT_PREFIX = "synthetic_realtabformer_SGP_rep"

MODEL_ROOT = os.path.join(DATA_DIR, "realtabformer_tabular")

# -------------------------------------------------
# Load data
# -------------------------------------------------
df = pd.read_csv(INPUT_CSV)
print("Loaded data shape:", df.shape)

# -------------------------------------------------
# OPTIONAL: keep numeric only (recommended but not required)
# REaLTabFormer supports categorical columns as well
# -------------------------------------------------
df = df.select_dtypes(include=[np.number])
print("Numeric-only shape:", df.shape)

# -------------------------------------------------
# Standardize numeric data
# -------------------------------------------------
scaler = StandardScaler()

df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns,
    index=df.index
)

print("Post-standardization check:")
print(df_scaled.mean().round(3))
print(df_scaled.std(ddof=0).round(3))

# Optional precision reduction (helps token length)
df_scaled = df_scaled.round(3)

# -------------------------------------------------
# Import REaLTabFormer
# -------------------------------------------------
from realtabformer import REaLTabFormer

# -------------------------------------------------
# Initialise REaLTabFormer (tabular / parent model)
# -------------------------------------------------
rtf = REaLTabFormer(
    model_type="tabular",

    # ---------- HARD SAFETY CAPS ----------
    epochs=150,                  # upper bound, not target
    early_stopping_patience=3,   # allows longer learning on large data

    # ---------- BATCHING ----------
    batch_size=16,               # safe default (CPU/GPU)
    gradient_accumulation_steps=1,  # increase only if GPU OOM

    # ---------- REGULARISATION / CRITIC ----------
    mask_rate=0.08,   
                                  # scales well with data size

    # ---------- TOKEN LENGTH CONTROL ----------
    numeric_precision=2,
    numeric_max_len=8,

    # ---------- REPRODUCIBILITY ----------
    random_state=1029,
)



# -------------------------------------------------
# Train or load model
# -------------------------------------------------
model_root = Path(MODEL_ROOT)
model_root.mkdir(parents=True, exist_ok=True)

existing_models = sorted(
    [p for p in model_root.glob("id*") if p.is_dir()],
    key=os.path.getmtime
)

if existing_models:
    model_path = existing_models[-1]
    print(f"[INFO] Loading existing REaLTabFormer model from {model_path}", flush=True)
    rtf = REaLTabFormer.load_from_dir(model_path)
else:
    print("[INFO] Training REaLTabFormer from scratch", flush=True)
    rtf.fit(df_scaled)

    print("[INFO] Saving REaLTabFormer model", flush=True)
    rtf.save(model_root)

# -------------------------------------------------
# GPU check (informational only)
# -------------------------------------------------
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# -------------------------------------------------
# Sample synthetic data
# -------------------------------------------------
N_REPS = 5
N_SAMPLES = len(df)

for i in range(1, N_REPS + 1):

    synthetic_scaled = rtf.sample(n_samples=N_SAMPLES)

    # Ensure column order
    synthetic_scaled = synthetic_scaled[df_scaled.columns]

    # Inverse transform back to original scale
    synthetic_df = pd.DataFrame(
        scaler.inverse_transform(synthetic_scaled),
        columns=synthetic_scaled.columns
    )

    out_path = os.path.join(
        DATA_DIR,
        f"{OUTPUT_PREFIX}_{i}.csv"
    )

    synthetic_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved {out_path}", flush=True)
