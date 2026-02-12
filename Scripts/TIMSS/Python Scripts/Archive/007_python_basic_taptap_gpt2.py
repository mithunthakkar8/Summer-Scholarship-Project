import os
import pandas as pd
import torch
from taptap.taptap import Taptap

from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# Paths
# -------------------------------------------------
# DATA_DIR = "/nesi/project/vuw04485/TIMSS"

DATA_DIR = "/nfs/scratch/thakkamith/TIMSS/data"
INPUT_CSV = os.path.join(DATA_DIR, "df_core_with_acm_latent_SGP.csv")
OUTPUT_PREFIX = "synthetic_taptap_ACM_gpt2_rep"

# -------------------------------------------------
# Load RAW data (NO SCALING)
# -------------------------------------------------
df = pd.read_csv(INPUT_CSV)
print("Loaded data shape:", df.shape)

target_col = "LV_ACM"
task = "regression"

# -------------------------------------------------
# Scale ALL numeric columns (TapTap expects numeric-only here)
# -------------------------------------------------
scaler = StandardScaler()

# store column order explicitly (critical for inverse transform)
num_cols = df.columns.tolist()

df_scaled = pd.DataFrame(
    scaler.fit_transform(df[num_cols]),
    columns=num_cols,
    index=df.index,
)

print("Scaling applied (mean≈0, std≈1)")


df_scaled = df_scaled.round(3)


# -------------------------------------------------
# Initialise TapTap
# -------------------------------------------------
from taptap.taptap import Taptap


model = Taptap(
    llm="gpt2",
    experiment_dir="taptap_SGP_run_stable",
    steps=10000,                    # ↑ from 2k → ensures convergence
    batch_size=2,                   # avoid batch=1 instability
    gradient_accumulation_steps=16, # effective batch = 32

    # -----------------------------
    # NUMERICAL HANDLING
    # -----------------------------
    numerical_modeling="split"     
)


# -------------------------------------------------
# Fit (fine-tune)
# -------------------------------------------------
# NOTE:
# - target_col is only used to define start-token distribution
# - this is NOT regression conditioning
model.fit(
    data=df_scaled,
    target_col="LV_ACM",
    task="regression"
)


# -------------------------------------------------
# Sample synthetic data
# -------------------------------------------------
N_REPS = 5
N_SAMPLES = int(round(len(df)/0.85,0))

for i in range(1, N_REPS + 1):

    synthetic_scaled = model.sample(
        n_samples=N_SAMPLES,
        data=df_scaled,
        task="regression",
        temperature=0.7,
        k=100,
        max_length=400,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # ensure correct column order
    synthetic_scaled = synthetic_scaled[num_cols]

    # -----------------------------
    # Inverse transform to original space
    # -----------------------------
    synthetic_original = pd.DataFrame(
        scaler.inverse_transform(synthetic_scaled),
        columns=num_cols,
    )

    out_path = os.path.join(DATA_DIR, f"{OUTPUT_PREFIX}_{i}.csv")
    synthetic_original.to_csv(out_path, index=False)

    print(f"Saved (inverse-scaled): {out_path}")
