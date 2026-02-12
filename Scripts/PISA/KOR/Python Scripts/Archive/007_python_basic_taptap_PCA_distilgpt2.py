import os
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------------------------------
# Diagnostics
# -------------------------------------------------
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))

if not hasattr(np, "float"):
    np.float = float

# -------------------------------------------------
# Paths
# -------------------------------------------------
DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data"
INPUT_CSV = os.path.join(DATA_DIR, "df_coreSGP.csv")
OUTPUT_PREFIX = "synthetic_taptap_LV_rep"

# -------------------------------------------------
# Load data
# -------------------------------------------------
df = pd.read_csv(INPUT_CSV)
print("Loaded data shape:", df.shape)

# -------------------------------------------------
# PCA construction
# -------------------------------------------------
PCA_VARS = [
    "ST268Q01JA", "ST268Q04JA", "ST268Q07JA",
    "SC064Q01TA", "SC064Q02TA", "SC064Q03TA",
    "SC064Q04NA", "SC064Q05WA", "SC064Q06WA", "SC064Q07WA",
    "PV1MATH", "PV2MATH", "PV3MATH", "PV4MATH", "PV5MATH",
    "PV6MATH", "PV7MATH", "PV8MATH", "PV9MATH", "PV10MATH"
]

CONTROL_VARS = [
    "ST001D01T", "ST004D01T", "AGE", "IMMIG",
    "MISCED", "ESCS", "SCHSIZE", "MCLSIZE"
]

# PCA must be computed only on complete rows
df_pca = df.dropna(subset=PCA_VARS).copy()

scaler = StandardScaler()
X_meas = scaler.fit_transform(df_pca[PCA_VARS])

pca = PCA(n_components=1, random_state=42)
df_pca["PC1_MEASUREMENT"] = pca.fit_transform(X_meas).flatten()

df = df.merge(
    df_pca[["PC1_MEASUREMENT"]],
    left_index=True,
    right_index=True,
    how="left"
)

explained_var = pca.explained_variance_ratio_[0]
print(f"PC1 explained variance: {explained_var:.3f}")
assert explained_var > 0.15, "PC1 too weak for anchoring"

# -------------------------------------------------
# Modeling dataframe
# -------------------------------------------------
MODEL_COLS = PCA_VARS + CONTROL_VARS + ["PC1_MEASUREMENT"]
df_model = df[MODEL_COLS].copy()

# IMPORTANT: TapTap tolerates NaNs, but they increase parsing noise
# You may optionally restrict to rows with observed PC1
df_model = df_model.dropna(subset=["PC1_MEASUREMENT"]).reset_index(drop=True)

# -------------------------------------------------
# Initialise TapTap
# -------------------------------------------------
from taptap import Taptap   # adjust import path if needed

taptap = Taptap(
    llm="distilgpt2",
    experiment_dir="taptap_pca_run",
    steps=2000,
    batch_size=8,
    numerical_modeling="split",   # recommended by TapTap for numbers
    max_tokens=400,
)

# -------------------------------------------------
# Fit (fine-tune)
# -------------------------------------------------
# NOTE:
# - target_col is only used to define start-token distribution
# - this is NOT regression conditioning
trainer = taptap.fit(
    data=df_model,
    target_col="PC1_MEASUREMENT",
    task="regression",
    conditional_col="PC1_MEASUREMENT",
)

# -------------------------------------------------
# Sample synthetic data
# -------------------------------------------------
N_REPS = 5
N_SAMPLES = len(df_model)

for i in range(1, N_REPS + 1):
    synthetic_df = taptap.sample(
        n_samples=N_SAMPLES,
        data=df_model,
        task="regression",
        temperature=0.7,
        k=100,
        max_length=400,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    out_path = os.path.join(DATA_DIR, f"{OUTPUT_PREFIX}_{i}.csv")
    synthetic_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
