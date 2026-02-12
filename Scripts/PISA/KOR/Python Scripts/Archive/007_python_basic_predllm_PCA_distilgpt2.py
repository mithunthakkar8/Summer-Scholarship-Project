import os
import pandas as pd
import numpy as np
import torch


print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: running on CPU")


# -------------------------------------------------
# NumPy compatibility
# -------------------------------------------------
if not hasattr(np, "float"):
    np.float = float

# -------------------------------------------------
# Paths
# -------------------------------------------------
DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data"

# DATA_DIR = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Datasets\PISA 2022"
INPUT_CSV = os.path.join(DATA_DIR, "df_coreSGP.csv")
OUTPUT_PREFIX = "synthetic_predllm_PCA_rep"


# -------------------------------------------------
# Load data
# -------------------------------------------------
df = pd.read_csv(INPUT_CSV)
print("Loaded data shape:", df.shape)

# -------------------------------------------------
# PCA      
# -------------------------------------------------

PCA_VARS = [
    "ST268Q01JA", "ST268Q04JA", "ST268Q07JA",
    "SC064Q01TA", "SC064Q02TA", "SC064Q03TA",
    "SC064Q04NA", "SC064Q05WA", "SC064Q06WA", "SC064Q07WA",
    "PV1MATH", "PV2MATH", "PV3MATH", "PV4MATH", "PV5MATH",
    "PV6MATH", "PV7MATH", "PV8MATH", "PV9MATH", "PV10MATH"
]


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# df = your dataframe

# -------------------------------------------------
# Handle missing values BEFORE PCA (MANDATORY)
# -------------------------------------------------
df_pca = df.dropna(subset=PCA_VARS).copy()

scaler = StandardScaler()
X_meas = scaler.fit_transform(df_pca[PCA_VARS])

pca = PCA(n_components=1, random_state=42)
df_pca["PC1_MEASUREMENT"] = pca.fit_transform(X_meas).flatten()

# -------------------------------------------------
# Merge PC1 back into full dataframe (index-safe)
# -------------------------------------------------
df = df.merge(
    df_pca[["PC1_MEASUREMENT"]],
    left_index=True,
    right_index=True,
    how="left"
)


explained_var = pca.explained_variance_ratio_[0]
print(f"PC1 explained variance: {explained_var:.3f}")

assert explained_var > 0.15, "PC1 too weak to act as generative anchor"


# -------------------------------------------------
# SELECT TARGET COLUMN (MANDATORY FOR PredLLM)
# -------------------------------------------------
TARGET_COL = "PC1_MEASUREMENT"   # <-- CHANGE IF NEEDED

assert TARGET_COL in df.columns, "Target column not found"

# -------------------------------------------------
# Keep numeric columns only (PredLLM requirement)
# -------------------------------------------------
# -------------------------------------------------
# Explicitly select columns for PredLLM
# -------------------------------------------------
CONTROL_VARS = [
    "ST001D01T", "ST004D01T", "AGE", "IMMIG",
    "MISCED", "ESCS", "SCHSIZE", "MCLSIZE"
]

MODEL_COLS = PCA_VARS + CONTROL_VARS + ["PC1_MEASUREMENT"]
df = df[MODEL_COLS]


# -------------------------------------------------
# Move target to last column (PredLLM convention)
# -------------------------------------------------
feature_cols = [c for c in df.columns if c != TARGET_COL]
df = df[feature_cols + [TARGET_COL]]

# -------------------------------------------------
# NORMALIZE ALL NUMERIC COLUMNS (INCLUDING TARGET)
# -------------------------------------------------
from sklearn.preprocessing import StandardScaler

scaler_predllm = StandardScaler()

df_scaled = pd.DataFrame(
    scaler_predllm.fit_transform(df),
    columns=df.columns,
    index=df.index
)

# Optional but recommended: reduce numeric precision
df_scaled = df_scaled.round(3)

# Replace df used for training
df = df_scaled

print("Data normalized for PredLLM training (features + target)")


# -------------------------------------------------
# Import PredLLM
# -------------------------------------------------
from tabular_llm.predllm import PredLLM
from tabular_llm.predllm_utils import _encode_row_partial

# -------------------------------------------------
# Initialise PredLLM
# -------------------------------------------------
predllm = PredLLM(
    llm="distilgpt2",
    batch_size=32,
    epochs=50,
)

# -------------------------------------------------
# Train (MANDATORY)
# -------------------------------------------------
predllm.fit(df)


# -------------------------------------------------
# # Check for GPU
# ------------------------------------------------- 
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# # -------------------------------------------------
# # Compute prompt length (IMPORTANT)
# # -------------------------------------------------
encoded_text = _encode_row_partial(df.iloc[0], shuffle=False)
prompt_len = len(predllm.tokenizer(encoded_text)["input_ids"])
print("Prompt length:", prompt_len)

# -------------------------------------------------
# Sample synthetic data
# -------------------------------------------------
N_REPS = 5
N_SAMPLES = len(df)

for i in range(1, N_REPS + 1):
    synthetic_df = predllm.sample_new(
        n_samples=N_SAMPLES,
        max_length=prompt_len,
        task="regression",
    )

    # -------------------------------------------------
    # RESTORE COLUMN NAMES (PredLLM outputs integer cols)
    # -------------------------------------------------
    print(f"[INFO] Starting inverse scaling for rep {i}", flush=True)

    synthetic_df.columns = df.columns

    print(f"[INFO] Inverse transform for rep {i}", flush=True)
    syn_features = synthetic_df.drop(columns=[TARGET_COL])
    syn_features_inv = scaler_predllm.inverse_transform(syn_features)
    syn_target = synthetic_df[TARGET_COL].values

    # Rebuild dataframe
    synthetic_df = pd.DataFrame(
        syn_features_inv,
        columns=syn_features.columns
    )

    # Reattach target
    synthetic_df[TARGET_COL] = syn_target


    out_path = os.path.join(
        DATA_DIR,
        f"{OUTPUT_PREFIX}_{i}.csv",
    )

    synthetic_df.to_csv(out_path, index=False, compression="gzip")
    print(f"Saved: {out_path}")
