import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import socket

HOST = socket.gethostname().lower()

print(f"Running on host: {HOST}")


# =================================================
# GPU / CUDA diagnostics (HPC-safe)
# =================================================
def log_gpu_info():
    print("===== GPU / CUDA INFO =====")

    # Slurm-assigned GPUs (authoritative)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

    # PyTorch view
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  - Total memory: {props.total_memory / 1e9:.2f} GB")
    else:
        print("No GPU detected by PyTorch")

    print("===========================")


log_gpu_info()

# =================================================
# HPC-specific paths
# =================================================

if os.path.isdir("/nesi/project"):
    SYSTEM = "NESI"

    PROJECT_ROOT = "/nesi/project/vuw04485/PISA"
    HF_CACHE = "/nesi/project/vuw04485/hf_cache"

elif os.path.isdir("/nfs/home"):
    SYSTEM = "RAAPOI"

    PROJECT_ROOT = "/nfs/scratch/thakkamith/PISA/data"
    HF_CACHE = "/nfs/scratch/thakkamith/hf_cache"

else:
    raise RuntimeError(f"Unknown HPC environment (filesystem check failed): {HOST}")

print(f"Detected system: {SYSTEM}")



# -------------------------------------------------
# NumPy compatibility
# -------------------------------------------------
if not hasattr(np, "float"):
    np.float = float

# -------------------------------------------------
# Paths
# -------------------------------------------------
DATA_DIR = PROJECT_ROOT
INPUT_CSV = os.path.join(DATA_DIR, "df_core_with_smp_latent_SGP.csv")
OUTPUT_PREFIX = "synthetic_predllm_SGP_rep"

# -------------------------------------------------
# Load data
# -------------------------------------------------
df = pd.read_csv(INPUT_CSV)
print("Loaded data shape:", df.shape)

# -------------------------------------------------
# SELECT TARGET COLUMN (MANDATORY FOR PredLLM)
# -------------------------------------------------
TARGET_COL = "LV_SMP"   # <-- CHANGE IF NEEDED

assert TARGET_COL in df.columns, "Target column not found"

# -------------------------------------------------
# Keep numeric columns only (PredLLM requirement)
# -------------------------------------------------
df = df.select_dtypes(include=[np.number])
print("Numeric-only shape:", df.shape)



# -------------------------------------------------
# Move target to last column (PredLLM convention)
# -------------------------------------------------
feature_cols = [c for c in df.columns if c != TARGET_COL]


# -------------------------------------------------
# Standardize numeric data using sklearn
# -------------------------------------------------
scaler = StandardScaler()



df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns,
    index=df.index
)

df_scaled = df_scaled[feature_cols + [TARGET_COL]]

print("Post-standardization check:")
print(df_scaled.mean().round(3))
print(df_scaled.std(ddof=0).round(3))


# -------------------------------------------------
# Reduce numeric precision to shorten token sequences
# -------------------------------------------------
df_scaled = df_scaled.round(3)


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
    batch_size=8,
    epochs=350,
)

# -------------------------------------------------
# Train (MANDATORY)
# -------------------------------------------------
from pathlib import Path

PREDLLM_MODEL_DIR = os.path.join(DATA_DIR, "predllm_distilgpt2_SGP_")

model_path = Path(PREDLLM_MODEL_DIR)

print("[INFO] Training PredLLM from scratch", flush=True)
predllm.fit(df_scaled)



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
encoded_text = _encode_row_partial(df_scaled.iloc[0], shuffle=False)
prompt_len = len(predllm.tokenizer(encoded_text)["input_ids"])
print("Prompt length:", prompt_len)

# -------------------------------------------------
# Sample synthetic data
# -------------------------------------------------
N_REPS = 5
N_SAMPLES = int(round(len(df)/0.78,0))
GEN_EXTRA_TOKENS = 100  # safe margin

for i in range(1, N_REPS + 1):
    torch.manual_seed(i)
    np.random.seed(i)

    synthetic_scaled = predllm.sample_new(
    n_samples=N_SAMPLES,
    max_length=prompt_len+GEN_EXTRA_TOKENS,
    task="regression",
    )

    # Ensure column order matches training data
    synthetic_scaled.columns = df_scaled.columns
    

    out_path = os.path.join(
        DATA_DIR,
        f"{OUTPUT_PREFIX}_350_8_{i}.csv",
    )

    # Inverse transform back to original scale
    synthetic_df = pd.DataFrame(
        scaler.inverse_transform(synthetic_scaled),
        columns=synthetic_scaled.columns
    )

    synthetic_df.to_csv(out_path, index=False)
