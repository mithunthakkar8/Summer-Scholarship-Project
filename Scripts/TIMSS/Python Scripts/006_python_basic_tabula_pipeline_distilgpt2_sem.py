import os
import pandas as pd
import numpy as np
import torch
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

    PROJECT_ROOT = "/nesi/project/vuw04485/TIMSS"
    HF_CACHE = "/nesi/project/vuw04485/hf_cache"

elif os.path.isdir("/nfs/home"):
    SYSTEM = "RAAPOI"

    PROJECT_ROOT = "/nfs/scratch/thakkamith/TIMSS/data"
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
INPUT_CSV = os.path.join(DATA_DIR, "df_coreSGP.csv")

EXPERIMENT_DIR = os.path.join(DATA_DIR, "tabula_distilgpt2")

# -------------------------------------------------
# Load data
# -------------------------------------------------
df = pd.read_csv(INPUT_CSV)
print("Loaded data shape:", df.shape)

# -------------------------------------------------
# Categorical columns
# -------------------------------------------------
categorical_columns = [
    "BTBG06I", "BTBG06J", "BTBG06K",
    "BTBS17A", "BTBS17B", "BTBS17C", "BTBS17D",
    "BTDGSOS", "BTBG13A", "BTBG13B", "BTBG13C", "BTBG13D",
    "BTBG13E", "BTBG13F", "BTBG13G", "BTBG13H", "BTBG13I"
]

# -------------------------------------------------
# Import Tabula
# -------------------------------------------------
from tabula import Tabula

# -------------------------------------------------
# Initialise model
# -------------------------------------------------
model = Tabula(
    llm="distilgpt2",
    experiment_dir=EXPERIMENT_DIR,
    batch_size=4, epochs=120, categorical_columns = categorical_columns,
    gradient_accumulation_steps = 2
)



# -------------------------------------------------
# Train (MANDATORY)
# -------------------------------------------------

model.fit(df)


# -------------------------------------------------
# Sample immediately (MANDATORY)
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model.model.to(device)
model.model.eval()

N_REPS = 5
N_SAMPLES = len(df)

for i in range(1, N_REPS + 1):
    synthetic_df = model.sample(
        n_samples=N_SAMPLES,
        max_length=400,
    )

    out_path = os.path.join(
        DATA_DIR,
        f"synthetic_tabula_distilgpt2_700EPochs_1Batch8accum_rep_{i}.csv",
    )

    synthetic_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
