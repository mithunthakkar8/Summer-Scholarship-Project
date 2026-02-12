import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import random
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
    DATA_DIR = PROJECT_ROOT

elif os.path.isdir("/nfs/home"):
    SYSTEM = "RAAPOI"

    SCRATCH_ROOT = "/nfs/scratch/thakkamith"
    DATA_DIR = f"{SCRATCH_ROOT}/PISA/data"
    MODEL_ROOT = f"{SCRATCH_ROOT}/great_models"
    HF_CACHE = f"{SCRATCH_ROOT}/hf_cache"

else:
    raise RuntimeError(f"Unknown HPC environment (filesystem check failed): {HOST}")

print(f"Detected system: {SYSTEM}")

# =================================================
# CREATE REQUIRED DIRECTORIES
# =================================================
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_ROOT, exist_ok=True)
os.makedirs(HF_CACHE, exist_ok=True)


SEED = 1029
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE


# -------------------------------------------------
# Compatibility fix (required for newer NumPy)
# -------------------------------------------------
if not hasattr(np, "float"):
    np.float = float


from be_great import GReaT

# -------------------------------------------------
# Paths
# -------------------------------------------------

INPUT_CSV = os.path.join(DATA_DIR, "df_coreSGP.csv")

MODEL_FILE = "great_model_basic"
MODEL_PATH = os.path.join(MODEL_ROOT, MODEL_FILE)
# -------------------------------------------------
# Load data
# -------------------------------------------------
df = pd.read_csv(INPUT_CSV)

print("Loaded data shape:", df.shape)
print(df.head())

# -------------------------------------------------
# LOAD OR TRAIN MODEL
# -------------------------------------------------
if os.path.isdir(MODEL_PATH):
    print(f"Found existing model at {MODEL_PATH}")
    print("Loading model instead of training...")

    model = GReaT.load(MODEL_PATH)

else:
    print("No existing model found.")
    print("Training a new GReaT model...")

    model = GReaT(
        llm="gpt2-large",
        epochs=200,
        batch_size=16,
        save_steps=2000,
        bf16=True
    )

    model.fit(df)
    print("Training completed.")

    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

# -------------------------------------------------
# Move model to GPU if available
# -------------------------------------------------
if torch.cuda.is_available():
    model.model.to("cuda")
    print("Using GPU for generation")
else:
    model.model.to("cpu")
    print("Using CPU for generation")

model.model.eval()
# -------------------------------------------------
# GENERATE MULTIPLE SYNTHETIC UNIVERSES
# -------------------------------------------------
N_REPS = 5
N_SAMPLES = len(df)

for i in range(1, N_REPS + 1):
    synthetic_df = model.sample(
        n_samples=N_SAMPLES,
        k=30,
        max_length=400
    )

    out_path = os.path.join(
        DATA_DIR,
        f"synthetic_great_gpt2large_rep_{i}.csv"
    )

    synthetic_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    print(f"Synthetic data saved to: {out_path}")

