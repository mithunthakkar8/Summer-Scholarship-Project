import pandas as pd
import numpy as np
import os
import torch

os.environ["HF_HOME"] = "/nesi/project/vuw04485/hf_cache"
os.environ["HF_HUB_CACHE"] = "/nesi/project/vuw04485/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/nesi/project/vuw04485/hf_cache"

# -------------------------------------------------
# Compatibility fix (required for newer NumPy)
# -------------------------------------------------
if not hasattr(np, "float"):
    np.float = float

from be_great import GReaT

# -------------------------------------------------
# Paths
# -------------------------------------------------
DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data"
INPUT_CSV = os.path.join(DATA_DIR, "df_core_with_latent_scores_SGP.csv")

MODEL_FILE = "great_model_basic_distilgpt2_conditional"
MODEL_PATH = os.path.join(DATA_DIR, MODEL_FILE)
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
        llm="distilgpt2",
        epochs=100,
        batch_size=16,
        save_steps=2000
    )

    model.fit(df, conditional_col = 'LV_SMP')
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
# GENERATE MULTIPLE SYNTHETIC UNIVERSES (IMPUTATION)
# -------------------------------------------------
N_REPS = 5

FIXED_COLS = ["LV_SMS", "LV_SPI", "LV_SMP"]

# Sanity check
missing = set(FIXED_COLS) - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

for i in range(1, N_REPS + 1):

    # -------------------------------------------------
    # Create imputation template
    # -------------------------------------------------
    impute_df = df.copy()

    # Set all non-fixed columns to NaN
    for col in impute_df.columns:
        if col not in FIXED_COLS:
            impute_df[col] = np.nan

    # -------------------------------------------------
    # Run conditional imputation
    # -------------------------------------------------
    synthetic_df = model.impute(
        impute_df,
        k=50,
        max_length=400
    )

    # -------------------------------------------------
    # Save output
    # -------------------------------------------------
    out_path = os.path.join(
        DATA_DIR,
        f"synthetic_great_distilgpt2_conditional_rep_{i}.csv"
    )

    synthetic_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
