import os
import pandas as pd
import numpy as np
import torch

import os

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("RUNNING ON CPU")



# -------------------------------------------------
# NumPy compatibility
# -------------------------------------------------
if not hasattr(np, "float"):
    np.float = float

# -------------------------------------------------
# Paths
# -------------------------------------------------
DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data"
INPUT_CSV = os.path.join(DATA_DIR, "df_coreSGP.csv")
EXPERIMENT_DIR = os.path.join(DATA_DIR, "tabula_gpt2")

# -------------------------------------------------
# Load data
# -------------------------------------------------
df = pd.read_csv(INPUT_CSV)
print("Loaded data shape:", df.shape)

# -------------------------------------------------
# Categorical columns
# -------------------------------------------------
categorical_columns = [
    "ST001D01T",  
    "ST004D01T",
    "ST268Q01JA",
    "ST268Q04JA",
    "ST268Q07JA",
    "IMMIG",
    "MISCED"  
]


# -------------------------------------------------
# Import Tabula
# -------------------------------------------------
from tabula import Tabula

# -------------------------------------------------
# Initialise model
# -------------------------------------------------
model = Tabula(
    llm="gpt2",
    experiment_dir=EXPERIMENT_DIR,
    batch_size=4,
    epochs=50,
    categorical_columns=categorical_columns,
)


# -------------------------------------------------
# (Optional) load pretrained GPT-2 weights
# -------------------------------------------------
# torch.load(...) here is ONLY for initialisation
# model.model.load_state_dict(torch.load("some_pretrained.pt"), strict=False)

from sklearn.preprocessing import StandardScaler

cont_cols = [
    "AGE", "ESCS", "SCHSIZE", "MCLSIZE",
    "PV1MATH","PV2MATH","PV3MATH","PV4MATH","PV5MATH",
    "PV6MATH","PV7MATH","PV8MATH","PV9MATH","PV10MATH"
]

scaler = StandardScaler()
df[cont_cols] = scaler.fit_transform(df[cont_cols])


# -------------------------------------------------
# Train (MANDATORY)
# -------------------------------------------------
resume_ckpt = (
    EXPERIMENT_DIR if os.path.exists(os.path.join(EXPERIMENT_DIR, "trainer_state.json"))
    else False
)

model.fit(df, val_fraction=0.1, resume_from_checkpoint=resume_ckpt)


BEST_MODEL_DIR = os.path.join(EXPERIMENT_DIR, "best_model")
model.save(BEST_MODEL_DIR)
print("Saved best model to:", BEST_MODEL_DIR)

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

    synthetic_df[cont_cols] = scaler.inverse_transform(synthetic_df[cont_cols])

    out_path = os.path.join(
        DATA_DIR,
        f"synthetic_tabula_gpt2_rep_{i}.csv",
    )

    synthetic_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
