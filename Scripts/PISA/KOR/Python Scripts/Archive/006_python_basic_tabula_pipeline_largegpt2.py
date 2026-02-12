import os
import pandas as pd
import numpy as np
import torch

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
    "ST004D01T",
    "ST268Q01JA",
    "ST268Q04JA",
    "ST268Q07JA",
    "IMMIG",
]

# -------------------------------------------------
# Import Tabula
# -------------------------------------------------
from tabula import Tabula

# -------------------------------------------------
# Initialise model
# -------------------------------------------------
model = Tabula(
    llm="gpt2-large",
    experiment_dir=EXPERIMENT_DIR,
    batch_size=16,
    epochs=80,
    categorical_columns=categorical_columns,
)

# -------------------------------------------------
# (Optional) load pretrained GPT-2 weights
# -------------------------------------------------
# torch.load(...) here is ONLY for initialisation
# model.model.load_state_dict(torch.load("some_pretrained.pt"), strict=False)

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
        f"synthetic_tabula_distilgpt2_rep_{i}.csv",
    )

    synthetic_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
