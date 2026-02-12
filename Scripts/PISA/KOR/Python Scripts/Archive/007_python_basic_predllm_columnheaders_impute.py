import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

REAL_DATA_PATH = Path(
    r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Datasets\PISA 2022\df_core_with_smp_latent_SGP.csv"
)

SYNTH_DIR = Path(
    r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PredLLM\DistilGPT2"
)

FILE_PATTERN = "synthetic_predllm_rep_*.csv"
OVERWRITE = False

# -------------------------------------------------
# LOAD TRUE COLUMN NAMES
# -------------------------------------------------

df_real = pd.read_csv(REAL_DATA_PATH)
real_columns = df_real.columns.tolist()
n_cols = len(real_columns)

print(f"Loaded {n_cols} column names.")

# -------------------------------------------------
# SELECT ONLY RAW FILES
# -------------------------------------------------

files = sorted(
    p for p in SYNTH_DIR.glob(FILE_PATTERN)
    if "_with_headers" not in p.name
)

if not files:
    raise FileNotFoundError("No raw Pred-LLM files found.")

# -------------------------------------------------
# PROCESS FILES
# -------------------------------------------------

for path in files:
    print(f"\nProcessing: {path.name}")

    df = pd.read_csv(path, header=None)

    # ---- Detect numeric index row [0,1,2,...]
    is_index_row = (
        df.shape[1] == n_cols and
        np.allclose(df.iloc[0].values, np.arange(n_cols))
    )

    if is_index_row:
        print("  Detected bogus index row. Removing.")
        df = df.iloc[1:].reset_index(drop=True)
    else:
        print("  No bogus index row detected.")

    # ---- Safety check
    if df.shape[1] != n_cols:
        raise ValueError(
            f"Column mismatch in {path.name}: "
            f"{df.shape[1]} vs {n_cols}"
        )

    # ---- Assign headers
    df.columns = real_columns

    # ---- Save
    out_path = (
        path if OVERWRITE
        else path.with_name(path.stem + "_with_headers.csv")
    )

    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path.name}")

print("\nAll Pred-LLM replications cleaned correctly.")
