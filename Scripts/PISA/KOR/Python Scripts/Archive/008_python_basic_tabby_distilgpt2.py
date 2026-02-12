import os
import time
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import transformers

import random
from transformers import set_seed

print("[BOOT] Script starting...", flush=True)

# -------------------------------------------------
# Reproducibility
# -------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"[BOOT] Random seed set to {SEED}", flush=True)

# -------------------------------------------------
# NumPy compatibility
# -------------------------------------------------
if not hasattr(np, "float"):
    np.float = float

# -------------------------------------------------
# Paths
# -------------------------------------------------
DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data"
INPUT_CSV = os.path.join(DATA_DIR, "df_core_with_smp_latent_SGP.csv")
OUTPUT_PREFIX = "synthetic_tabby_rep"

print(f"[PATH] Input CSV: {INPUT_CSV}", flush=True)

# -------------------------------------------------
# Load data
# -------------------------------------------------
df = pd.read_csv(INPUT_CSV)
print(f"[DATA] Loaded data shape: {df.shape}", flush=True)

cols = list(df.columns)
print(f"[DATA] Number of columns: {len(cols)}", flush=True)
print(f"[DATA] Columns: {cols}", flush=True)

# -------------------------------------------------
# Standardize data
# -------------------------------------------------
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=cols,
    index=df.index
)

print("[DATA] Post-standardization check (mean):", flush=True)
print(df_scaled.mean().round(3), flush=True)
print("[DATA] Post-standardization check (std):", flush=True)
print(df_scaled.std(ddof=0).round(3), flush=True)

# -------------------------------------------------
# Reduce numeric precision
# -------------------------------------------------
df_scaled = df_scaled.round(3)
print("[DATA] Rounded numeric precision to 3 decimals", flush=True)

# -------------------------------------------------
# Load Tabby model
# -------------------------------------------------
from src.tabby import MHTabbyGPT2Config, MHTabbyGPT2

MODEL_NAME = "sonicc/tabby-distilgpt2-diabetes"
print(f"[MODEL] Loading Tabby model: {MODEL_NAME}", flush=True)

config = MHTabbyGPT2Config.from_pretrained(MODEL_NAME)
model = MHTabbyGPT2.from_pretrained(MODEL_NAME)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
# Ensure EOC is the EOS token used by the model
tokenizer.eos_token = "<EOC>"
tokenizer.pad_token = tokenizer.eos_token


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"[MODEL] Device: {device}", flush=True)
if torch.cuda.is_available():
    print(f"[MODEL] GPU: {torch.cuda.get_device_name(0)}", flush=True)

# -------------------------------------------------
# Configure column order
# -------------------------------------------------
column_names_tokens = tokenizer(cols, add_special_tokens=False).input_ids
token_heads = list(range(len(cols)))

model.set_generation_mode(
    token_heads=token_heads,
    column_names_tokens=column_names_tokens
)

print("[MODEL] Generation mode set", flush=True)
print(f"[MODEL] token_heads: {token_heads}", flush=True)

# -------------------------------------------------
# Parsing helper
# -------------------------------------------------
import re

PAIR_RE = re.compile(
    r"^\s*([^\s]+)\s+is\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)

def parse_line(line, columns):
    # Split into candidate chunks using <EOC>
    chunks = [c for c in line.split("<EOC>") if " is " in c]
    d = {}

    for c in chunks:
        m = PAIR_RE.match(c)
        if not m:
            continue

        col = m.group(1)
        val = m.group(2)

        if col in columns and col not in d:
            try:
                d[col] = float(val)
            except ValueError:
                continue

    MIN_COLS = int(0.8 * len(columns))  # 80% coverage
    return d if len(d) >= MIN_COLS else None


# -------------------------------------------------
# Sampling configuration
# -------------------------------------------------
N_REPS = 5
N_SAMPLES = len(df)
MAX_LEN = len(cols) * 10

print("[GEN] Sampling configuration:", flush=True)
print(f"[GEN] N_REPS     = {N_REPS}", flush=True)
print(f"[GEN] N_SAMPLES  = {N_SAMPLES}", flush=True)
print(f"[GEN] MAX_LEN    = {MAX_LEN}", flush=True)

# -------------------------------------------------
# Sample synthetic data
# -------------------------------------------------
for i in range(1, N_REPS + 1):
    print(f"\n[GEN] Starting replication {i}/{N_REPS}", flush=True)
    start_time = time.time()

    rows = []
    attempts = 0

    while len(rows) < N_SAMPLES:
        attempts += 1

        inputs = torch.full(
            (1, 1),
            tokenizer.bos_token_id,
            device=device
        )

        toks = model.generate(
            inputs,
            do_sample=True,
            num_beams=1,
            max_length=MAX_LEN,
            pad_token_id=tokenizer.pad_token_id
        )[..., 1:]

        decoded = tokenizer.decode(toks[0], skip_special_tokens=False)
        if attempts == 1:
            print("[DEBUG] decoded sample:", decoded[:300], flush=True)

        parsed = parse_line(decoded, cols)

        if parsed is not None:
            rows.append(parsed)

        if attempts % 100 == 0:
            print(
                f"[GEN][rep {i}] attempts={attempts}, "
                f"accepted={len(rows)}, "
                f"acceptance_rate={len(rows)/attempts:.3f}",
                flush=True
            )

    elapsed = time.time() - start_time
    print(
        f"[GEN] Rep {i} complete: {N_SAMPLES} rows "
        f"in {attempts} attempts "
        f"(acceptance_rate={N_SAMPLES/attempts:.3f}, "
        f"time={elapsed/60:.2f} min)",
        flush=True
    )

    synthetic_scaled = pd.DataFrame(rows).reindex(columns=cols)
    synthetic_scaled = synthetic_scaled.fillna(
        synthetic_scaled.median(numeric_only=True)
    )


    synthetic_df = pd.DataFrame(
        scaler.inverse_transform(synthetic_scaled),
        columns=cols
    )

    out_path = os.path.join(
        DATA_DIR,
        f"{OUTPUT_PREFIX}_{i}.csv"
    )
    synthetic_df.to_csv(out_path, index=False)

    print(f"[SAVE] Saved {out_path}", flush=True)

print("\n[DONE] All replications completed successfully", flush=True)
