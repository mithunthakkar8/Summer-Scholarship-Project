import os
import torch
import numpy as np
import pandas as pd
from semopy import Model
from scipy.stats import spearmanr
from transformers import TrainerCallback
import logging
import sys


print("===== GPU / CUDA INFO =====")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
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

# =================================================
# LOGGING SETUP
# =================================================

LOG_DIR = "/nfs/scratch/thakkamith/TIMSS/logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "tabula_sem_epoch_selection.log")

logger = logging.getLogger("TABULA_SEM")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("===== Tabula GPT2 PIPELINE STARTED =====")

# -------------------------------------------------
# GPU INFO
# -------------------------------------------------
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# PATHS
# -------------------------------------------------
DATA_DIR = "/nfs/scratch/thakkamith/TIMSS/data"
INPUT_CSV = os.path.join(DATA_DIR, "df_coreSGP.csv")
EXPERIMENT_DIR = os.path.join(DATA_DIR, "tabula_gpt2")
BEST_MODEL_DIR = os.path.join(EXPERIMENT_DIR, "best_model")
FINAL_SYN_DIR = os.path.join(DATA_DIR, "tabula_best_epoch_final")

os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(FINAL_SYN_DIR, exist_ok=True)

logger.info(f"Input CSV: {INPUT_CSV}")
logger.info(f"Experiment dir: {EXPERIMENT_DIR}")

# -------------------------------------------------
# LOAD REAL DATA
# -------------------------------------------------
df_real = pd.read_csv(INPUT_CSV)
logger.info(f"Loaded real data: {df_real.shape}")

# -------------------------------------------------
# CATEGORICAL COLUMNS
# -------------------------------------------------
categorical_columns = [
    "BTBG06I", "BTBG06J", "BTBG06K",
    "BTBS17A", "BTBS17B", "BTBS17C", "BTBS17D",
    "BTDGSOS", "BTBG13A", "BTBG13B", "BTBG13C", "BTBG13D",
    "BTBG13E", "BTBG13F", "BTBG13G", "BTBG13H", "BTBG13I"
]

# -------------------------------------------------
# SEM MODEL (RAW DATA — NO STANDARDIZATION)
# -------------------------------------------------
SEM_MODEL = """
# Measurement model
ACM =~ BTBG06I + BTBG06J + BTBG06K
SSF =~ BTBGSOS + BTDGSOS
TCI =~ BTBS17A + BTBS17B + BTBS17C + BTBS17D
LEI =~ BTBG13A + BTBG13B + BTBG13C + BTBG13D + BTBG13E + BTBG13F + BTBG13G + BTBG13H + BTBG13I

# Structural model
ACM ~ SSF + TCI + LEI + BTBG10
"""


# -------------------------------------------------
# REAL STRUCTURAL PATHS (TARGET)
# -------------------------------------------------
logger.info("Running SEM on REAL data...")

sem_vars = [
    # Academic Mindedness
    "BTBG06I", "BTBG06J", "BTBG06K",

    # School Safety
    "BTBGSOS", "BTDGSOS",

    # Teacher–Student Interaction
    "BTBS17A", "BTBS17B", "BTBS17C", "BTBS17D",

    # Learning Environment
    "BTBG13A", "BTBG13B", "BTBG13C", "BTBG13D",
    "BTBG13E", "BTBG13F", "BTBG13G", "BTBG13H", "BTBG13I",

    # Number of Students in Class (single-item)
    "BTBG10"
]

df_real_sem = df_real[sem_vars].copy()
for c in df_real_sem.columns:
    df_real_sem[c] = pd.to_numeric(df_real_sem[c], errors="coerce")
before = len(df_real_sem)
df_real_sem = df_real_sem.dropna()
logger.info(f"Real SEM rows after NA drop: {before} → {len(df_real_sem)}")

real_sem = Model(SEM_MODEL)
real_sem.fit(df_real_sem)
real_est = real_sem.inspect(std_est=True)

real_paths = real_est[
    (real_est["op"] == "~") &
    (real_est["lval"] == "ACM")
][["lval", "rval", "Est. Std"]].copy()

real_paths["path"] = real_paths["rval"] + " -> " + real_paths["lval"]
logger.info(f"Number of real structural paths: {len(real_paths)}")

# -------------------------------------------------
# SEM-DRIVEN CALLBACK (EVERY X EPOCHS)
# -------------------------------------------------
class SEMEpochSelectionCallback(TrainerCallback):
    def __init__(
        self,
        tabula_model,
        real_paths,
        sem_model_desc,
        best_model_dir,
        eval_interval=5,        # <<< HYPERPARAMETER
        n_eval_samples=400,
        start_eval_epoch=10,     # <<< NEW
        device="cuda",
    ):
        self.tabula_model = tabula_model
        self.real_paths = real_paths
        self.sem_model_desc = sem_model_desc
        self.best_model_dir = best_model_dir
        self.eval_interval = eval_interval
        self.start_eval_epoch = start_eval_epoch
        self.n_eval_samples = n_eval_samples
        self.device = device

        self.best_score = -np.inf
        self.best_epoch = None

    def on_epoch_end(self, args, state, control, **kwargs):

        epoch = int(round(state.epoch))

        # --- do nothing before start epoch ---
        if epoch < self.start_eval_epoch:
            return

        # --- then apply interval ---
        if (epoch - self.start_eval_epoch) % self.eval_interval != 0:
            return


        logger.info(f"[SEM EVAL] Epoch {epoch} started")

        self.tabula_model.model.eval()
        self.tabula_model.model.to(self.device)

        # ---- SAMPLE SYNTHETIC ----
        try:
            synth = self.tabula_model.sample(
                n_samples=self.n_eval_samples,
                max_length=400,
                device=self.device
            ).dropna()
        except Exception as e:
            logger.error(f"[SEM EVAL] Sampling failed: {e}", exc_info=True)
            return

        logger.info(f"[SEM EVAL] Synthetic rows: {len(synth)}")

        # ---- RUN SEM ----
        try:
            # ----- Prepare data for SEM: numeric only -----

            # columns actually used by SEM
            syn_sem_df = synth[sem_vars].copy()

            # force numeric
            for c in syn_sem_df.columns:
                syn_sem_df[c] = pd.to_numeric(syn_sem_df[c], errors="coerce")

            syn_sem_df = syn_sem_df.dropna()

            # ---- run SEM safely ----
            syn_sem = Model(self.sem_model_desc)
            syn_sem.fit(syn_sem_df)
            syn_est = syn_sem.inspect(std_est=True)

            logger.debug(f"[SEM EVAL] SEM input shape: {syn_sem_df.shape}")

        except Exception as e:
            logger.error(f"[SEM EVAL] SEM failed: {e}", exc_info=True)
            return

        syn_paths = syn_est[
            (syn_est["op"] == "~") &
            (syn_est["lval"] == "ACM")
        ][["lval", "rval", "Est. Std"]].copy()

        syn_paths["path"] = syn_paths["rval"] + " -> " + syn_paths["lval"]

        merged = self.real_paths.merge(
            syn_paths,
            on="path",
            suffixes=("_real", "_syn")
        )

        real_beta = merged["Est. Std_real"].values
        syn_beta = merged["Est. Std_syn"].values
        if len(merged) == 0:
            dir_consistency = np.nan
        else:
            dir_consistency = np.mean(
                np.sign(real_beta) == np.sign(syn_beta)
            )

        logger.info(f"[SEM EVAL] Directional consistency = {dir_consistency:.3f}")

        from scipy.stats import spearmanr

        if len(merged) >= 3:
            rank_corr, _ = spearmanr(real_beta, syn_beta)
        else:
            rank_corr = np.nan

        logger.info(f"[SEM EVAL] Rank correlation = {rank_corr:.3f}")

        ALPHA = 0.5   # weight on directional

        if np.isnan(dir_consistency) or np.isnan(rank_corr):
            hybrid_score = np.nan
        else:
            hybrid_score = ALPHA * dir_consistency + (1 - ALPHA) * rank_corr

        logger.info(f"[SEM EVAL] Hybrid score = {hybrid_score:.3f}")

        if not np.isnan(hybrid_score) and hybrid_score > self.best_score:
            self.best_score = hybrid_score
            self.best_epoch = epoch

            logger.info(f"[SEM EVAL] NEW BEST MODEL at epoch {epoch}")

            torch.save(
                self.tabula_model.model.state_dict(),
                os.path.join(self.best_model_dir, "model.pt")
            )

# -------------------------------------------------
# TRAIN TABULA WITH CALLBACK
# -------------------------------------------------
from tabula import Tabula

MAX_EPOCHS = 60
BATCH_SIZE = 16
EVAL_INTERVAL = 5
N_EVAL_SAMPLES = min(1000, len(df_real))
start_eval_epoch = 20

logger.info(f"Training config: epochs={MAX_EPOCHS}, eval_interval={EVAL_INTERVAL}, eval_samples={N_EVAL_SAMPLES}")

model = Tabula(
    llm="gpt2",
    experiment_dir=EXPERIMENT_DIR,
    batch_size=BATCH_SIZE,
    epochs=MAX_EPOCHS,
    categorical_columns=categorical_columns,
)

# register callback
callback = SEMEpochSelectionCallback(
    tabula_model=model,
    real_paths=real_paths,
    sem_model_desc=SEM_MODEL,
    best_model_dir=BEST_MODEL_DIR,
    eval_interval=EVAL_INTERVAL,
    start_eval_epoch=start_eval_epoch,
    n_eval_samples=N_EVAL_SAMPLES,
    device=device,
)

model.trainer_callbacks.append(callback)

logger.info("Starting Tabula training with SEM-driven selection...")
model.fit(df_real, resume_from_checkpoint=False)

# -------------------------------------------------
# LOAD BEST MODEL & GENERATE FINAL DATASETS
# -------------------------------------------------
logger.info("Loading best SEM-selected model...")

best_model_path = os.path.join(BEST_MODEL_DIR, "model.pt")

if not os.path.exists(best_model_path):
    logger.warning("No SEM-best model found — saving final epoch model instead.")
    torch.save(model.model.state_dict(), best_model_path)

import shutil

logger.info("Cleaning experiment directory (keeping best model only)...")

for fname in os.listdir(EXPERIMENT_DIR):
    fpath = os.path.join(EXPERIMENT_DIR, fname)

    # keep only best_model directory
    if fpath == BEST_MODEL_DIR:
        continue

    try:
        if os.path.isdir(fpath):
            shutil.rmtree(fpath)
        else:
            os.remove(fpath)
    except Exception as e:
        logger.warning(f"Could not delete {fpath}: {e}")



model.model.to(device)
model.model.eval()

N_FINAL_REPS = 5
N_FINAL_SAMPLES = len(df_real)

logger.info("Generating final synthetic datasets...")



for i in range(1, N_FINAL_REPS + 1):
    logger.info(f"Generating synthetic run {i}")
    syn = model.sample(
        n_samples=N_FINAL_SAMPLES,
        max_length=400,
        device=device
    )

    out_path = os.path.join(
        FINAL_SYN_DIR,
        f"synthetic_tabula_gpt2_bestepoch_rep_{i}.csv"
    )
    syn.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")

logger.info("===== PIPELINE COMPLETE =====")
