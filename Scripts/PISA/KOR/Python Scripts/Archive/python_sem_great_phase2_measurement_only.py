#!/usr/bin/env python
"""
Phase 2 – GReaT measurement-model generator (NeSI version)

Pipeline:
    - Train GReaT on REAL data: exogenous + latent factors + indicators.
    - Use Phase 1 synthetic structural data (exo + latents) as fixed input.
    - Impute indicator columns using GReaT.impute().
    - Output: full synthetic dataset with structural layer from SEM (Phase 1)
      and measurement layer from GReaT (Phase 2).

Assumptions:
    DATA_DIR contains:
      - df_core_fullnames_with_latents.csv
      - df_exo_gc_with_latents.csv   (output of Phase 1)

Author: Mithun + ChatGPT
"""

import os
import random

import numpy as np

if not hasattr(np, "float"):
    np.float = float
import pandas as pd

from be_great import GReaT



# ============================
# CONFIG
# ============================

DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data/"
# DATA_DIR = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Datasets\PISA 2022"

# ------------------------------------------
# LOAD VARIABLE NAME MAPPING (shortnames)
# ------------------------------------------
MAPPING_CSV = os.path.join(DATA_DIR, "pisa_shortname_mapping_with_safe_short.csv")
mapping_df = pd.read_csv(MAPPING_CSV)

# map REAL → short
MAP_TO_SHORT = dict(zip(mapping_df["old"], mapping_df["safe_short"]))
MAP_FROM_FULL = dict(zip(mapping_df["full"], mapping_df["safe_short"]))

# OPTIONAL: if you want to later recover full names
MAP_TO_CODE = dict(zip(mapping_df["safe_short"], mapping_df["old"]))


REAL_WITH_LATENTS_CSV = os.path.join(
    DATA_DIR, "df_core_fullnames_with_latents.csv"
)
STRUCT_SYN_CSV = os.path.join(DATA_DIR, "df_exo_gc_with_latents.csv")

OUT_MODEL_DIR = os.path.join(DATA_DIR, "great_phase2_model")
OUT_SYNTH_FULL_CSV = os.path.join(
    DATA_DIR, "df_phase2_great_synthetic_full.csv"
)

# Exogenous + latent factor columns
COND_COLS = [
    "ESCS_z",
    "female",
    "MISCED",
    "SCHSIZE_z",
    "ST001D01T"
]

COND_COLS = [MAP_TO_SHORT[c] for c in COND_COLS]

# Indicator columns (targets for GReaT)
TARGET_COLS = [
    # SMP indicators
    "PV1MATH", "PV2MATH", "PV3MATH", "PV4MATH", "PV5MATH",
    "PV6MATH", "PV7MATH", "PV8MATH", "PV9MATH", "PV10MATH",

    # SMS indicators
    "ST268Q01JA", "ST268Q04JA", "ST268Q07JA",

    # SPI indicators (corrected)
    "SC064Q01TA",
    "SC064Q02TA",
    "SC064Q03TA",
    "SC064Q04NA",   # ← corrected
    "SC064Q05WA",   # ← corrected
    "SC064Q06WA",   # ← corrected
]

TARGET_COLS = [MAP_TO_SHORT[c] for c in TARGET_COLS]

ALL_COLS = COND_COLS + TARGET_COLS


LATENT_MAP_TO_SHORT = {
    "Latent Factor: Math Performance (SMP)": "latent_factor_student_math_performance",
    "Latent Factor: Student Self-Efficiency in Math (SMS)": "latent_factor_student_math_self_efficacy",
    "Latent Factor: School Parental Involvement (SPI)": "latent_factor_school_level_parental_involvement"
}


# ============================
# UTILS
# ============================

def set_global_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ============================
# MAIN PIPELINE
# ============================

def main():
    set_global_seed(42)

    print("=== Phase 2: GReaT measurement model (NeSI) ===")
    print(f"DATA_DIR: {DATA_DIR}")

    # ----------------------------
    # 1. Load real data
    # ----------------------------
    print("[1/6] Loading real data with latents...")
    if not os.path.exists(REAL_WITH_LATENTS_CSV):
        raise FileNotFoundError(
            f"Real data file not found: {REAL_WITH_LATENTS_CSV}"
        )

    df_real = pd.read_csv(REAL_WITH_LATENTS_CSV)
    df_real = df_real.rename(columns=MAP_FROM_FULL)
    df_real = df_real.rename(columns=LATENT_MAP_TO_SHORT)
    print(f"    Real data shape: {df_real.shape}")

    # Check for required columns
    missing_cond = [c for c in COND_COLS if c not in df_real.columns]
    missing_tgt = [c for c in TARGET_COLS if c not in df_real.columns]

    if missing_cond or missing_tgt:
        raise ValueError(
            "Real training data is missing required columns.\n"
            f"  Missing COND_COLS: {missing_cond}\n"
            f"  Missing TARGET_COLS: {missing_tgt}\n"
            "Please fix column names or update COND_COLS / TARGET_COLS."
        )

    df_train = df_real[ALL_COLS].copy()
    print(f"    Training frame shape: {df_train.shape}")

    # ----------------------------
    # 2. Train GReaT
    # ----------------------------
    print("[2/6] Initializing GReaT model...")

    # NOTE: adjust hyperparams if needed (epochs, batch_size, etc.)
    model = GReaT(
        llm="distilgpt2",
        batch_size=32,
        epochs=5,
        fp16=True,
        dataloader_num_workers=4,
    )

    print("[3/6] Fitting GReaT on real data (this may take a while)...")
    model.fit(df_train)

    # Save model checkpoint (useful on NeSI)
    print(f"[4/6] Saving trained GReaT model to: {OUT_MODEL_DIR}")
    model.save(OUT_MODEL_DIR)

    # ----------------------------
    # 3. Load Phase 1 synthetic structural data
    # ----------------------------
    print("[5/6] Loading Phase 1 structural synthetic data...")
    if not os.path.exists(STRUCT_SYN_CSV):
        raise FileNotFoundError(
            f"Phase 1 structural data file not found: {STRUCT_SYN_CSV}"
        )

    df_struct_syn = pd.read_csv(STRUCT_SYN_CSV)
    df_struct_syn = df_struct_syn.rename(columns=MAP_TO_SHORT)
    print(f"    Structural synthetic shape: {df_struct_syn.shape}")

    # Ensure required conditioning columns are present
    missing_struct = [c for c in COND_COLS if c not in df_struct_syn.columns]
    if missing_struct:
        raise ValueError(
            "Phase 1 structural synthetic data is missing required columns.\n"
            f"  Missing: {missing_struct}\n"
            f"  Expected conditioning columns: {COND_COLS}"
        )

    # Keep only the structural layer
    # df_cond_only = df_struct_syn[COND_COLS].copy()

    # ----------------------------
    # 4. Build imputation frame
    # ----------------------------
    print("    Building imputation frame (fix exo+latents, NaN indicators)...")
    df_to_impute = df_struct_syn[COND_COLS].copy()

    for col in TARGET_COLS:
        df_to_impute[col] = np.nan

    print(f"    Imputation frame shape: {df_to_impute.shape}")
    print("    Example row BEFORE imputation:")
    print(df_to_impute.iloc[0])

    # ----------------------------
    # 5. Impute indicators with GReaT
    # ----------------------------
    print("[6/6] Imputing indicators with GReaT.impute...")
    df_imputed = model.impute(
    df_to_impute,
    max_length=1024)


    print("    Example row AFTER imputation:")
    print(df_imputed.iloc[0])

    # Keep exact order of ALL_COLS for clarity
    df_synth_full = df_imputed[ALL_COLS].copy()

    print(f"    Final synthetic full shape: {df_synth_full.shape}")
    print(f"    Saving to: {OUT_SYNTH_FULL_CSV}")
    df_synth_full.to_csv(OUT_SYNTH_FULL_CSV, index=False)

    print("=== Phase 2 complete. ===")
    print("You can now:")
    print(f"  - load {OUT_SYNTH_FULL_CSV} in R")
    print("  - run your lavaan SEM")
    print("  - compare measurement + structural parameters vs real data.")


if __name__ == "__main__":
    main()
