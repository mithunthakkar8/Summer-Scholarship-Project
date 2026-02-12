# ======================================================
# Phase 1: GaussianCopulaSynthesizer → Exogenous Synthetic Data
# + Structural Latent Generation (SMS, SMP)
#
# Output: df_exo_gc_with_latents.csv
# ======================================================

import os
import pandas as pd
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

# ------------------------------------------------------
# PATHS
# ------------------------------------------------------
DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data/"
# DATA_DIR = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Datasets\PISA 2022"

INPUT_CORE = os.path.join(DATA_DIR, "df_core.csv")
STRUCTURAL_PATHS = os.path.join(DATA_DIR, "sem_real_structural_paths.csv")
R2_PATH = os.path.join(DATA_DIR, "sem_real_rsquare_raw.csv")

OUT_PHASE1 = os.path.join(DATA_DIR, "df_exo_gc_with_latents.csv")

# ------------------------------------------------------
# LOAD REAL DATA
# ------------------------------------------------------
df = pd.read_csv(INPUT_CORE)

# Exogenous predictors used in SEM structural model
EXO_COLS = ["ESCS_z", "female", "MISCED", "SCHSIZE_z", "ST001D01T"]
df_exo = df[EXO_COLS].copy()

# ------------------------------------------------------
# METADATA
# ------------------------------------------------------
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_exo)

# ------------------------------------------------------
# FIT GAUSSIAN COPULA SYNTHESIZER
# ------------------------------------------------------
synth = GaussianCopulaSynthesizer(
    metadata=metadata,
    enforce_min_max_values=True
)

synth.fit(df_exo)

# ------------------------------------------------------
# SAMPLE SYNTHETIC EXOGENOUS DATA
# ------------------------------------------------------
N_SYN = 1000
exo_gc = synth.sample(num_rows=N_SYN)

CSV_WITH_LATENTS_CSV = os.path.join(DATA_DIR, "df_core_fullnames_with_latents.csv")

# Load real data
df_with_latents = pd.read_csv(CSV_WITH_LATENTS_CSV)


exo_gc["SPI"] = df_with_latents['Latent Factor: School-level Parental Involvement (SPI)']

# ------------------------------------------------------
# LOAD SEM STRUCTURAL PATHS (Real)
# ------------------------------------------------------
betas = pd.read_csv(STRUCTURAL_PATHS)

# SMS regressions
beta_sms_df = betas[betas["DV"] == "SMS"]
beta_sms = dict(zip(beta_sms_df["IV"], beta_sms_df["B_unstd"]))

# SMP regressions
beta_smp_df = betas[betas["DV"] == "SMP"]
beta_smp = dict(zip(beta_smp_df["IV"], beta_smp_df["B_unstd"]))

# ------------------------------------------------------
# LOAD R² VALUES (Real SEM)
# ------------------------------------------------------
r2_df = pd.read_csv(R2_PATH)
r2_dict = dict(zip(r2_df["latent"], r2_df["R2"]))

R2_SMS = r2_dict["SMS"]
R2_SMP = r2_dict["SMP"]

# ------------------------------------------------------
# LATENT GENERATION FUNCTION
# ------------------------------------------------------
def compute_latent(df_local, beta_dict, r2_value):
    """
    Compute synthetic latent variable:
        latent = Σ β_i * X_i + ε
    where Var(ε) = (1 - R²)
    """
    pred = np.zeros(len(df_local))

    for iv, coef in beta_dict.items():
        pred += coef * df_local[iv]

    residual_sd = np.sqrt(max(0, 1 - r2_value))
    noise = np.random.normal(0, residual_sd, size=len(df_local))

    return pred + noise

# ------------------------------------------------------
# COMPUTE STRUCTURAL LATENTS (SMS, SMP)
# ------------------------------------------------------
exo_gc["SMS"] = compute_latent(exo_gc, beta_sms, R2_SMS)
exo_gc["SMP"] = compute_latent(exo_gc, beta_smp, R2_SMP)

# ------------------------------------------------------
# SAVE OUTPUT
# ------------------------------------------------------
exo_gc.to_csv(OUT_PHASE1, index=False)
print("Saved:", OUT_PHASE1)
