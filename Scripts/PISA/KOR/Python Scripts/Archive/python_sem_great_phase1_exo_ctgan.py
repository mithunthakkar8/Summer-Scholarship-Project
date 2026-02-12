from sdv.single_table import CTGANSynthesizer
import os
import pandas as pd
from sdv.metadata import SingleTableMetadata

DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data/"
# DATA_DIR = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Datasets\PISA 2022"
INPUT_CSV = os.path.join(DATA_DIR, "df_core.csv")

# Load real data
df = pd.read_csv(INPUT_CSV)

EXO_COLS = ["ESCS_z", "female", "MISCED", "SCHSIZE_z", "ST001D01T"]

df_exo = df[EXO_COLS]

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_exo)

# Setup CTGAN
ctgan = CTGANSynthesizer(
    epochs=300,
    batch_size=128,
    verbose=True,
    pac = 1,
    metadata=metadata
)

ctgan.fit(df_exo)

N_SYN = 1000
exo_ctgan = ctgan.sample(N_SYN)
DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data/"
CSV_WITH_LATENTS_CSV = os.path.join(DATA_DIR, "df_core_fullnames_with_latents.csv")

# Load real data
df_with_latents = pd.read_csv(CSV_WITH_LATENTS_CSV)

exo_ctgan["SPI"] = df_with_latents['Latent Factor: School Parental Involvement (SPI)']
structural_paths_csv = os.path.join(DATA_DIR, "sem_real_structural_paths.csv")
betas = pd.read_csv(structural_paths_csv)

# Extract β paths for SMS and SMP
betas_sms = betas[betas["DV"] == "SMS"]
betas_smp = betas[betas["DV"] == "SMP"]

# Convert IV → β mapping for each latent
beta_sms = dict(zip(betas_sms["IV"], betas_sms["B_unstd"]))
beta_smp = dict(zip(betas_smp["IV"], betas_smp["B_unstd"]))


# Load R-square values
rsquare_raw_csv = os.path.join(DATA_DIR, "sem_real_rsquare_raw.csv")
r2_df = pd.read_csv(rsquare_raw_csv)

# Convert to dictionary
r2_dict = dict(zip(r2_df["latent"], r2_df["R2"]))

# Extract the latent R² values
R2_SMS = r2_dict["SMS"]
R2_SMP = r2_dict["SMP"]


import numpy as np

def compute_latent(df, beta_dict, r2_value):
    """
    Compute synthetic latent variable using:
        latent = Σ β_i * IV_i + ε
    where Var(ε) = residual variance = (1 - R²)
    """
    # Linear combination of predictors
    pred = np.zeros(len(df))

    for iv, coef in beta_dict.items():
        pred += coef * df[iv]

    # Residual noise
    residual_sd = np.sqrt(max(0, 1 - r2_value))
    noise = np.random.normal(0, residual_sd, size=len(df))

    return pred + noise


# SMS (endogenous latent 1)
exo_ctgan["SMS"] = compute_latent(
    df=exo_ctgan,
    beta_dict=beta_sms,
    r2_value=R2_SMS
)

# SMP (endogenous latent 2)
exo_ctgan["SMP"] = compute_latent(
    df=exo_ctgan,
    beta_dict=beta_smp,
    r2_value=R2_SMP
)


exo_ctgan.to_csv(os.path.join(DATA_DIR, "df_exo_ctgan_with_latents.csv"))