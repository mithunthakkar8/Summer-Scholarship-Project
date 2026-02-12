import os
import numpy as np
import pandas as pd
import torch

if not hasattr(np, "float"):
    np.float = float

DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data"
INPUT_CSV = os.path.join(DATA_DIR, "df_coreSGP.csv")

from tabular_llm.predllm import PredLLM
from tabular_llm.predllm_utils import _encode_row_partial

from be_great import GReaT

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def compute_pc1(df, cols, name):
    df_sub = df[cols].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(df_sub)
    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(X).flatten()
    out = pd.Series(index=df.index, dtype=float)
    out.loc[df_sub.index] = pc1
    print(f"{name}: PC1 explained variance = {pca.explained_variance_ratio_[0]:.3f}")
    return out

df = pd.read_csv(INPUT_CSV)

STRUCT_COLS = [
    "PC_SPI",
    "PC_SMS",
    "ST001D01T",
    "AGE",
    "IMMIG",
    "MISCED",
    "ESCS",
    "SCHSIZE",
    "MCLSIZE",
    "PC_SMP",
    "ST004D01T",
]

TARGET_COL = "PC_SMP"


PV_COLS = [f"PV{i}MATH" for i in range(1, 11)]
SC_COLS = [
    "SC064Q01TA","SC064Q02TA","SC064Q03TA",
    "SC064Q04NA","SC064Q05WA","SC064Q06WA","SC064Q07WA",
]
ST_COLS = ["ST268Q01JA","ST268Q04JA","ST268Q07JA"]

df["PC_SMP"] = compute_pc1(df, PV_COLS, "PC_SMP")
df["PC_SMS"] = compute_pc1(df, SC_COLS, "PC_SMS")
df["PC_SPI"] = compute_pc1(df, ST_COLS, "PC_SPI")

df_struct = df[STRUCT_COLS].select_dtypes(include=[np.number])

# move target last
feature_cols = [c for c in df_struct.columns if c != TARGET_COL]
df_struct = df_struct[feature_cols + [TARGET_COL]]

# =================================================
# STANDARDIZE PredLLM INPUT (CRITICAL)
# =================================================
from sklearn.preprocessing import StandardScaler

scaler_predllm = StandardScaler()

# Fit on ALL columns (including target)
# PredLLM treats everything symmetrically
df_struct_scaled = pd.DataFrame(
    scaler_predllm.fit_transform(df_struct),
    columns=df_struct.columns,
    index=df_struct.index,
)

# Replace df_struct used for training
df_struct = df_struct_scaled

print("[INFO] PredLLM input standardized")


predllm = PredLLM(
    llm="distilgpt2",
    batch_size=32,
    epochs=50,
)


predllm.fit(df_struct)

encoded = _encode_row_partial(df_struct.iloc[0], shuffle=False)
prompt_len = len(predllm.tokenizer(encoded)["input_ids"])

N_SAMPLES = len(df_struct)

df_latent = predllm.sample_new(
    n_samples=N_SAMPLES,
    max_length=prompt_len,
    task="regression",
)

df_latent.columns = df_struct.columns

# =================================================
# UNSTANDARDIZE PredLLM OUTPUT
# =================================================
df_latent = pd.DataFrame(
    scaler_predllm.inverse_transform(df_latent),
    columns=df_latent.columns,
    index=df_latent.index,
)

print("[INFO] PredLLM output unstandardized")


def great_generate_block(
    df_real: pd.DataFrame,      # real data for training
    df_latent: pd.DataFrame,    # PredLLM-generated latents
    fixed_cols: list,
    target_cols: list,
    model_dir: str,
    epochs: int = 60,
    batch_size: int = 16,
    max_length: int = 300,
):
    """
    Train GReaT on REAL data (LV → items),
    then impute target items conditioned on
    PredLLM-generated latent variables.
    """

    # =========================
    # 1. Train on REAL data
    # =========================
    train_df = df_real[fixed_cols + target_cols].dropna()

    model = GReaT(
        llm="distilgpt2",
        batch_size=batch_size,
        epochs=epochs,
        experiment_dir=model_dir,
    )

    model.fit(train_df)


    # =========================
    # 2. Build imputation frame
    #    using PredLLM latents
    # =========================
    df_miss = pd.DataFrame(index=df_latent.index)

    for c in fixed_cols:
        df_miss[c] = df_latent[c]

    for c in target_cols:
        df_miss[c] = np.nan

    assert df_latent[fixed_cols].isna().sum().sum() == 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.model.to(device)
    model.model.eval()

    df_gen = model.impute(
        df_miss=df_miss,
        temperature=0.7,
        k=100,
        max_length=max_length,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    return df_gen[target_cols]




PV_COLS = [f"PV{i}MATH" for i in range(1, 11)]

df_pv = great_generate_block(
    df_real=df,
    df_latent=df_latent,
    fixed_cols=["PC_SMP"],
    target_cols=PV_COLS,
    model_dir=os.path.join(DATA_DIR, "great_pv_distilgpt2"),
    epochs=150,
    max_length=350,
)



SC_COLS = [
    "SC064Q01TA","SC064Q02TA","SC064Q03TA",
    "SC064Q04NA","SC064Q05WA","SC064Q06WA","SC064Q07WA",
]

df_sc = great_generate_block(
    df_real=df,
    df_latent=df_latent,
    fixed_cols=["PC_SMS"],
    target_cols=SC_COLS,
    model_dir=os.path.join(DATA_DIR, "great_sc_distilgpt2"),
    epochs=70,
    max_length=220,
)



ST_COLS = ["ST268Q01JA","ST268Q04JA","ST268Q07JA"]

df_st = great_generate_block(
    df_real=df,
    df_latent=df_latent,
    fixed_cols=["PC_SPI"],
    target_cols=ST_COLS,
    model_dir=os.path.join(DATA_DIR, "great_st_distilgpt2"),
    epochs=60,
    max_length=180,
)



df_final = pd.concat(
    [
        df_latent.reset_index(drop=True),
        df_pv.reset_index(drop=True),
        df_sc.reset_index(drop=True),
        df_st.reset_index(drop=True),
    ],
    axis=1,
)

df_final.to_csv(
    os.path.join(DATA_DIR, "synthetic_hybrid_predllm_great.csv"),
    index=False,
)
