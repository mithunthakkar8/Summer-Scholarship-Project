"""
synthetic_sem_with_great_shortnames.py

Train a GReaT model on PISA SEM data and generate synthetic data
conditioned on Latent Factors: SMP, SMS, SPI.

Uses SHORT column headers for non-latent variables to keep prompts small.
"""

import os
import numpy as np
if not hasattr(np, "float"):
    np.float = float  # Fix removed numpy alias for GReaT compatibility

import pandas as pd
from be_great import GReaT  # High-level API


# -----------------------------
# CONFIG
# -----------------------------

DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data/"
INPUT_CSV = os.path.join(DATA_DIR, "df_core_fullnames_with_latents.csv")

os.makedirs(DATA_DIR, exist_ok=True)

LLM_NAME = "distilgpt2"
EPOCHS = 5
BATCH_SIZE = 32
FP16 = True  # you are on GPU

# Latent factor names as they appear in the CSV
SMP_COL = "Latent Factor: Student Math Performance (SMP)"
SMS_COL = "Latent Factor: Student Math self-efficacy (SMS)"
SPI_COL = "Latent Factor: School-level Parental Involvement (SPI)"

CONDITIONAL_COLS = [SMP_COL, SMS_COL, SPI_COL]


# -----------------------------
# 1. Load + shorten column names
# -----------------------------

import re
import pandas as pd
import os

def make_safe_shortname(col: str) -> str:
    """
    Convert a column name into a safe GPT-2 compatible version:
    - lowercase
    - replace unsafe chars with underscore
    - collapse multiple underscores
    - strip leading/trailing underscores
    """
    # lowercase
    s = col.lower()

    # replace all unsafe characters with underscore
    # keep only a-z, 0-9, underscore
    s = re.sub(r'[^a-z0-9]', '_', s)

    # collapse multiple underscores
    s = re.sub(r'_+', '_', s)

    # strip leading/trailing _
    s = s.strip('_')

    # ensure non-empty
    if s == "":
        s = "col"

    return s


def load_into_df(path: str, shorten=True) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded real data from {path}")
    print("Shape:", df.shape)

    # sanity check: latent columns exist
    for col in CONDITIONAL_COLS:
        if col not in df.columns:
            raise ValueError(f"Expected conditional column not found: {col}")

    if not shorten:
        return df

    original_cols = df.columns.tolist()

    short_cols = []
    mapping_rows = []
    used_short_names = set()

    for col in original_cols:

        # latent columns keep full long names (your choice)
        if col in CONDITIONAL_COLS:
            short = col

        else:
            # safe short name
            short = make_safe_shortname(col)

            # ensure uniqueness
            base = short
            counter = 1
            while short in used_short_names:
                counter += 1
                short = f"{base}_{counter}"

        used_short_names.add(short)
        short_cols.append(short)
        mapping_rows.append({"short_name": short, "full_name": col})

    # apply safe short names
    df.columns = short_cols

    # save mapping
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_path = os.path.join(OUT_DIR, "short_to_fullname_mapping.csv")
    mapping_df.to_csv(mapping_path, index=False)

    print(f"✔ Saved short→full name mapping to: {mapping_path}")
    print("\nColumns after safe shortening:")
    print(df.columns.tolist())

    return df




# -----------------------------
# 2. Train GReaT model
# -----------------------------

def train_great(df: pd.DataFrame) -> GReaT:
    """
    Train a GReaT model on the full SEM dataset with SHORT column names.
    Conditioning is done later via impute() using the latent columns
    (which still have long names).
    """
    print("\n=== Training GReaT model (short column names) ===\n")
    model = GReaT(
        llm=LLM_NAME,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        fp16=FP16,
        dataloader_num_workers=4,
        report_to=[],
        float_precision=3,
    )
    model.fit(df)
    print("\n✔ Training complete.")

    save_dir = os.path.join(OUT_DIR, "great_sem_model_short")
    model.save(save_dir)
    print(f"✔ Model saved to: {save_dir}")

    return model


# -----------------------------
# 3. Build conditional seeds
# -----------------------------

def build_condition_seed(
    df_real: pd.DataFrame,
    n_samples: int,
    conditional_cols: list[str],
    strategy: str = "sample_from_real",
) -> pd.DataFrame:
    """
    Create a DataFrame with:
      - conditional_cols filled (SMP, SMS, SPI)
      - all other columns = NaN
    """
    all_cols = df_real.columns.tolist()
    seed = pd.DataFrame(np.nan, index=range(n_samples), columns=all_cols)

    if strategy == "sample_from_real":
        idx = np.random.choice(len(df_real), size=n_samples, replace=True)
        sampled = df_real.iloc[idx][conditional_cols].reset_index(drop=True)
        for col in conditional_cols:
            seed[col] = sampled[col]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return seed


# -----------------------------
# 4. Generate synthetic data
# -----------------------------

def generate_synthetic_sem(
    model: GReaT,
    df_real: pd.DataFrame,
    n_samples: int,
    conditional_cols: list[str],
    max_length: int = 1024,
) -> pd.DataFrame:
    """
    Generate synthetic SEM data conditioned on given latent columns,
    using SHORT column names for non-latent variables.
    """
    print(f"\n=== Generating {n_samples} synthetic samples "
          f"conditioned on {conditional_cols} ===")

    seed_df = build_condition_seed(
        df_real=df_real,
        n_samples=n_samples,
        conditional_cols=conditional_cols,
        strategy="sample_from_real",
    )

    print("\n=== Seed Columns (in order) ===")
    for c in seed_df.columns:
        print(repr(c))

    synthetic_df = model.impute(seed_df, max_length=max_length)

    print("✔ Synthetic data generated.")
    print("Synthetic shape:", synthetic_df.shape)
    return synthetic_df


# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":
    np.random.seed(42)

    # 1. Load real SEM data (with latent factors, full names) and shorten
    df_real_short = load_into_df(INPUT_CSV, shorten=True)

    # 2. Train GReaT on shortened headers
    save_dir = os.path.join(OUT_DIR, "great_sem_model_short")

    if os.path.exists(save_dir):
        print("Loading existing GReaT model...")
        model = GReaT(llm=LLM_NAME)
        model = GReaT.load_from_dir(save_dir)
    else:
        print("Training new GReaT model...")
        model = train_great(df_real_short)


    # 3. Generate synthetic SEM data, conditioning on SMP/SMS/SPI
    N_SYNTH = 1000
    synthetic_short = generate_synthetic_sem(
        model=model,
        df_real=df_real_short,
        n_samples=N_SYNTH,
        conditional_cols=CONDITIONAL_COLS,
        max_length=1024,
    )

    # 4. Save output (short-header version)
    out_path = os.path.join(OUT_DIR, "synthetic_sem_conditioned_on_latents_shortnames.csv")
    synthetic_short.to_csv(out_path, index=False)
    print(f"\n🎉 Done. Saved synthetic data (short headers) to:\n    {out_path}")
    print("Use short_to_fullname_mapping.csv to restore full question names if needed.")
