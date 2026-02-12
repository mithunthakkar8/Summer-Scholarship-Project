import os
from pathlib import Path
from typing import Optional

import pandas as pd
from realtabformer import REaLTabFormer




# -----------------------------
# Paths / I/O
# -----------------------------
DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data"
INPUT_CSV = os.path.join(DATA_DIR, "df_coreSGP_with_keys.csv")
OUT_CSV = os.path.join(DATA_DIR, "synthetic_realtabformer_relational.csv")

MODEL_ROOT = Path(os.path.join(DATA_DIR, "realtabformer_relational"))
PARENT_DIR = MODEL_ROOT / "parent"
CHILD_DIR = MODEL_ROOT / "children"

PARENT_DIR.mkdir(parents=True, exist_ok=True)
CHILD_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Keys
# -----------------------------
JOIN_ON = "CNTSTUID"   # parent-child join key (must exist in all tables)
SCHOOL_KEY = "CNTSCHID"


# -----------------------------
# Helpers
# -----------------------------
def latest_experiment_dir(root: Path) -> Optional[Path]:
    """Return most recent id* subdir, or None."""
    if not root.exists():
        return None
    candidates = [p for p in root.glob("id*") if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=os.path.getmtime)[-1]


def fit_or_load_parent(parent_df: pd.DataFrame) -> tuple[REaLTabFormer, Path]:
    """
    Train/load the parent (tabular) model.
    Per README, the parent model should not include the join key.
    """
    exp_dir = latest_experiment_dir(PARENT_DIR)
    if exp_dir is not None:
        parent_model = REaLTabFormer.load_from_dir(path=str(exp_dir))
        return parent_model, exp_dir

    parent_model = REaLTabFormer(
        model_type="tabular",
        gradient_accumulation_steps=4,
        logging_steps=100
    )

    parent_model.fit(parent_df.drop(JOIN_ON, axis=1))
    parent_model.save(str(PARENT_DIR))

    return parent_model, exp_dir


def fit_or_load_child(child_name: str, child_df: pd.DataFrame, parent_path: Path) -> REaLTabFormer:
    """
    Train/load one relational child model, conditioned on the parent model.
    """
    this_child_root = CHILD_DIR / child_name
    this_child_root.mkdir(parents=True, exist_ok=True)

    exp_dir = latest_experiment_dir(this_child_root)
    if exp_dir is not None:
        return REaLTabFormer.load_from_dir(path=str(exp_dir))

    child_model = REaLTabFormer(
        model_type="relational",
        parent_realtabformer_path=parent_path,
        output_max_length=128,
        train_size=0.8
    )



    # docs: fit(df=child_df, in_df=parent_df, join_on=JOIN_ON)
    # df = table to be generated (decoder), in_df = conditioning table (encoder)
    child_model.fit(
        df=child_df,
        in_df=parent_df,     # uses outer-scope parent_df
        join_on=JOIN_ON
    )

    child_model.save(str(this_child_root))
    return child_model


def ensure_join_key(df: pd.DataFrame, name: str):
    if JOIN_ON not in df.columns:
        raise ValueError(f"{name} is missing join key column: {JOIN_ON}")


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(INPUT_CSV)

# -----------------------------
# Build parent + child tables
# -----------------------------
parent_cols = [
    JOIN_ON,
    "ST004D01T", "ST001D01T", "AGE", "IMMIG",
    "MISCED", "ESCS", "MCLSIZE", "SCHSIZE"
]

spi_cols = [
    JOIN_ON,
    "SC064Q01TA", "SC064Q02TA", "SC064Q03TA",
    "SC064Q04NA", "SC064Q05WA", "SC064Q06WA", "SC064Q07WA"
]
sms_cols = [
    JOIN_ON,
    "ST268Q01JA", "ST268Q04JA", "ST268Q07JA"
]
smp_cols = [JOIN_ON] + [f"PV{i}MATH" for i in range(1, 11)]

parent_df = df[parent_cols].copy()
spi_df = df[spi_cols].copy()
sms_df = df[sms_cols].copy()
smp_df = df[smp_cols].copy()



# -----------------------------
# Measurement-level corrections
# -----------------------------

ORDINAL_COLS = [
    "ST001D01T",          # grade (ordered)
    "ST268Q01JA",
    "ST268Q04JA",
    "ST268Q07JA"
]

NOMINAL_COLS = [
    "ST004D01T",          # gender
    "IMMIG"               # immigrant background
]

def apply_measurement_types(df: pd.DataFrame):
    """
    Enforce correct dtypes for REaLTabFormer:
    - ordinal → Int64
    - nominal → Int64
    - continuous → float (leave untouched)
    """
    for c in ORDINAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("Int64")

    for c in NOMINAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("Int64")

    return df


parent_df = apply_measurement_types(parent_df)
spi_df    = apply_measurement_types(spi_df)
sms_df    = apply_measurement_types(sms_df)
# smp_df intentionally untouched (continuous plausible values)


ensure_join_key(parent_df, "parent_df")
ensure_join_key(spi_df, "spi_df")
ensure_join_key(sms_df, "sms_df")
ensure_join_key(smp_df, "smp_df")

assert not any(
    c.startswith("SC064Q") and pd.api.types.is_integer_dtype(df[c])
    for c in df.columns
), "SC064Q variables must remain continuous"


# Optional: if JOIN_ON is numeric but read as float due to NaNs, fix dtype consistency
# parent_df[JOIN_ON] = parent_df[JOIN_ON].astype("int64")

# -----------------------------
# Train/load models
# -----------------------------
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

parent_model, parent_exp_dir = fit_or_load_parent(parent_df)

import realtabformer.rtf_analyze as _rtf_analyze
print("Sensitivity threshold function:", _rtf_analyze.SyntheticDataBench.compute_sensitivity_threshold)


# NOTE: parent_df is referenced inside fit_or_load_child via outer-scope variable.
# If you prefer, pass parent_df as an argument instead.
spi_model = fit_or_load_child("spi", spi_df, parent_exp_dir)
sms_model = fit_or_load_child("sms", sms_df, parent_exp_dir)
smp_model = fit_or_load_child("smp", smp_df, parent_exp_dir)

# -----------------------------
# Sampling (MULTIPLE REPS)
# -----------------------------
N_REPS = 5
n_samples = len(df)

for rep in range(1, N_REPS + 1):
    print(f"\n=== Generating replication {rep}/{N_REPS} ===")

    # -----------------------------
    # Parent samples
    # -----------------------------
    parent_samples = parent_model.sample(n_samples=n_samples)

    # README pattern: use index as unique id
    parent_samples.index.name = JOIN_ON
    parent_samples = parent_samples.reset_index()

    # -----------------------------
    # Child samples (conditioned)
    # -----------------------------
    parent_features = parent_samples.drop(JOIN_ON, axis=1)

    spi_samples = spi_model.sample(
        input_unique_ids=parent_samples[JOIN_ON],
        input_df=parent_features,
        gen_batch=256
    )

    sms_samples = sms_model.sample(
        input_unique_ids=parent_samples[JOIN_ON],
        input_df=parent_features,
        gen_batch=256
    )

    smp_samples = smp_model.sample(
        input_unique_ids=parent_samples[JOIN_ON],
        input_df=parent_features,
        gen_batch=256
    )

    # Defensive JOIN_ON restore (version-safe)
    for child_df_out in [spi_samples, sms_samples, smp_samples]:
        if JOIN_ON not in child_df_out.columns:
            child_df_out[JOIN_ON] = parent_samples[JOIN_ON].values

    # -----------------------------
    # Merge wide table
    # -----------------------------
    synthetic_full = (
        parent_samples
        .merge(spi_samples, on=JOIN_ON, how="left")
        .merge(sms_samples, on=JOIN_ON, how="left")
        .merge(smp_samples, on=JOIN_ON, how="left")
    )

    # Tag replication
    synthetic_full["rep_id"] = rep

    # -----------------------------
    # Save
    # -----------------------------
    out_csv_rep = os.path.join(
        DATA_DIR,
        f"synthetic_realtabformer_relational_rep{rep}.csv"
    )

    synthetic_full.to_csv(out_csv_rep, index=False)
    print(f"✔ Saved replication {rep}: {out_csv_rep}  shape={synthetic_full.shape}")
