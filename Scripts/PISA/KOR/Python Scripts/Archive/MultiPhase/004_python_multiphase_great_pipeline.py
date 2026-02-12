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
import os
import json
from typing import Optional


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================

COMBOS = {

    "TEST": dict(
        name="Smoke_Test_distilgpt2",
        llm="distilgpt2",

        use_lora=False,  # distilgpt2 does not benefit meaningfully from LoRA

        learning_rate=5e-5,
        epochs=2,        # very small
        batch_size=16,   # fits everywhere

        shuffle_fields=False,   # keep deterministic for debugging

        temperature=0.8,  # slightly higher to detect sampling issues
        top_p=0.95,
        top_k=50
    ),


    "A": dict(
        name="Structural_Coherence_Optimized",
        llm="meta-llama/Meta-Llama-3-3B-Instruct",

        use_lora=True,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,

        learning_rate=5e-5,
        epochs=10,
        batch_size=4,

        shuffle_fields=True,

        temperature=0.5,
        top_p=0.9,
        top_k=50
    ),

    "B": dict(
        name="Distribution_Fidelity_Optimized",
        llm="microsoft/Phi-3-mini-4k-instruct",

        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,

        learning_rate=2e-5,
        epochs=12,
        batch_size=6,

        shuffle_fields=False,

        temperature=0.4,
        top_p=0.95,
        top_k=40
    ),

    "C": dict(
        name="High_Diversity_Synthetic_Universe",
        llm="gpt2-large",

        use_lora=False,

        learning_rate=1e-4,
        epochs=20,
        batch_size=8,

        shuffle_fields=False,

        temperature=0.9,
        top_p=0.95,
        top_k=100
    ),

    "D": dict(
        name="Best_Overall_Default",
        llm="meta-llama/Meta-Llama-3-1B-Instruct",

        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,

        learning_rate=5e-5,
        epochs=15,
        batch_size=4,

        shuffle_fields=True,

        temperature=0.6,
        top_p=0.9,
        top_k=50
    ),

    "E": dict(
        name="Mediation_Sensitive",
        llm="microsoft/Phi-3-mini-4k-instruct",

        use_lora=True,
        lora_r=32,
        lora_alpha=32,
        lora_dropout=0.05,

        learning_rate=2e-5,
        epochs=10,
        batch_size=4,

        shuffle_fields=True,

        temperature=0.5,
        top_p=0.9,
        top_k=40
    ),
}


def preprocess_df(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()

    if cfg.get("shuffle_fields", False):
        cols = df.columns.tolist()
        np.random.shuffle(cols)
        df = df[cols]

    return df



def load_into_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded real data from {path}")
    print("Shape:", df.shape)

    return df

# -----------------------------
# 2. Train GReaT model
# -----------------------------

def train_great(df: pd.DataFrame, phase: str, cfg: dict, conditional_col = None) -> GReaT:

    print(f"\n=== Training GReaT | {CFG['name']} | {phase} ===\n")

    great_kwargs = dict(
        llm=cfg["llm"],
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        fp16=True,
        dataloader_num_workers=4,
        report_to=[],
        float_precision=3,
    )

    # Only add LoRA args if explicitly enabled
    if cfg.get("use_lora", False):
        great_kwargs.update(
            use_lora=True,
            lora_r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
        )

    model = GReaT(**great_kwargs)


    model.fit(df, conditional_col)

    save_dir = os.path.join(COMBO_DIR, "models", phase)

    if os.path.exists(save_dir):
        raise RuntimeError(
            f"Model directory already exists for this run:\n{save_dir}\n"
            f"Refusing to overwrite or reuse an existing model."
        )

    os.makedirs(save_dir, exist_ok=True)
    model.save(save_dir)
    print(f"✔ Model saved to: {save_dir}")

    return model


# -----------------------------
# 3. Build conditional seeds
# -----------------------------

def build_condition_seed(
    df_real: pd.DataFrame,
    n_samples: int,
    conditional_col: str,
    conditional_values: Optional[np.ndarray] = None,
):
    """
    Build a seed df with:
    - conditional_col filled (either from given values, or sampled from real)
    - all other columns = NaN
    """
    all_cols = df_real.columns.tolist()
    seed = pd.DataFrame(np.nan, index=range(n_samples), columns=all_cols)

    if conditional_values is not None:
        if len(conditional_values) != n_samples:
            raise ValueError(
                f"conditional_values length ({len(conditional_values)}) "
                f"must match n_samples ({n_samples})"
            )
        seed[conditional_col] = conditional_values
    else:
        idx = np.random.choice(len(df_real), size=n_samples, replace=True)
        seed[conditional_col] = df_real.iloc[idx][conditional_col].values

    return seed

# -----------------------------
# 4. Generate synthetic data
# -----------------------------

def generate_synthetic_sem(
    model: GReaT,
    df_real: pd.DataFrame,
    n_samples: int,
    conditional_col: str,
    max_length: int = 512,
    conditional_values: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Generate synthetic SEM data conditioned on a single column.

    If conditional_values is provided, those values are used as the condition.
    Otherwise, conditional values are sampled from df_real.
    """
    print(f"\n=== Generating {n_samples} synthetic samples "
          f"conditioned on {conditional_col} ===")

    seed_df = build_condition_seed(
        df_real=df_real,
        n_samples=n_samples,
        conditional_col=conditional_col,
        conditional_values=conditional_values,
    )

    print("\n=== Seed Columns (in order) ===")
    for c in seed_df.columns:
        print(repr(c))

    synthetic_df = model.impute(
        seed_df,
        temperature=CFG["temperature"],
        k=CFG["top_k"],          # <-- map top_k → k
        max_length=max_length,
    )




    print("✔ Synthetic data generated.")
    print("Synthetic shape:", synthetic_df.shape)
    return synthetic_df



# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":

    # -----------------------------
    # CONFIG
    # -----------------------------

    DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data/"
    # DATA_DIR = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Datasets\PISA 2022"
    INPUT_CSV = os.path.join(DATA_DIR, "sem_full_dataset_raw_plus_selected_latent_scores_.csv")
    os.makedirs(DATA_DIR, exist_ok=True)

    ACTIVE_COMBO_KEY = "TEST"   # Test | A | B | C | D | E
    CFG = COMBOS[ACTIVE_COMBO_KEY]
    COMBO_NAME = f"COMBO_{ACTIVE_COMBO_KEY}_{CFG['name']}"

    COMBO_DIR = os.path.join(DATA_DIR, COMBO_NAME)
    os.makedirs(COMBO_DIR, exist_ok=True)


    # =============================
    # SAVE COMBO CONFIG (ONCE)
    # =============================
    with open(os.path.join(COMBO_DIR, "combo_config.json"), "w") as f:
        json.dump(CFG, f, indent=2)


    # Latent factor names as they appear in the CSV
    SMP_COL = "latent_score_sem_student_math_performance"
    SMS_COL = "latent_score_sem_student_math_self_efficacy"
    SPI_COL = "latent_score_sem_school_based_parental_involvement"

    CONDITIONAL_COL = None

    # -----------------------------
    # 1. Load + shorten column names
    # -----------------------------

    np.random.seed(42)

    # 1. Load real SEM data (with latent factors, full names) and shorten
    df_real_short = load_into_df(INPUT_CSV)

    # --------------------
    # PHASE 1
    # --------------------
    CONDITIONAL_COL = SMP_COL

    cols_phase1 = [
        SMP_COL,
        SMS_COL,
        SPI_COL,
        "country_code_3_character",
        "student_international_grade_derived",
        "student_standardized_gender",
        "students_age",
        "index_on_immigrant_background_oecd_definition",
        "mother_s_level_of_education_isced",
        "index_of_economic_social_and_cultural_status",
        "school_size_sum",
        "math_class_size"
    ]


    df_phase1 = preprocess_df(
        df_real_short[cols_phase1],
        CFG
    )

    model_phase1 = train_great(
        df_phase1,
        phase="SMP_Structural_phase1",
        cfg=CFG,
        conditional_col=CONDITIONAL_COL
    )


    phase1_path = os.path.join(
        COMBO_DIR,
        f"synthetic_phase1_{COMBO_NAME}.csv"
    )


    if os.path.exists(phase1_path):
        print(f"✔ Synthetic data already exists at: {phase1_path} — loading instead of regenerating.")
        syn_phase1 = pd.read_csv(phase1_path)
    else:
        print("Generating new synthetic Phase 1 data...")
        syn_phase1 = generate_synthetic_sem(
            model=model_phase1,
            df_real=df_phase1,
            n_samples=1000,
            conditional_col=CONDITIONAL_COL,
        )
        syn_phase1.to_csv(phase1_path, index=False)
        print(f"✔ Synthetic Phase 1 data saved to: {phase1_path}")


    # --------------------
    # PHASE 2: SMP -> PVs
    # --------------------
    CONDITIONAL_COL = SMP_COL

    cols_phase2 = [
        SMP_COL,
        "plausible_value_1_in_mathematics",
        "plausible_value_2_in_mathematics",
        "plausible_value_3_in_mathematics",
        "plausible_value_4_in_mathematics",
        "plausible_value_5_in_mathematics",
        "plausible_value_6_in_mathematics",
        "plausible_value_7_in_mathematics",
        "plausible_value_8_in_mathematics",
        "plausible_value_9_in_mathematics",
        "plausible_value_10_in_mathematics"
    ]


    df_phase2 = preprocess_df(
        df_real_short[cols_phase2],
        CFG
    )

    model_phase2 = train_great(
        df_phase2,
        phase="SMP_Measurement_phase2",
        cfg=CFG,
        conditional_col=CONDITIONAL_COL
    )


    phase2_path = os.path.join(
        COMBO_DIR,
        f"synthetic_phase2_PVs_{COMBO_NAME}.csv"
    )


    if os.path.exists(phase2_path):
        print(f"✔ Synthetic Phase 2 data already exists at: {phase2_path} — loading instead of regenerating.")
        syn_phase2 = pd.read_csv(phase2_path)
    else:
        print("Generating new synthetic Phase 2 PVs data...")
        syn_phase2 = generate_synthetic_sem(
            model=model_phase2,
            df_real=df_phase2,
            n_samples=1000,
            conditional_col=CONDITIONAL_COL,
        )
        syn_phase2.to_csv(phase2_path, index=False)
        print(f"✔ Synthetic Phase 2 PV data saved to: {phase2_path}")

    # --------------------
    # PHASE 3: SMS -> ST268 items
    # --------------------
    CONDITIONAL_COL = SMS_COL

    cols_phase3 = [
        SMS_COL,
        "agree_disagree_mathematics_is_one_of_my_favourite_subjects",
        "agree_disagree_mathematics_is_easy_for_me",
        "agree_disagree_i_want_to_do_well_in_my_mathematics_class"
    ]


    df_phase3 = df_real_short[cols_phase3]

    model_phase3 = train_great(
        df_phase3,
        phase="SMS_Measurement_phase3",
        cfg=CFG,
        conditional_col=SMS_COL
    )


    phase3_path = os.path.join(
        COMBO_DIR,
        f"synthetic_phase3_SMS_{COMBO_NAME}.csv"
    )


    if os.path.exists(phase3_path):
        print(f"✔ Synthetic Phase 3 data already exists at: {phase3_path} — loading instead of regenerating.")
        syn_phase3 = pd.read_csv(phase3_path)
    else:
        print("Generating new synthetic Phase 3 (SMS items) data...")
        n_samples_phase3 = len(syn_phase1)  # link to Phase 1

        syn_phase3 = generate_synthetic_sem(
            model=model_phase3,
            df_real=df_phase3,
            n_samples=n_samples_phase3,
            conditional_col=CONDITIONAL_COL,          # SMS_COL
            conditional_values=syn_phase1[SMS_COL].values,  # <<< synthetic SMS from Phase 1
        )

        syn_phase3.to_csv(phase3_path, index=False)
        print(f"✔ Synthetic Phase 3 SMS data saved to: {phase3_path}")



    # --------------------
    # PHASE 4: SPI -> SC064 items
    # --------------------
    CONDITIONAL_COL = SPI_COL

    cols_phase4 = [
        SPI_COL,
        "proportion_parent_guardians_who_discussed_their_child_s_behaviour_with_a_teacher_on_the_parents_or_guardians_own_initiative",
        "proportion_parent_guardians_who_discussed_their_child_s_behaviour_on_the_initiative_of_one_of_their_child_s_teachers",
        "proportion_parent_guardians_who_discussed_their_child_s_progress_with_a_teacher_on_the_parents_or_guardians_own_initiative",
        "proportion_parent_guardians_who_discussed_their_child_s_progress_on_the_initiative_of_one_of_their_child_s_teachers",
        "proportion_parent_guardians_who_volunteered_in_physical_or_extra_curricular_activities_e_g_building_maintenance_carpentry_gardening_or_yard_work_school_play_sports_field_trip",
        "proportion_parent_guardians_who_participated_in_local_school_government_e_g_parent_council_or_school_management_committee",
        "proportion_parent_guardians_who_assisted_in_fundraising_for_the_school"
    ]


    df_phase4 = df_real_short[cols_phase4]

    model_phase4 = train_great(
        df_phase4,
        phase="SPI_Measurement_phase4",
        cfg=CFG,
        conditional_col=SPI_COL
    )


    phase4_path = os.path.join(
        COMBO_DIR,
        f"synthetic_phase4_SPI_{COMBO_NAME}.csv"
    )


    if os.path.exists(phase4_path):
        print(f"✔ Synthetic Phase 4 data already exists at: {phase4_path} — loading instead of regenerating.")
        syn_phase4 = pd.read_csv(phase4_path)
    else:
        print("Generating new synthetic Phase 4 (SPI items) data...")
        n_samples_phase4 = len(syn_phase1)  # link to Phase 1

        syn_phase4 = generate_synthetic_sem(
            model=model_phase4,
            df_real=df_phase4,
            n_samples=n_samples_phase4,
            conditional_col=CONDITIONAL_COL,          # SPI_COL
            conditional_values=syn_phase1[SPI_COL].values,  # <<< synthetic SPI from Phase 1
        )

        syn_phase4.to_csv(phase4_path, index=False)
        print(f"✔ Synthetic Phase 4 SPI data saved to: {phase4_path}")


    print(f"\n🎉 Done. Saved synthetic data (short headers) to:\n    {DATA_DIR}")
    print("Use short_to_fullname_mapping.csv to restore full question names if needed.")


    # --------------------
    # PHASE 5: MERGE ALL SYNTHETIC BLOCKS
    # --------------------

    # Reset index to ensure merge consistency
    syn_phase1 = syn_phase1.reset_index(drop=True)
    syn_phase2 = syn_phase2.reset_index(drop=True)
    syn_phase3 = syn_phase3.reset_index(drop=True)
    syn_phase4 = syn_phase4.reset_index(drop=True)

    # Merge all synthetic blocks horizontally
    synthetic_full = pd.concat(
        [syn_phase1, syn_phase2.drop(columns=[SMP_COL], errors='ignore'),
                     syn_phase3.drop(columns=[SMS_COL], errors='ignore'),
                     syn_phase4.drop(columns=[SPI_COL], errors='ignore')],
        axis=1
    )

    # Save unified CSV
    unified_path = os.path.join(
        COMBO_DIR,
        f"synthetic_sem_full_{COMBO_NAME}.csv"
    )

    synthetic_full.to_csv(unified_path, index=False)

    print(f"\n✔ Unified synthetic dataset saved to: {unified_path}")
