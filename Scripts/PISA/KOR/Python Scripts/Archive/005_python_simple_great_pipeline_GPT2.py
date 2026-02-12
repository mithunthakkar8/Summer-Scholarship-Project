import os
import json
import numpy as np
import pandas as pd
from be_great import GReaT

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data"
INPUT_CSV = os.path.join(DATA_DIR, "sem_full_dataset_raw_plus_selected_latent_scores_.csv")

ACTIVE_COMBO_KEY = "D"
CFG = {
    "name": "Best_Overall_Default",
    "llm": "gpt2",
    "learning_rate": 1e-4,
    "epochs": 6,
    "batch_size": 8,
    "temperature": 0.8,
    "top_k": 100
}

COMBO_DIR = os.path.join(DATA_DIR, f"COMBO_{ACTIVE_COMBO_KEY}_{CFG['name']}")
os.makedirs(COMBO_DIR, exist_ok=True)

with open(os.path.join(COMBO_DIR, "combo_config.json"), "w") as f:
    json.dump(CFG, f, indent=2)

# ------------------------------------------------------------------
# LATENT COLUMNS
# ------------------------------------------------------------------

SMP_COL = "latent_score_sem_student_math_performance"
SMS_COL = "latent_score_sem_student_math_self_efficacy"
SPI_COL = "latent_score_sem_school_based_parental_involvement"

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

TRAIN_COLS = [
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
    "math_class_size",
    # measurement items
    *[f"plausible_value_{i}_in_mathematics" for i in range(1, 11)],
    "agree_disagree_mathematics_is_one_of_my_favourite_subjects",
    "agree_disagree_mathematics_is_easy_for_me",
    "agree_disagree_i_want_to_do_well_in_my_mathematics_class",
    "proportion_parent_guardians_who_discussed_their_child_s_behaviour_with_a_teacher_on_the_parents_or_guardians_own_initiative",
    "proportion_parent_guardians_who_discussed_their_child_s_progress_with_a_teacher_on_the_parents_or_guardians_own_initiative",
]

df_train = df[TRAIN_COLS].copy()

# ------------------------------------------------------------------
# TRAIN MODEL (SMP ONLY)
# ------------------------------------------------------------------

model = GReaT(
    llm=CFG["llm"],
    batch_size=CFG["batch_size"],
    epochs=CFG["epochs"],
    learning_rate=CFG["learning_rate"],
    fp16=True,
    report_to=[],
)

print("Training GReaT (conditional = SMP)")
model.fit(df_train, conditional_col=SMP_COL)

MODEL_DIR = os.path.join(COMBO_DIR, "model")
model.save(MODEL_DIR)

# ------------------------------------------------------------------
# GENERATION (SMP + SMS + SPI)
# ------------------------------------------------------------------

N = 1000

seed = pd.DataFrame(np.nan, index=range(N), columns=df_train.columns)

# sample coherent latent triples from real data
idx = np.random.choice(len(df_train), size=N, replace=True)
seed[SMP_COL] = df_train.iloc[idx][SMP_COL].values
seed[SMS_COL] = df_train.iloc[idx][SMS_COL].values
seed[SPI_COL] = df_train.iloc[idx][SPI_COL].values

print("Generating synthetic SEM data (conditioned on SMP + SMS + SPI)")
synthetic = model.impute(
    seed,
    temperature=CFG["temperature"],
    k=CFG["top_k"],
    max_length=1024,
)

OUT_PATH = os.path.join(COMBO_DIR, "synthetic_sem_full.csv")
synthetic.to_csv(OUT_PATH, index=False)

print(f"✔ Synthetic SEM dataset saved to {OUT_PATH}")
