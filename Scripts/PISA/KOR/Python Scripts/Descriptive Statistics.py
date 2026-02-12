import os
import glob
import pandas as pd
import numpy as np

# =====================================================
# PATHS
# =====================================================
TECHNIQUE_PATHS = {
    "REAL": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Datasets\PISA 2022",

    "GReaT_DistilGPT2": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\GReaT\DistilGPT2",
    "GReaT_GPT2":       r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\GReaT\GPT2",

    "Tabula_DistilGPT2": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\Tabula\DistilGPT2",
    "Tabula_GPT2":       r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\Tabula\GPT2",

    "TapTap_DistilGPT2": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\TapTap\DistilGPT2",
    "TapTap_GPT2":       r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\TapTap\GPT2",

    "PredLLM_DistilGPT2": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\PredLLM\DistilGPT2",
    "PredLLM_GPT2":       r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\PredLLM\GPT2",

    "TabDiff": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\TabDiff",
    "REaLTabFormer": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\REaLTabFormer",
}

# =====================================================
# HELPERS
# =====================================================
def col_exists(df, col):
    return col in df.columns

def safe_mean(df, col):
    return df[col].mean() if col_exists(df, col) else np.nan

def safe_std(df, col):
    return df[col].std() if col_exists(df, col) else np.nan

def safe_min(df, col):
    return df[col].min() if col_exists(df, col) else np.nan

def safe_max(df, col):
    return df[col].max() if col_exists(df, col) else np.nan

def safe_count(mask):
    return mask.sum() if mask is not None else np.nan

def safe_pct(mask):
    return mask.mean() * 100 if mask is not None else np.nan


def load_real_csv(real_dir):
    real_csv = os.path.join(real_dir, "df_coreSGP.csv")
    if not os.path.exists(real_csv):
        raise FileNotFoundError(f"REAL file not found: {real_csv}")
    return [real_csv]  # return as list for uniform handling

LIKERT_ITEMS = [
    "ST268Q01JA", "ST268Q04JA", "ST268Q07JA",
    "SC064Q05WA", "SC064Q06WA", "SC064Q01TA",
    "SC064Q02TA", "SC064Q04NA", "SC064Q03TA",
    "SC064Q07WA",
]

def likert_stats(df, col):
    if col not in df.columns:
        return {
            f"{col}_mean": np.nan,
            f"{col}_sd": np.nan,
            f"{col}_min": np.nan,
            f"{col}_max": np.nan,
            f"{col}_missing_pct": np.nan,
        }

    s = pd.to_numeric(df[col], errors="coerce")
    return {
        f"{col}_mean": s.mean(),
        f"{col}_sd": s.std(),
        f"{col}_min": s.min(),
        f"{col}_max": s.max(),
        f"{col}_missing_pct": s.isna().mean() * 100,
    }


def load_all_csvs(directory):
    csvs = [
        f for f in glob.glob(os.path.join(directory, "*.csv"))
        if "correction" not in os.path.basename(f).lower()
    ]
    if len(csvs) == 0:
        raise FileNotFoundError(f"No CSVs found in {directory}")
    return csvs


def compute_math_score(df):
    pv_cols = [c for c in df.columns if c.startswith("PV") and "MATH" in c]
    if len(pv_cols) == 0:
        raise ValueError("No PV*MATH columns found")
    return df[pv_cols].mean(axis=1)


def compute_distribution_stats(df):
    # -----------------
    # Math score (optional)
    # -----------------
    try:
        math_score = compute_math_score(df)
        math_mean = math_score.mean()
        math_sd = math_score.std()
    except Exception:
        math_mean = np.nan
        math_sd = np.nan

    # -----------------
    # Gender
    # -----------------
    if col_exists(df, "ST004D01T"):
        gender = pd.to_numeric(df["ST004D01T"], errors="coerce").round()
        female_mask = gender == 1
        male_mask   = gender == 2
    else:
        female_mask = None
        male_mask = None

    # -----------------
    # ISCED (mother education)
    # -----------------
    if col_exists(df, "MISCED"):
        misced = pd.to_numeric(df["MISCED"], errors="coerce")
        isced_le = misced <= 2
        isced_ge = misced >= 3
    else:
        isced_le = None
        isced_ge = None

    
    # -----------------
    # Immigration status
    # -----------------
    if col_exists(df, "IMMIG"):
        immig = round(pd.to_numeric(df["IMMIG"], errors="coerce"),0)
        immig_1 = immig == 1
        immig_2 = immig == 2
    else:
        immig_1 = None
        immig_2 = None


    # -----------------
    # Grade
    # -----------------
    grade_col = "ST001D01T" if col_exists(df, "ST001D01T") else None


    stats = {
        # -----------------
        # Sample size
        # -----------------
        "N": len(df),

        # -----------------
        # Age
        # -----------------
        "Age_mean": safe_mean(df, "AGE"),
        "Age_sd": safe_std(df, "AGE"),

        # -----------------
        # Gender
        # -----------------
        "Female_n": safe_count(female_mask),
        "Male_n": safe_count(male_mask),
        "Female_%": safe_pct(female_mask),
        "Male_%": safe_pct(male_mask),

        # -----------------
        # ESCS
        # -----------------
        "ESCS_mean": safe_mean(df, "ESCS"),
        "ESCS_sd": safe_std(df, "ESCS"),

        # -----------------
        # Math
        # -----------------
        "Math_mean": math_mean,
        "Math_sd": math_sd,

        # -----------------
        # Grade
        # -----------------
        "Grade_min": safe_min(df, grade_col) if grade_col else np.nan,
        "Grade_max": safe_max(df, grade_col) if grade_col else np.nan,

        # -----------------
        # ISCED
        # -----------------
        "ISCED_le_2_n": safe_count(isced_le),
        "ISCED_ge_3_n": safe_count(isced_ge),
        "ISCED_le_2_%": safe_pct(isced_le),
        "ISCED_ge_3_%": safe_pct(isced_ge),

        # -----------------
        # Class size
        # -----------------
        "ClassSize_mean": safe_mean(df, "MCLSIZE"),
        "ClassSize_sd": safe_std(df, "MCLSIZE"),

        
        # -----------------
        # Immigration
        # -----------------
        "IMMIG_1_n": safe_count(immig_1),
        "IMMIG_2_n": safe_count(immig_2),
        "IMMIG_1_%": safe_pct(immig_1),
        "IMMIG_2_%": safe_pct(immig_2),

    }
    # -----------------
    # Likert items
    # -----------------
    for col in LIKERT_ITEMS:
        stats.update(likert_stats(df, col))

    return stats

    


def aggregate_stats_across_csvs(csv_paths):
    per_run_stats = []

    for csv in csv_paths:
        df = pd.read_csv(csv)
        per_run_stats.append(compute_distribution_stats(df))

    stats_df = pd.DataFrame(per_run_stats)

    # Mean across runs (NOT pooling rows)
    return stats_df.mean().to_dict()


# =====================================================
# MAIN
# =====================================================
rows = []

for name, path in TECHNIQUE_PATHS.items():
    print(f"Processing: {name}")

    if name == "REAL":
        csvs = load_real_csv(path)
        print("  Using df_coreSGP.csv only")
    else:
        csvs = load_all_csvs(path)
        print(f"  Found {len(csvs)} CSVs")

    agg_stats = aggregate_stats_across_csvs(csvs)
    agg_stats["Technique"] = name
    rows.append(agg_stats)

comparison_df = pd.DataFrame(rows).set_index("Technique")
COUNT_COLS = [
    "Female_n", "Male_n",
    "ISCED_le_2_n", "ISCED_ge_3_n",
    "IMMIG_1_n", "IMMIG_2_n",
    "N"
]

for c in COUNT_COLS:
    comparison_df[c] = comparison_df[c].round().astype("Int64")

CATEGORICAL_MINMAX_COLS = [
    "Grade_min", "Grade_max",
    "ST268Q01JA_min", "ST268Q01JA_max",
    "ST268Q04JA_min", "ST268Q04JA_max",
    "ST268Q07JA_min", "ST268Q07JA_max",
    "SC064Q05WA_min", "SC064Q05WA_max",
    "SC064Q06WA_min", "SC064Q06WA_max",
    "SC064Q01TA_min", "SC064Q01TA_max",
    "SC064Q02TA_min", "SC064Q02TA_max",
    "SC064Q04NA_min", "SC064Q04NA_max",
    "SC064Q03TA_min", "SC064Q03TA_max",
    "SC064Q07WA_min", "SC064Q07WA_max",
]

for c in CATEGORICAL_MINMAX_COLS:
    comparison_df[c] = comparison_df[c].round().astype("Int64")


out_path = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\real_vs_synthetic_distribution_comparison.csv"
comparison_df.round(3).to_csv(out_path)

print("Saved:", out_path)
print(comparison_df.round(3))
