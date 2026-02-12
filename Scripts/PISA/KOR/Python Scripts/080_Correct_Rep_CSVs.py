import os
import glob
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

TECHNIQUE_PATHS = {

    # "GReaT_DistilGPT2": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\GReaT\DistilGPT2",
    # "GReaT_GPT2":       r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\GReaT\GPT2",

    # "Tabula_DistilGPT2": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\Tabula\DistilGPT2",
    # "Tabula_GPT2":       r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\Tabula\GPT2",

    "TapTap_DistilGPT2": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\TapTap\DistilGPT2",
    "TapTap_GPT2":       r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\TapTap\GPT2",

    # "PredLLM_DistilGPT2": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\PredLLM\DistilGPT2",
    # "PredLLM_GPT2":       r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\PredLLM\GPT2",

    # "TabDiff": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\TabDiff",
    # "REaLTabFormer": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\SGP\REaLTabFormer",
}

# ============================================================
# RULES
# ============================================================

LIKERT_4_COLS = ["ST268Q01JA", "ST268Q04JA", "ST268Q07JA"]

PROP_0_100_COLS = [
    "SC064Q05WA", "SC064Q06WA", "SC064Q01TA", "SC064Q02TA",
    "SC064Q04NA", "SC064Q03TA", "SC064Q07WA"
]

CATEGORICAL_RULES = {
    "ST004D01T": [1, 2],           # gender
    "IMMIG": [1, 2, 3],
    "MISCED": list(range(1, 11)),  # 1–10
}

CONTINUOUS_CLIP_RULES = {
    "AGE": (15.0, 17.0),
    "ST001D01T": (7, 12),      # grade
    "SCHSIZE": (1, 30000),
    "MCLSIZE": (5, 60),
}

# ESCS and PV*MATH intentionally excluded


# ============================================================
# UTILITIES
# ============================================================

import shutil

def move_rep_files_to_old_reps(base_path):

    old_reps_dir = os.path.join(base_path, "old reps")
    os.makedirs(old_reps_dir, exist_ok=True)

    rep_files = [
        f for f in glob.glob(os.path.join(base_path, "*.csv"))
        if (
            ("rep" in os.path.basename(f).lower() or "samples" in os.path.basename(f).lower())
            and "corrected" not in os.path.basename(f).lower()
        )
    ]

    moved = 0
    for f in rep_files:
        dst = os.path.join(old_reps_dir, os.path.basename(f))
        shutil.move(f, dst)
        moved += 1

    return moved


def delete_old_files(base_path, pattern):
    files = glob.glob(os.path.join(base_path, pattern))
    for f in files:
        os.remove(f)
    return len(files)


def get_csv_files(tech, base_path):
    return [f for f in glob.glob(os.path.join(base_path, "*.csv"))
            if not f.endswith("_corrected.csv")]


# ============================================================
# CORRECTION + LOGGING
# ============================================================

def correct_dataframe(df, log_rows, fname, tech):

    df =df.dropna()
    df = df.copy()

        # ----- Round integer-valued variables (before any validation) -----
    INTEGER_COLS = [
        "ST268Q01JA", "ST268Q04JA", "ST268Q07JA",  # Likert
        "ST004D01T", "IMMIG", "MISCED",            # categorical
        "ST001D01T",                               # grade
        "MCLSIZE", "SCHSIZE"                       # counts
    ]

    for c in INTEGER_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round()


    def log(var, action, pct):
        log_rows.append({
            "technique": tech,
            "file": fname,
            "variable": var,
            "action": action,
            "percent_rows_affected": round(100 * pct, 3)
        })

    n = len(df)

    # Helper: drop rows and log
    def drop_and_log(mask, var, reason):
        nonlocal df
        if mask.any():
            pct = mask.mean()
            log(var, f"drop_rows_{reason}", pct)
            df = df.loc[~mask].copy()

    # ----- Likert (DROP invalid rows) -----
    for c in LIKERT_4_COLS:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            invalid = vals.isna() | (round(vals,0) < 1) | (round(vals,0) > 4)
            drop_and_log(invalid, c, "invalid_likert_[1,4]")


    # ----- Proportions (DROP invalid rows) -----
    for c in PROP_0_100_COLS:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            invalid = vals.isna() | (round(vals,0) < 0) | (round(vals,0) > 100)
            drop_and_log(invalid, c, "invalid_prop_[0,100]")


    # ----- Categorical (DROP invalid rows) -----
    for c, valid in CATEGORICAL_RULES.items():
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            invalid = vals.isna() | (~round(vals,0).isin(valid))
            drop_and_log(invalid, c, "invalid_categorical_code")



    # ----- Continuous bounds (DROP invalid rows) -----
    for c, (lo, hi) in CONTINUOUS_CLIP_RULES.items():
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            invalid = vals.isna() | (round(vals,0) < lo) | (round(vals,0) > hi)
            drop_and_log(invalid, c, f"invalid_continuous_[{lo},{hi}]")

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("===== CLEAN + CORRECT + LOG PIPELINE =====")

    for tech, base_path in TECHNIQUE_PATHS.items():

        print(f"\n--- {tech} ---")

        removed = delete_old_files(base_path, "*_corrected.csv")
        if removed:
            print(f"Deleted {removed} old corrected files.")

        removed = delete_old_files(base_path, "correction_log_*.csv")
        if removed:
            print(f"Deleted {removed} old log files.")

        csv_files = get_csv_files(tech, base_path)

        if not csv_files:
            print("No CSV files found.")
            continue

        log_rows = []

        for csv_path in csv_files:

            fname = os.path.basename(csv_path)
            print(f"Processing: {fname}")

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print("  ERROR loading:", e)
                continue

            df_corr = correct_dataframe(df, log_rows, fname, tech)

            out_path = csv_path.replace(".csv", "_corrected.csv")
            df_corr.to_csv(out_path, index=False)

        moved = move_rep_files_to_old_reps(base_path)
        if moved:
            print(f"Moved {moved} rep CSV files to 'old reps' folder.")


        if log_rows:
            log_df = pd.DataFrame(log_rows)
            log_path = os.path.join(base_path, f"correction_log_{tech}.csv")
            log_df.to_csv(log_path, index=False)
            print(f"Log written: {log_path}")
        else:
            print("No corrections needed; no log written.")

    print("\n===== DONE =====")


if __name__ == "__main__":
    main()
