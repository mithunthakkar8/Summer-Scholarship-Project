import os
import glob
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import seaborn as sns
import numpy as np



# =====================================================
# PATHS (AS PROVIDED)
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

OUTPUT_DIR = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Plots\Distributions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD REAL DATA
# =====================================================
CNT = "SGP"   # change to "KOR" if needed

real_pattern = os.path.join(TECHNIQUE_PATHS["REAL"], f"df_core{CNT}.csv")
real_csvs = glob.glob(real_pattern)

if len(real_csvs) != 1:
    raise ValueError(f"Expected exactly 1 real CSV for {CNT}, found {len(real_csvs)}")

print("Loading REAL:", real_csvs[0])
df_real = pd.read_csv(real_csvs[0])



# =====================================================
# HELPERS
# =====================================================
def load_and_concat_csvs(folder):
    csvs = glob.glob(os.path.join(folder, "*.csv"))
    if len(csvs) == 0:
        raise ValueError(f"No CSVs found in {folder}")
    dfs = [pd.read_csv(f) for f in csvs]
    return pd.concat(dfs, ignore_index=True)

CATEGORICAL_COLS = [
    "IMMIG", "ST004D01T", "ST001D01T",  # gender, grade if included
]

ORDINAL_COLS = [
    "MISCED",
    "ST268Q01JA", "ST268Q04JA", "ST268Q07JA",
]

CONTINUOUS_COLS = [
    c for c in df_real.columns
    if c not in CATEGORICAL_COLS + ORDINAL_COLS
]




# =====================================================
# PLOTTING
# =====================================================
def plot_hist_grid(df_real, df_syn, technique_name, metrics_rows):
    cols = df_real.columns.tolist()
    n_cols = len(cols)

    grid_cols = 5
    grid_rows = math.ceil(n_cols / grid_cols)

    fig, axes = plt.subplots(
        grid_rows, grid_cols,
        figsize=(24, 4.2 * grid_rows),
        sharey=False
    )
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]

        real_vals = df_real[col].dropna()
        syn_vals = df_syn[col].dropna()

        # ---- shared bins for fair comparison ----
        combined = pd.concat([real_vals, syn_vals])
        bins = 30

        # ---- real: filled histogram ----
        ax.hist(
            real_vals,
            bins=bins,
            density=True,
            alpha=0.65,
            edgecolor="black",
            linewidth=0.4,
            label="Real"
        )

        # ---- synthetic: step outline ----
        ax.hist(
            syn_vals,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label="Synthetic"
        )

        # ---- metric ----
        if col in CONTINUOUS_COLS:
            dist = compute_wasserstein(real_vals, syn_vals)
            metric_txt = f"W={dist:.2f}"
            vtype = "continuous"
        else:
            dist = compute_js_divergence(real_vals, syn_vals)
            metric_txt = f"JS={dist:.2f}"
            vtype = "categorical"

        ax.set_title(f"{col}\n{metric_txt}", fontsize=10, pad=6)

        # ---- subtle grid for readability ----
        ax.grid(True, linestyle="--", alpha=0.3)

        # ---- reduce tick clutter ----
        ax.tick_params(axis="both", labelsize=8)

        metrics_rows.append({
            "technique": technique_name,
            "variable": col,
            "type": vtype,
            "distance": dist
        })

    # ---- remove empty subplots ----
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # ---- global layout polish ----
    fig.suptitle(
        f"Marginal Distribution Comparison: REAL vs {technique_name}",
        fontsize=16,
        y=0.995
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    # ---- single legend for whole figure ----
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    out_path = os.path.join(OUTPUT_DIR, f"hist_grid_{technique_name}.png")
    plt.savefig(out_path, dpi=220)
    plt.close()

    print("Saved:", out_path)





def compute_wasserstein(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return wasserstein_distance(a, b)


def compute_js_divergence(a, b):
    # frequency-based JS for categorical / ordinal
    all_vals = sorted(set(a) | set(b))

    pa = pd.Series(a).value_counts(normalize=True).reindex(all_vals, fill_value=0).values
    pb = pd.Series(b).value_counts(normalize=True).reindex(all_vals, fill_value=0).values

    return jensenshannon(pa, pb)

# =====================================================
# CORRELATION DIFFERENCE HEATMAP
# =====================================================
def plot_corr_diff_heatmap(df_real, df_syn, technique_name):
    # numeric columns only
    real_corr = df_real.corr(numeric_only=True)
    syn_corr  = df_syn.corr(numeric_only=True)

    diff = real_corr - syn_corr

    plt.figure(figsize=(14, 12))
    sns.heatmap(
    diff,
    center=0,
    cmap="coolwarm",
    square=True,
    annot=True,              # <<< SHOW NUMBERS
    fmt=".2f",               # <<< 2 decimal places
    annot_kws={"size": 7},   # <<< font size of numbers
    cbar_kws={"label": "Correlation Difference (Real − Synthetic)"}
    )

    plt.title(f"Correlation Difference Heatmap: {technique_name}", fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"corr_diff_{technique_name}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Saved correlation diff heatmap:", out_path)


# =====================================================
# MAIN LOOP
# =====================================================
metrics_rows = []

for tech, path in TECHNIQUE_PATHS.items():

    if tech == "REAL":
        continue

    print(f"\nProcessing {tech}")

    df_syn = load_and_concat_csvs(path)

    common_cols = sorted(set(df_real.columns) & set(df_syn.columns))
    df_real_use = df_real[common_cols]
    df_syn_use = df_syn[common_cols]

    plot_hist_grid(df_real_use, df_syn_use, tech, metrics_rows)

    plot_corr_diff_heatmap(df_real_use, df_syn_use, tech)

csv_path = os.path.join(OUTPUT_DIR, "marginal_distances.csv")
pd.DataFrame(metrics_rows).to_csv(csv_path, index=False)
print("Saved marginal metrics to:", csv_path)

