import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from numpy.linalg import norm

import logging

from datetime import datetime



# =====================================================
# CONFIG: TECHNIQUES & PATHS
# =====================================================

CNT = 'SGP'
TECHNIQUE_PATHS = {
    # -------------------------------------------------
    # REAL (unchanged)
    # -------------------------------------------------
    "REAL": r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Datasets\PISA 2022",

    # -------------------------------------------------
    # GReaT (Baseline / DistilGPT2, GPT2)
    # -------------------------------------------------
    "GReaT_DistilGPT2": rf"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\{CNT}\GReaT\DistilGPT2",
    "GReaT_GPT2":       rf"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\{CNT}\GReaT\GPT2",

    # -------------------------------------------------
    # Tabula
    # -------------------------------------------------
    "Tabula_DistilGPT2": rf"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\{CNT}\Tabula\DistilGPT2",
    "Tabula_GPT2":       rf"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\{CNT}\Tabula\GPT2",

    # -------------------------------------------------
    # TapTap
    # -------------------------------------------------
    "TapTap_DistilGPT2": rf"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\{CNT}\TapTap\DistilGPT2",
    "TapTap_GPT2":       rf"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\{CNT}\TapTap\GPT2",

    # -------------------------------------------------
    # PredLLM
    # -------------------------------------------------
    "PredLLM_DistilGPT2": rf"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\{CNT}\PredLLM\DistilGPT2",
    "PredLLM_GPT2":       rf"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\{CNT}\PredLLM\GPT2",

    # -------------------------------------------------
    # TabDiff (single-level)
    # -------------------------------------------------
    "TabDiff": rf"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\{CNT}\TabDiff",

    # -------------------------------------------------
    # REaLTabFormer (single-level)
    # -------------------------------------------------
    "REaLTabFormer": rf"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments\PISA-SEM\{CNT}\REaLTabFormer",
}


OUTPUT_FILE = r"C:/Users/mithu/Documents/MEGA/VUW/Summer Research Project/SEM_TECHNIQUE_COMPARISON.xlsx"

# =====================================================
# LOGGING CONFIGURATION
# =====================================================

LOG_LEVEL = logging.INFO   # change to DEBUG for deep inspection

LOG_DIR = os.path.dirname(OUTPUT_FILE)
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(
    LOG_DIR,
    f"sem_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# =====================================================
# HELPERS
# =====================================================
def numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def id_cols(df):
    return [c for c in df.columns if c not in numeric_cols(df)]

def normalize_sheet_name(name):
    if name.endswith("_mea"):
        return name[:-4] + "_mean"
    if name.endswith("_ran"):
        return name[:-4] + "_range"
    if name.endswith("_std"):
        return name
    if name.endswith("_mean") or name.endswith("_range"):
        return name
    return name


# =====================================================
# LOAD RESULTS FOR ONE TECHNIQUE
# =====================================================
def load_technique_results(technique, base_path):

    if technique == "REAL":
        file_path = os.path.join(base_path, f"sem_results_df_core{CNT}.xlsx")
        logger.debug(f"{technique}: Checking file path: {file_path}")
    else:
        file_path = os.path.join(base_path, "SEM_AGGREGATED_RESULTS.xlsx")
        logger.debug(f"{technique}: Checking file path: {file_path}")
    

    if not os.path.exists(file_path):
        logger.warning(f"[WARN] Missing file for {technique}: {file_path}")
        return {}

    xl = pd.ExcelFile(file_path, engine="openpyxl")
    logger.info(f"[INFO] {technique}: Loaded Excel with {len(xl.sheet_names)} sheets")

    sheets = {}
    for raw_name in xl.sheet_names:
        norm_name = normalize_sheet_name(raw_name)
        sheets[norm_name] = xl.parse(raw_name)
        logger.debug(f"[DEBUG] {technique}: Registered sheet '{raw_name}' → '{norm_name}'")

    logger.info(f"[INFO] {technique}: {len(sheets)} normalized sheets loaded")
    return sheets



# =====================================================
# CORE COMPARISON LOGIC
# =====================================================
def get_real_equivalent_sheet(sheet_name):
    # REAL has only point estimates
    if sheet_name.endswith("_range"):
        return sheet_name.replace("_range", "")
    if sheet_name.endswith("_std"):
        return None   # REAL has no std
    if sheet_name.endswith("_mean"):
        return sheet_name.replace("_mean", "")
    return sheet_name



def compare_sheet(sheet_name, all_results):
    logger.info(f"[INFO] Comparing logical sheet: {sheet_name}")

    dfs = {}

    for tech, sheets in all_results.items():
        if tech == "REAL":
            real_sheet = get_real_equivalent_sheet(sheet_name)
            if real_sheet and real_sheet in sheets:
                dfs[tech] = sheets[real_sheet]
        else:
            if sheet_name in sheets:
                dfs[tech] = sheets[sheet_name]

    logger.debug(f"[DEBUG] {sheet_name}: Techniques available = {list(dfs.keys())}")

    if len(dfs) < 2:
        return None

    # Choose identifier structure
    base_df = dfs["REAL"] if "REAL" in dfs else list(dfs.values())[0]
    keys = id_cols(base_df)

    # Preserve original row & column order from base (REAL preferred)
    base_id_order = id_cols(base_df)
    # Base numeric order comes from REAL if present
    if sheet_name.endswith("_range"):
        base_metric_order = [c for c in base_df.columns if c not in base_id_order]
    else:
        base_metric_order = numeric_cols(base_df)


    logger.debug(f"[DEBUG] {sheet_name}: ID columns = {keys}")
    logger.debug(f"[DEBUG] {sheet_name}: Base metric columns = {base_metric_order}")



    merged = None

    for tech, df in dfs.items():
        df = df.copy()
        # For _range sheets, treat ALL non-ID columns as metrics
        if sheet_name.endswith("_range"):
            metric_cols = [c for c in df.columns if c not in keys]
        else:
            metric_cols = numeric_cols(df)

        rename_map = {c: f"{tech}__{c}" for c in metric_cols}
        df = df.rename(columns=rename_map)

        keep = keys + list(rename_map.values())


        if merged is None:
            merged = df[keep]
        else:
            merged = merged.merge(
                df[keep],
                on=keys,
                how="left",        # anchor on base
                sort=False         # DO NOT reorder
            )

    


    # Enforce deterministic column order
    ordered_cols = base_id_order.copy()

    for metric in base_metric_order:
        for tech in dfs.keys():
            col = f"{tech}__{metric}"
            if col in merged.columns:
                ordered_cols.append(col)

    merged = merged[ordered_cols]

    logger.info(
    f"[INFO] {sheet_name}: Merged shape = {merged.shape} "
    f"(rows={merged.shape[0]}, cols={merged.shape[1]})"
    )
    return merged

def mae_rmse(x, y):
    """
    Compute MAE and RMSE with NA-safe alignment
    """
    valid = ~(x.isna() | y.isna())
    if valid.sum() == 0:
        logger.warning("[WARN] MAE/RMSE skipped: no overlapping non-NA values")
        return np.nan, np.nan

    x, y = x[valid].astype(float), y[valid].astype(float)

    mae = np.mean(np.abs(x - y))
    rmse = np.sqrt(np.mean((x - y) ** 2))

    return mae, rmse

def grid_error(real_df, synth_df, construct_col="Construct", drop_diagonal=False):
    """
    Returns:
      - cellwise absolute error grid (AE)
      - overall RMSE scalar across comparable cells

    If drop_diagonal=True, diagonal is excluded from overall RMSE (often desirable).
    """

    real_df = real_df.set_index(construct_col).copy()
    synth_df = synth_df.set_index(construct_col).copy()

    # Align to common constructs (rows & cols)
    common = real_df.index.intersection(synth_df.index)
    real_df = real_df.loc[common, common]
    synth_df = synth_df.loc[common, common]

    # Force numeric (robust against accidental object dtype)
    real_df = real_df.apply(pd.to_numeric, errors="coerce")
    synth_df = synth_df.apply(pd.to_numeric, errors="coerce")

    diff = real_df - synth_df

    # Cellwise Absolute Error grid (what your old "RMSE grid" effectively was)
    ae = diff.abs()

    # Overall RMSE across all comparable cells
    sq = diff.pow(2)

    if drop_diagonal:
        np.fill_diagonal(sq.values, np.nan)

    rmse_scalar = np.sqrt(np.nanmean(sq.values))

    # Format AE grid back to your "Construct + columns" layout
    ae_out = ae.copy()
    ae_out.insert(0, construct_col, ae_out.index)
    ae_out.reset_index(drop=True, inplace=True)

    return ae_out, rmse_scalar



def structural_coherence_metrics(df, tech, real_col, tech_col):
    logger.info(f"[INFO] Structural paths: {tech}")

    if tech_col not in df.columns:
        logger.warning(f"[WARN] {tech}: Missing column '{tech_col}' in structural paths")
        return {
            "MAE_Std_Path": np.nan,
            "Directional_Consistency": np.nan,
            "Rank_Preservation": np.nan
        }

    x = df[real_col].astype(float)
    y = df[tech_col].astype(float)

    valid = ~(x.isna() | y.isna())
    if valid.sum() == 0:
        logger.debug(f"[DEBUG] {tech}: Structural paths valid count = {valid.sum()}")

        return {
            "MAE_Std_Path": np.nan,
            "Directional_Consistency": np.nan,
            "Rank_Preservation": np.nan
        }

    x, y = x[valid], y[valid]

    mae, rmse = mae_rmse(x, y)
    direction_consistency = np.mean(np.sign(x) == np.sign(y))
    rank_corr, _ = spearmanr(np.abs(x), np.abs(y))

    return {
        "MAE_Std_Path": mae,
        "RMSE_Std_Path": rmse,
        "Directional_Consistency": direction_consistency,
        "Rank_Preservation": rank_corr
    }



def loading_similarity_metrics(df, tech, real_col, tech_col):
    logger.info(f"[INFO] Loadings comparison: {tech}")

    if tech_col not in df.columns:
        return np.nan, np.nan

    x = df[real_col].astype(float)
    y = df[tech_col].astype(float)

    valid = ~(x.isna() | y.isna())
    if valid.sum() == 0:
        logger.warning(f"[WARN] {tech}: No valid loading pairs")
        return np.nan, np.nan

    x, y = x[valid], y[valid]

    mae, rmse = mae_rmse(x, y)
    
    return mae, rmse



def reliability_metrics(df, tech, metric_col):
    logger.info(f"[INFO] Reliability: {tech} | Metric={metric_col}")

    real_col = f"REAL__{metric_col}"
    tech_col = f"{tech}__{metric_col}"

    if real_col not in df.columns or tech_col not in df.columns:
        logger.warning(
        f"[WARN] Reliability metric missing for {tech}: "
        f"{real_col} or {tech_col}"
          )
        return np.nan, np.nan, np.nan

    mae, rmse = mae_rmse(df[real_col], df[tech_col])
    rank_corr, _ = spearmanr(df[real_col], df[tech_col], nan_policy="omit")

    return mae, rmse, rank_corr


def cb_fit_error_long(df, tech, metric_name):
    """
    df is sem_cb_fit_measures_{CNT}_mean comparison sheet (long format):
      columns: metric, REAL__value, <tech>__value, ...
    metric_name examples: 'cfi', 'tli', 'rmsea', 'srmr'
    """
    id_col = "metric"
    real_col = "REAL__value"
    tech_col = f"{tech}__value"

    if id_col not in df.columns or real_col not in df.columns or tech_col not in df.columns:
        return np.nan, np.nan

    sub = df[df[id_col].astype(str).str.lower() == metric_name.lower()]
    if sub.empty:
        return np.nan, np.nan

    # single value expected, but NA-safe anyway
    x = sub[real_col]
    y = sub[tech_col]
    return mae_rmse(x, y)


def r2_metrics(df, tech):
    logger.info(f"[INFO] R² comparison: {tech}")

    real_col = "REAL__R2"
    tech_col = f"{tech}__R2"

    if real_col not in df.columns or tech_col not in df.columns:
        logger.warning(f"[WARN] R² missing for {tech}")
        return np.nan, np.nan, np.nan

    mae, rmse = mae_rmse(df[real_col], df[tech_col])
    rho, _ = spearmanr(df[real_col], df[tech_col], nan_policy="omit")

    return mae, rmse, rho

def total_effect_metrics(df, tech):
    real_col = "REAL__Original Est."
    tech_col = f"{tech}__Original Est."

    if real_col not in df.columns or tech_col not in df.columns:
        logger.warning(f"[WARN] Total effects missing for {tech}")
        return np.nan, np.nan, np.nan

    x = df[real_col].astype(float)
    y = df[tech_col].astype(float)

    valid = ~(x.isna() | y.isna())
    if valid.sum() == 0:
        return np.nan, np.nan, np.nan

    mae = np.mean(np.abs(x[valid] - y[valid]))
    rmse = np.sqrt(np.mean((x[valid] - y[valid]) ** 2))
    rho, _ = spearmanr(x[valid], y[valid])

    return mae, rmse, rho



def run_sem_evaluation_from_comparison(xl_path):
    
    xl = pd.ExcelFile(xl_path, engine="openpyxl")

    path_df = xl.parse(find_mean_sheet(xl, f"sem_full_std_paths_{CNT}"))
    load_df = xl.parse(find_mean_sheet(xl, f"pls_sem_loadings_R2_{CNT}"))
    htmt_df = xl.parse(find_mean_sheet(xl, f"pls_sem_htmt_{CNT}"))
    fl_df   = xl.parse(find_mean_sheet(xl, f"pls_sem_fornell_larcker_{CNT}"))


    rows = []

    for tech in TECHNIQUE_PATHS:
        logger.info("\n-----------------------------------------------")
        logger.info(f"[INFO] Evaluating technique: {tech}")
        logger.info("-----------------------------------------------")

        if tech == "REAL":
            continue

        row = {"Technique": tech}

        row.update(
            structural_coherence_metrics(
                path_df,
                tech,
                "REAL__Std_B",
                f"{tech}__Std_B"
            )
        )

        mae, rmse = loading_similarity_metrics(
            load_df,
            tech,
            "REAL__Loading",
            f"{tech}__Loading"
        )
        row["Loading_MAE"] = mae
        row["Loading_RMSE"] = rmse

        cb_fit_df = xl.parse(find_mean_sheet(xl, f"sem_cb_fit_measures_{CNT}"))

        for metric in ["cfi", "tli", "rmsea", "srmr"]:
            mae, rmse = cb_fit_error_long(cb_fit_df, tech, metric)
            row[f"{metric.upper()}_MAE"] = mae
            row[f"{metric.upper()}_RMSE"] = rmse

        RELIAB_MAP = {
            "Cronbach_Alpha": "alpha",
            "Composite_Reliability": "rhoC",
            "AVE": "AVE",
            "rhoA": "rhoA",  # optional if you want it in the output
        }


        reliab_df = xl.parse(find_mean_sheet(xl, f"pls_sem_reliability_{CNT}"))

        for nice_name, col in RELIAB_MAP.items():
            mae, rmse, rho = reliability_metrics(reliab_df, tech, col)
            row[f"{nice_name}_MAE"] = mae
            row[f"{nice_name}_RMSE"] = rmse
            row[f"{nice_name}_Spearman"] = rho


        r2_df = xl.parse(find_mean_sheet(xl, f"sem_cb_rsquare_{CNT}"))

        mae, rmse, rho = r2_metrics(r2_df, tech)
        row["R2_MAE"] = mae
        row["R2_RMSE"] = rmse
        row["R2_Spearman"] = rho

        te_df = xl.parse(find_mean_sheet(xl, f"sem_full_total_effects_{CNT}"))

        mae, rmse, rho = total_effect_metrics(te_df, tech)
        row["Total_Effect_MAE"] = mae
        row["Total_Effect_RMSE"] = rmse
        row["Total_Effect_Spearman"] = rho

        logger.info(
            f"[INFO] Completed {tech}: "
            f"Non-NA metrics = {sum(pd.notna(v) for v in row.values())}"
        )


        rows.append(row)


    return pd.DataFrame(rows)

def find_mean_sheet(xl, prefix):
    """
    Find the MEAN sheet for a given logical prefix,
    robust to Excel's 31-char truncation.
    """
    candidates = [
        s for s in xl.sheet_names
        if s.startswith(prefix) and (s.endswith("_mean") or s.endswith("_mea"))
    ]

    if len(candidates) == 0:
        raise ValueError(
            f"No MEAN sheet found for prefix '{prefix}'. "
            f"Available: {xl.sheet_names}"
        )

    if len(candidates) > 1:
        raise ValueError(
            f"Multiple MEAN sheets found for prefix '{prefix}': {candidates}"
        )

    return candidates[0]

# =====================================================
# LOAD ALL TECHNIQUES
# =====================================================
all_results = {}
for tech, path in TECHNIQUE_PATHS.items():
    logger.info(f"Loading {tech} at path {path}")
    all_results[tech] = load_technique_results(tech, path)

# =====================================================
# RUN COMPARISON FOR ALL SHEETS (PHASE A)
# =====================================================
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    VALID_SUFFIXES = ("_mean", "_std", "_range", "_mea")

    all_sheet_names = set()
    for sheets in all_results.values():
        for name in sheets.keys():
            if name.endswith(VALID_SUFFIXES):
                all_sheet_names.add(name)

    for sheet in sorted(all_sheet_names):
        logger.info(f"Comparing sheet: {sheet}")

        comp = compare_sheet(sheet, all_results)

        if comp is None:
            logger.warning(f"[SKIP] Not enough data for {sheet}")
            continue

        comp.to_excel(writer, sheet_name=sheet[:31], index=False)

# =====================================================
# RMSE/AE GRIDS FOR FORNELL–LARCKER & HTMT (PHASE B)
# =====================================================
logger.info("Generating error grids for Fornell-Larcker and HTMT")

xl_cmp = pd.ExcelFile(OUTPUT_FILE, engine="openpyxl")

fl_sheet = find_mean_sheet(xl_cmp, f"pls_sem_fornell_larcker_{CNT}")
htmt_sheet = find_mean_sheet(xl_cmp, f"pls_sem_htmt_{CNT}")

fl_df = xl_cmp.parse(fl_sheet)
htmt_df = xl_cmp.parse(htmt_sheet)

summary_rows = []

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:

    for tech in TECHNIQUE_PATHS:
        if tech == "REAL":
            continue

        logger.info(f"Error grids for {tech}")

        # -----------------------
        # Fornell–Larcker
        # -----------------------
        real_cols = ["Construct"] + [c.replace("REAL__", "") for c in fl_df.columns if c.startswith("REAL__")]
        tech_cols = ["Construct"] + [c.replace(f"{tech}__", "") for c in fl_df.columns if c.startswith(f"{tech}__")]

        fl_rmse_scalar = np.nan

        if len(real_cols) > 1 and len(tech_cols) > 1:
            real_mat = fl_df[["Construct"] + [f"REAL__{c}" for c in real_cols[1:]]].copy()
            tech_mat = fl_df[["Construct"] + [f"{tech}__{c}" for c in tech_cols[1:]]].copy()

            real_mat.columns = real_cols
            tech_mat.columns = tech_cols

            ae_fl, fl_rmse_scalar = grid_error(real_mat, tech_mat, construct_col="Construct", drop_diagonal=False)

            ae_fl.to_excel(
                writer,
                sheet_name=f"AE_FL_{tech}"[:31],
                index=False
            )

        # -----------------------
        # HTMT
        # -----------------------
        real_cols = ["Construct"] + [c.replace("REAL__", "") for c in htmt_df.columns if c.startswith("REAL__")]
        tech_cols = ["Construct"] + [c.replace(f"{tech}__", "") for c in htmt_df.columns if c.startswith(f"{tech}__")]

        htmt_rmse_scalar = np.nan

        if len(real_cols) > 1 and len(tech_cols) > 1:
            real_mat = htmt_df[["Construct"] + [f"REAL__{c}" for c in real_cols[1:]]].copy()
            tech_mat = htmt_df[["Construct"] + [f"{tech}__{c}" for c in tech_cols[1:]]].copy()

            real_mat.columns = real_cols
            tech_mat.columns = tech_cols

            # Often HTMT diagonal is not meaningful; if yours is blank/NA anyway,
            # drop_diagonal=True makes the scalar RMSE more stable.
            ae_htmt, htmt_rmse_scalar = grid_error(real_mat, tech_mat, construct_col="Construct", drop_diagonal=True)

            ae_htmt.to_excel(
                writer,
                sheet_name=f"AE_HTMT_{tech}"[:31],
                index=False
            )

        summary_rows.append({
            "Technique": tech,
            "FL_RMSE_scalar": fl_rmse_scalar,
            "HTMT_RMSE_scalar": htmt_rmse_scalar
        })

    # Optional: write summary scalars into the workbook
    pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Grid_RMSE_Summary"[:31], index=False)

eval_df = run_sem_evaluation_from_comparison(OUTPUT_FILE)

eval_df.to_excel(
    OUTPUT_FILE.replace(".xlsx", "_SEM_EVALUATION.xlsx"),
    index=False
)



logger.info("\n===================================================")
logger.info("SEM technique comparison COMPLETE")
logger.info(f"Comparison workbook : {OUTPUT_FILE}")
logger.info(f"Evaluation workbook : {OUTPUT_FILE.replace('.xlsx', '_SEM_EVALUATION.xlsx')}")
logger.info(f"Techniques evaluated: {len(TECHNIQUE_PATHS) - 1}")
logger.info("===================================================")








