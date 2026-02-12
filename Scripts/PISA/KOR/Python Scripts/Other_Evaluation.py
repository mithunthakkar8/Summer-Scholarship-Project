import os
import glob
import math
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ============================
# CONFIG
# ============================

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

# REAL file rule (your requirement)
REAL_FILENAME = "df_coreSGP.csv"

# Columns you typically care about (script will auto-detect numeric/categorical anyway)
# You may add/remove as needed.
PREFERRED_NUMERIC_PREFIXES = ("PV",)
PREFERRED_NUMERIC_EXACT = {"AGE", "ESCS", "MCLSIZE", "SCHSIZE", "IMMIG", "MISCED", "ST001D01T", "ST004D01T"}

# Speed controls
MAX_ROWS_FOR_DCR = 5000        # subsample for DCR to keep runtime reasonable
DCR_RANDOM_SEED = 42
TOPK_INTERACTIONS = 30         # for interaction consistency

# Marginal metrics controls
HIST_BINS = 20
KS_ALPHA = 0.05

# Optional LLM perplexity controls
ENABLE_LLM_PERPLEXITY = False
PERPLEXITY_MODEL_NAME = "gpt2"   # local HF model name
PPL_MAX_ROWS = 1000              # subsample rows for perplexity (per file)
PPL_MAX_TOKENS = 256             # truncate serialized row text tokens

# ============================
# QUASI-IDENTIFIER COLUMNS (Categorical)
# ============================

QI_COLUMNS = [
    "ST001D01T", 
    "ST004D01T",
    "ST268Q01JA",
    "ST268Q04JA",
    "ST268Q07JA",
    "IMMIG",
    "MISCED"  
]

# ============================
# 5b) Categorical QI Linkability Risk
# ============================

def qi_linkability_metrics(real_df: pd.DataFrame, synth_df: pd.DataFrame, qi_cols: List[str]) -> Dict[str, float]:

    # keep only columns that exist in both
    qi = [c for c in qi_cols if c in real_df.columns and c in synth_df.columns]
    if len(qi) == 0:
        return {
            "qi_cols_used": 0,
            "qi_match_rate": np.nan,
            "qi_unique_match_rate": np.nan,
            "qi_avg_equiv_class_size": np.nan,
            "qi_high_risk_rate": np.nan,
        }

    # build QI patterns as tuples
    real_qi = real_df[qi].fillna("__NA__").astype(str)
    synth_qi = synth_df[qi].fillna("__NA__").astype(str)

    real_tuples = list(map(tuple, real_qi.values))
    synth_tuples = list(map(tuple, synth_qi.values))

    # equivalence classes in real data
    from collections import Counter
    real_counts = Counter(real_tuples)

    matched = 0
    unique_matched = 0
    equiv_sizes = []
    high_risk = 0     # k <= 5 (common disclosure threshold)

    K_RISK = 5

    for t in synth_tuples:
        if t in real_counts:
            matched += 1
            k = real_counts[t]
            equiv_sizes.append(k)
            if k == 1:
                unique_matched += 1
            if k <= K_RISK:
                high_risk += 1

    n = len(synth_tuples)

    return {
        "qi_cols_used": len(qi),
        "qi_match_rate": matched / n if n else np.nan,
        "qi_unique_match_rate": unique_matched / n if n else np.nan,
        "qi_avg_equiv_class_size": float(np.mean(equiv_sizes)) if equiv_sizes else np.nan,
        "qi_high_risk_rate": high_risk / n if n else np.nan,
    }


# ============================
# IO helpers
# ============================

def list_csvs(directory: str) -> List[str]:
    return sorted(
        f for f in glob.glob(os.path.join(directory, "*.csv"))
        if "correction" not in os.path.basename(f).lower()
    )

def load_real_csv(real_dir: str) -> pd.DataFrame:
    p = os.path.join(real_dir, REAL_FILENAME)
    if not os.path.exists(p):
        raise FileNotFoundError(f"REAL file not found: {p}")
    return pd.read_csv(p)

def load_synth_csv_paths(synth_dir: str) -> List[str]:
    csvs = list_csvs(synth_dir)
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in: {synth_dir}")
    return csvs

def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None


# ============================
# Column selection helpers
# ============================

def split_numeric_categorical(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # Numeric columns: pandas numeric dtypes
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Prefer common PISA numeric columns even if parsed as object (rare)
    for c in df.columns:
        if c in PREFERRED_NUMERIC_EXACT and c not in numeric_cols:
            # try coercion
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().mean() > 0.95:
                df[c] = coerced
                numeric_cols.append(c)

    # PV*MATH columns
    for c in df.columns:
        if any(c.startswith(pfx) for pfx in PREFERRED_NUMERIC_PREFIXES) and "MATH" in c and c not in numeric_cols:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().mean() > 0.95:
                df[c] = coerced
                numeric_cols.append(c)

    numeric_cols = sorted(set(numeric_cols))

    # Categorical: non-numeric, but exclude obvious IDs if any
    FORCE_CATEGORICAL = {"ST001D01T", "IMMIG", "MISCED", "ST004D01T"}

    cat_cols = [c for c in df.columns if c not in numeric_cols or c in FORCE_CATEGORICAL]

    return numeric_cols, cat_cols

def compute_math_score(df: pd.DataFrame) -> pd.Series:
    pv_cols = [c for c in df.columns if c.startswith("PV") and "MATH" in c]
    if pv_cols:
        return df[pv_cols].mean(axis=1)
    # fallback: maybe a single math column
    for alt in ["MATH", "PV1MATH"]:
        if alt in df.columns:
            return pd.to_numeric(df[alt], errors="coerce")
    raise ValueError("No PV*MATH (or fallback math) columns found")

def sanitize_df_for_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # drop columns with too many missing
    keep = [c for c in out.columns if out[c].notna().mean() >= 0.9]
    out = out[keep]
    return out


# ============================
# 2) Dependency / correlation metrics
# ============================

def corr_matrix(df_num: pd.DataFrame, method: str) -> pd.DataFrame:
    # pairwise complete correlation
    return df_num.corr(method=method, min_periods=max(10, int(0.2 * len(df_num))))

def upper_triangle_values(mat: pd.DataFrame) -> np.ndarray:
    arr = mat.to_numpy()
    iu = np.triu_indices_from(arr, k=1)
    vals = arr[iu]
    vals = vals[~np.isnan(vals)]
    return vals

def correlation_error_metrics(real_num: pd.DataFrame, synth_num: pd.DataFrame) -> Dict[str, float]:
    # align columns
    common = sorted(set(real_num.columns).intersection(set(synth_num.columns)))
    if len(common) < 3:
        return {
            "corr_common_features": len(common),
            "pearson_mae": np.nan, "pearson_fro": np.nan,
            "spearman_mae": np.nan, "spearman_fro": np.nan,
            "mantel_flat_corr_pearson": np.nan,
        }

    rR_p = corr_matrix(real_num[common], "pearson")
    rS_p = corr_matrix(synth_num[common], "pearson")

    rR_s = corr_matrix(real_num[common], "spearman")
    rS_s = corr_matrix(synth_num[common], "spearman")

    # errors on upper triangle
    d_p = upper_triangle_values(rS_p - rR_p)
    d_s = upper_triangle_values(rS_s - rR_s)

    pearson_mae = float(np.mean(np.abs(d_p)))
    spearman_mae = float(np.mean(np.abs(d_s)))

    pearson_fro = float(np.linalg.norm((rS_p - rR_p).to_numpy(), ord="fro"))
    spearman_fro = float(np.linalg.norm((rS_s - rR_s).to_numpy(), ord="fro"))

    # structure-aware: correlation between flattened correlation matrices (Mantel-style, without permutation test)
    flat_r = upper_triangle_values(rR_p)
    flat_s = upper_triangle_values(rS_p)
    min_len = min(len(flat_r), len(flat_s))
    if min_len > 10:
        mantel = float(np.corrcoef(flat_r[:min_len], flat_s[:min_len])[0, 1])
    else:
        mantel = np.nan

    return {
        "corr_common_features": len(common),
        "pearson_mae": pearson_mae,
        "pearson_fro": pearson_fro,
        "spearman_mae": spearman_mae,
        "spearman_fro": spearman_fro,
        "mantel_flat_corr_pearson": mantel,
    }

def eigen_spectrum_distance(real_num: pd.DataFrame, synth_num: pd.DataFrame) -> Dict[str, float]:
    common = sorted(set(real_num.columns).intersection(set(synth_num.columns)))
    if len(common) < 3:
        return {"eig_common_features": len(common), "eig_l2": np.nan, "eig_cosine": np.nan}

    # correlation matrices (pearson), fill NaNs with 0 for eig stability
    CR = corr_matrix(real_num[common], "pearson").fillna(0.0).to_numpy()
    CS = corr_matrix(synth_num[common], "pearson").fillna(0.0).to_numpy()

    # eigenvalues sorted descending
    er = np.sort(np.real(np.linalg.eigvals(CR)))[::-1]
    es = np.sort(np.real(np.linalg.eigvals(CS)))[::-1]

    # normalize to compare shape
    er = er / (np.linalg.norm(er) + 1e-12)
    es = es / (np.linalg.norm(es) + 1e-12)

    l2 = float(np.linalg.norm(er - es))
    cos = float(np.dot(er, es) / ((np.linalg.norm(er) * np.linalg.norm(es)) + 1e-12))

    return {"eig_common_features": len(common), "eig_l2": l2, "eig_cosine": cos}

def topk_interaction_consistency(real_num: pd.DataFrame, synth_num: pd.DataFrame, k: int = 30) -> Dict[str, float]:
    common = sorted(set(real_num.columns).intersection(set(synth_num.columns)))
    if len(common) < 4:
        return {"interaction_common_features": len(common), "topk_k": k, "topk_overlap_rate": np.nan, "topk_sign_agreement": np.nan}

    R = corr_matrix(real_num[common], "pearson")
    S = corr_matrix(synth_num[common], "pearson")

    # score each pair by |corr_real|
    pairs = []
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a, b = common[i], common[j]
            val = R.loc[a, b]
            if not np.isnan(val):
                pairs.append((a, b, abs(val), np.sign(val)))

    pairs.sort(key=lambda x: x[2], reverse=True)
    top = pairs[: min(k, len(pairs))]

    # measure: do these pairs remain among top-k in synthetic?
    # rank synthetic by |corr_synth|
    spairs = []
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a, b = common[i], common[j]
            val = S.loc[a, b]
            if not np.isnan(val):
                spairs.append((a, b, abs(val), np.sign(val)))
    spairs.sort(key=lambda x: x[2], reverse=True)
    stop_set = set((a, b) for a, b, _, _ in spairs[: min(k, len(spairs))])

    real_top_set = set((a, b) for a, b, _, _ in top)
    overlap = len(real_top_set.intersection(stop_set)) / max(1, len(real_top_set))

    # sign agreement on real top-k
    sign_agree = []
    for a, b, _, sign_r in top:
        sign_s = np.sign(S.loc[a, b]) if not np.isnan(S.loc[a, b]) else 0
        sign_agree.append(1.0 if sign_s == sign_r else 0.0)
    sign_agree = float(np.mean(sign_agree)) if sign_agree else np.nan

    return {
        "interaction_common_features": len(common),
        "topk_k": min(k, len(pairs)),
        "topk_overlap_rate": float(overlap),
        "topk_sign_agreement": sign_agree,
    }


# ============================
# 4) Marginal distribution similarity
# ============================

def ks_numeric(real: pd.Series, synth: pd.Series) -> Tuple[float, float]:
    # returns (KS statistic, p-value) using scipy if available; otherwise approximate with empirical CDF max gap
    r = real.dropna().to_numpy()
    s = synth.dropna().to_numpy()
    if len(r) < 20 or len(s) < 20:
        return np.nan, np.nan

    try:
        from scipy.stats import ks_2samp
        res = ks_2samp(r, s, alternative="two-sided", mode="auto")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        # fallback approx KS (no p-value)
        r_sorted = np.sort(r)
        s_sorted = np.sort(s)
        all_vals = np.sort(np.unique(np.concatenate([r_sorted, s_sorted])))
        r_cdf = np.searchsorted(r_sorted, all_vals, side="right") / len(r_sorted)
        s_cdf = np.searchsorted(s_sorted, all_vals, side="right") / len(s_sorted)
        ks = float(np.max(np.abs(r_cdf - s_cdf)))
        return ks, np.nan

def tvd_categorical(real: pd.Series, synth: pd.Series) -> float:
    r = real.fillna("__NA__").astype(str)
    s = synth.fillna("__NA__").astype(str)

    pr = r.value_counts(normalize=True)
    ps = s.value_counts(normalize=True)

    idx = pr.index.union(ps.index)
    pr = pr.reindex(idx, fill_value=0.0)
    ps = ps.reindex(idx, fill_value=0.0)

    return float(0.5 * np.abs(pr - ps).sum())

def histogram_overlap(real: pd.Series, synth: pd.Series, bins: int = 20) -> float:
    r = real.dropna().to_numpy()
    s = synth.dropna().to_numpy()
    if len(r) < 20 or len(s) < 20:
        return np.nan

    lo = np.nanmin([r.min(), s.min()])
    hi = np.nanmax([r.max(), s.max()])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return np.nan

    hr, edges = np.histogram(r, bins=bins, range=(lo, hi), density=True)
    hs, _ = np.histogram(s, bins=bins, range=(lo, hi), density=True)

    # overlap area
    width = edges[1] - edges[0]
    return float(np.sum(np.minimum(hr, hs)) * width)

def marginal_metrics(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Dict[str, float]:
    real = real_df.copy()
    synth = synth_df.copy()

    num_r, cat_r = split_numeric_categorical(real)
    num_s, cat_s = split_numeric_categorical(synth)

    num_common = sorted(set(num_r).intersection(set(num_s)))
    cat_common = sorted(set(cat_r).intersection(set(cat_s)))

    # Numeric KS + histogram overlap
    ks_stats = []
    ks_sig = []
    overlaps = []
    for c in num_common:
        ks, p = ks_numeric(real[c], synth[c])
        if not np.isnan(ks):
            ks_stats.append(ks)
        if p is not None and not np.isnan(p):
            ks_sig.append(1.0 if p < KS_ALPHA else 0.0)
        ov = histogram_overlap(real[c], synth[c], bins=HIST_BINS)
        if not np.isnan(ov):
            overlaps.append(ov)

    # Categorical TVD
    tvds = []
    for c in cat_common:
        tvds.append(tvd_categorical(real[c], synth[c]))

    out = {
        "marg_num_common": len(num_common),
        "marg_cat_common": len(cat_common),
        "ks_mean": float(np.mean(ks_stats)) if ks_stats else np.nan,
        "ks_sig_frac": float(np.mean(ks_sig)) if ks_sig else np.nan,
        "hist_overlap_mean": float(np.mean(overlaps)) if overlaps else np.nan,
        "tvd_cat_mean": float(np.mean(tvds)) if tvds else np.nan,
    }
    return out


# ============================
# 5) Distance-based privacy metrics (DCR)
# ============================

def standardize_fit_transform(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - mu) / sd
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z, mu, sd

def standardize_transform(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    Z = (X - mu) / sd
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z

def dcr_metrics(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Dict[str, float]:
    r = real_df.copy()
    s = synth_df.copy()

    num_r, _ = split_numeric_categorical(r)
    num_s, _ = split_numeric_categorical(s)

    common = sorted(set(num_r).intersection(set(num_s)))
    if len(common) < 3:
        return {"dcr_common_features": len(common), "dcr_r2s_mean": np.nan, "dcr_s2r_mean": np.nan}

    rN = sanitize_df_for_numeric(r, common).dropna()
    sN = sanitize_df_for_numeric(s, common).dropna()

    if rN.empty or sN.empty:
        return {"dcr_common_features": len(common), "dcr_r2s_mean": np.nan, "dcr_s2r_mean": np.nan}

    # Subsample for speed
    rng = np.random.default_rng(DCR_RANDOM_SEED)
    if len(rN) > MAX_ROWS_FOR_DCR:
        rN = rN.sample(n=MAX_ROWS_FOR_DCR, random_state=DCR_RANDOM_SEED)
    if len(sN) > MAX_ROWS_FOR_DCR:
        sN = sN.sample(n=MAX_ROWS_FOR_DCR, random_state=DCR_RANDOM_SEED)

    XR = rN.to_numpy(dtype=float)
    XS = sN.to_numpy(dtype=float)

    ZR, mu, sd = standardize_fit_transform(XR)
    ZS = standardize_transform(XS, mu, sd)

    # nearest neighbor distance via sklearn if available; else brute force (may be slow)
    try:
        from sklearn.neighbors import NearestNeighbors
        nnS = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(ZS)
        d_r2s, _ = nnS.kneighbors(ZR, return_distance=True)
        d_r2s = d_r2s.ravel()

        nnR = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(ZR)
        d_s2r, _ = nnR.kneighbors(ZS, return_distance=True)
        d_s2r = d_s2r.ravel()
    except Exception:
        # brute force fallback
        def min_dist(A, B):
            out = []
            for i in range(A.shape[0]):
                d = np.sqrt(np.sum((B - A[i]) ** 2, axis=1))
                out.append(np.min(d))
            return np.array(out, dtype=float)
        d_r2s = min_dist(ZR, ZS)
        d_s2r = min_dist(ZS, ZR)

    def summarize(x: np.ndarray, prefix: str) -> Dict[str, float]:
        return {
            f"{prefix}_mean": float(np.mean(x)),
            f"{prefix}_min": float(np.min(x)),
            f"{prefix}_p01": float(np.quantile(x, 0.01)),
            f"{prefix}_p05": float(np.quantile(x, 0.05)),
            f"{prefix}_p50": float(np.quantile(x, 0.50)),
        }

    out = {"dcr_common_features": len(common)}
    out.update(summarize(d_r2s, "dcr_r2s"))
    out.update(summarize(d_s2r, "dcr_s2r"))
    return out


# ============================
# 3) LLM-specific privacy metrics (Perplexity-based)
# ============================

def try_load_lm():
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(PERPLEXITY_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(PERPLEXITY_MODEL_NAME)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return tok, model, device
    except Exception as e:
        print(f"[WARN] LLM perplexity disabled or unavailable: {e}")
        return None, None, None

def row_to_text(row: pd.Series, cols: List[str]) -> str:
    # stable, low-leak format: "col=value | col=value ..."
    parts = []
    for c in cols:
        v = row.get(c, "")
        if pd.isna(v):
            v = ""
        parts.append(f"{c}={v}")
    return " | ".join(parts)

def compute_perplexities(df: pd.DataFrame, cols: List[str], tok, model, device, max_rows: int) -> np.ndarray:
    import torch

    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=123)

    texts = [row_to_text(df.iloc[i], cols) for i in range(len(df))]

    ppl = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=PPL_MAX_TOKENS)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
            loss = out.loss.detach().cpu().item()
        ppl.append(float(math.exp(min(50.0, loss))))  # guard overflow
    return np.array(ppl, dtype=float)

def perplexity_privacy_metrics(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Dict[str, float]:
    tok, model, device = try_load_lm()
    if tok is None:
        return {
            "ppl_enabled": 0,
            "ppl_real_mean": np.nan,
            "ppl_synth_mean": np.nan,
            "ppl_gap_synth_minus_real": np.nan,
            "ppl_suspicious_low_rate": np.nan,
            "ppl_memorization_indicator": np.nan,
        }

    # choose a compact, comparable set of columns for serialization
    num_r, cat_r = split_numeric_categorical(real_df)
    num_s, cat_s = split_numeric_categorical(synth_df)
    common = sorted(set(num_r + cat_r).intersection(set(num_s + cat_s)))

    # avoid very high-dimensional serialization
    # keep common PISA columns preferentially if present, else first N columns
    preferred = [c for c in ["AGE", "ST004D01T", "ESCS", "MISCED", "ST001D01T", "MCLSIZE"] if c in common]
    cols = preferred + [c for c in common if c not in preferred]
    cols = cols[:25]

    ppl_real = compute_perplexities(real_df, cols, tok, model, device, max_rows=PPL_MAX_ROWS)
    ppl_syn = compute_perplexities(synth_df, cols, tok, model, device, max_rows=PPL_MAX_ROWS)

    # DLT-like proxy: perplexity gap
    gap = float(np.mean(ppl_syn) - np.mean(ppl_real))

    # suspiciously low perplexity fraction (synthetic below 5th percentile of real)
    thr5 = float(np.quantile(ppl_real, 0.05))
    suspicious = float(np.mean(ppl_syn <= thr5))

    # “memorization indicator” (synthetic below 1st percentile of real)
    thr1 = float(np.quantile(ppl_real, 0.01))
    memor = float(np.mean(ppl_syn <= thr1))

    return {
        "ppl_enabled": 1,
        "ppl_real_mean": float(np.mean(ppl_real)),
        "ppl_synth_mean": float(np.mean(ppl_syn)),
        "ppl_gap_synth_minus_real": gap,
        "ppl_suspicious_low_rate": suspicious,
        "ppl_memorization_indicator": memor,
    }


# ============================
# Orchestration: per-file -> per-technique averaging
# ============================

def mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    df = pd.DataFrame(dicts)
    return df.mean(numeric_only=True).to_dict()

def evaluate_one_synth_file(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    # add math score to both if needed
    # (not required for correlation unless PV columns exist; we keep as-is)
    # dependency
    real_num_cols, _ = split_numeric_categorical(real_df)
    synth_num_cols, _ = split_numeric_categorical(synth_df)
    real_num = sanitize_df_for_numeric(real_df, real_num_cols)
    synth_num = sanitize_df_for_numeric(synth_df, synth_num_cols)

    dep = {}
    dep.update(correlation_error_metrics(real_num, synth_num))
    dep.update(topk_interaction_consistency(real_num, synth_num, k=TOPK_INTERACTIONS))
    dep.update(eigen_spectrum_distance(real_num, synth_num))

    marg = marginal_metrics(real_df, synth_df)

    priv = dcr_metrics(real_df, synth_df)

    # categorical quasi-identifier linkage risk
    priv.update(qi_linkability_metrics(real_df, synth_df, QI_COLUMNS))

    if ENABLE_LLM_PERPLEXITY:
        priv.update(perplexity_privacy_metrics(real_df, synth_df))


    return dep, marg, priv

def main():
    out_root = os.path.join(
        r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project",
        "eval_metrics_outputs"
    )
    os.makedirs(out_root, exist_ok=True)

    # Load REAL once
    real_df = load_real_csv(TECHNIQUE_PATHS["REAL"])
    print(f"Loaded REAL: {REAL_FILENAME} shape={real_df.shape}")

    dep_rows = []
    marg_rows = []
    priv_rows = []

    for tech, path in TECHNIQUE_PATHS.items():
        if tech == "REAL":
            continue  # REAL is the reference

        print(f"\nProcessing technique: {tech}")
        csv_paths = load_synth_csv_paths(path)
        print(f"  Found {len(csv_paths)} CSVs")

        dep_list, marg_list, priv_list = [], [], []
        used = 0
        for p in csv_paths:
            df = safe_read_csv(p)
            if df is None:
                continue
            try:
                dep, marg, priv = evaluate_one_synth_file(real_df, df)
                dep_list.append(dep)
                marg_list.append(marg)
                priv_list.append(priv)
                used += 1
            except Exception as e:
                print(f"  [WARN] Skipping {os.path.basename(p)} due to error: {e}")

        if used == 0:
            print(f"  [WARN] No usable CSVs for {tech}, skipping technique.")
            continue

        dep_mean = mean_dict(dep_list)
        dep_mean["Technique"] = tech
        dep_mean["Runs_used"] = used
        dep_rows.append(dep_mean)

        marg_mean = mean_dict(marg_list)
        marg_mean["Technique"] = tech
        marg_mean["Runs_used"] = used
        marg_rows.append(marg_mean)

        priv_mean = mean_dict(priv_list)
        priv_mean["Technique"] = tech
        priv_mean["Runs_used"] = used
        priv_rows.append(priv_mean)

    dep_df = pd.DataFrame(dep_rows).set_index("Technique").sort_index()
    marg_df = pd.DataFrame(marg_rows).set_index("Technique").sort_index()
    priv_df = pd.DataFrame(priv_rows).set_index("Technique").sort_index()

    dep_path = os.path.join(out_root, "dependency_metrics_by_technique.csv")
    marg_path = os.path.join(out_root, "marginal_metrics_by_technique.csv")
    priv_path = os.path.join(out_root, "privacy_metrics_by_technique.csv")

    dep_df.round(6).to_csv(dep_path)
    marg_df.round(6).to_csv(marg_path)
    priv_df.round(6).to_csv(priv_path)

    print("\nSaved:")
    print(" ", dep_path)
    print(" ", marg_path)
    print(" ", priv_path)

    print("\nPreview (dependency):")
    print(dep_df.round(4).head(20))

    print("\nPreview (marginal):")
    print(marg_df.round(4).head(20))

    print("\nPreview (privacy):")
    print(priv_df.round(4).head(20))


if __name__ == "__main__":
    main()
