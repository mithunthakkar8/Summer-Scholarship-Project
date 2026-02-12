import os
import pandas as pd
import numpy as np
from semopy import Model, calc_stats, semplot

# =====================================================
# CONFIG
# =====================================================

DATA_DIR = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Datasets\PISA 2022"
INPUT_CSV = os.path.join(DATA_DIR, "df_coreSGP.csv")

OUTPUT_DIR = os.path.join(DATA_DIR, "semopy_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(INPUT_CSV)

print("Loaded:", df.shape)

# drop rows with missing values (CB-SEM requirement unless FIML used)
df = df.dropna()
print("After NA drop:", df.shape)

from sklearn.preprocessing import StandardScaler

cols_to_scale = df.columns  # all numeric
df_z = pd.DataFrame(
    StandardScaler().fit_transform(df[cols_to_scale]),
    columns=cols_to_scale
)


# =====================================================
# SEM MODEL SPECIFICATION
# lavaan-style syntax
# =====================================================
# !!! YOU MUST ADAPT THIS TO YOUR CONSTRUCTS !!!

model_desc = """
# -------------------------
# Measurement model
# -------------------------

SMP =~ PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH + PV6MATH + PV7MATH + PV8MATH + PV9MATH + PV10MATH
SMS =~ ST268Q01JA + ST268Q04JA + ST268Q07JA
SPI =~ SC064Q01TA + SC064Q02TA + SC064Q03TA + SC064Q04NA + SC064Q05WA + SC064Q06WA + SC064Q07WA

# -------------------------
# Structural model
# -------------------------

SMS ~ SPI + ST004D01T + ST001D01T + MISCED + ESCS + AGE + IMMIG + MCLSIZE + SCHSIZE
SMP ~ SPI + SMS + ST004D01T + ST001D01T + MISCED + ESCS + AGE + IMMIG + MCLSIZE + SCHSIZE
"""


# =====================================================
# FIT MODEL
# =====================================================

print("Fitting SEM model...")

model = Model(model_desc)
res = model.fit(df_z)

print("Optimization result:", res)

# =====================================================
# PARAMETER ESTIMATES
# =====================================================

est = model.inspect(std_est=True)

# split useful tables
struct_paths = est[
    (est["op"] == "~") &
    (est["lval"].isin(["SMS", "SMP"]))
].copy()

# ===== SAVE REAL STRUCTURAL PATHS =====

real_struct = est[
    (est["op"] == "~") &
    (est["lval"].isin(["SMS", "SMP"]))
][["lval", "rval", "Est. Std"]].copy()

real_struct["path"] = real_struct["rval"] + " -> " + real_struct["lval"]

real_out = os.path.join(OUTPUT_DIR, "real_structural_paths.csv")
real_struct.to_csv(real_out, index=False)
print("Saved real structural paths to:", real_out)

