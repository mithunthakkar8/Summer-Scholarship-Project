import os
import pyreadstat
import pandas as pd
import logging

# =============================
# LOGGING SETUP
# =============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger(__name__)

# =============================
# CONFIG
# =============================

DATA_DIR = r"C:\Users\mithu\Downloads\T23_Data_SAS_G8\SAS Data"

ssf_items = ["BTBGSOS", "BTDGSOS"]
tci_items = ["BTBS17A", "BTBS17B", "BTBS17C", "BTBS17D"]
lei_items = [f"TBG13{c}" for c in "ABCDEFGHI"]
acm_items = ["TBG06I", "TBG06J", "TBG06K"]
nsc_item = ["BTBG10"]

REQUIRED_VARS = ssf_items + tci_items + lei_items + acm_items + nsc_item

results = []
all_cols = set()


# =============================
# SCAN LOOP
# =============================

for fname in os.listdir(DATA_DIR):
    if not fname.lower().endswith(".sas7bdat"):
        continue

    fpath = os.path.join(DATA_DIR, fname)
    logger.info("Inspecting file: %s", fname)

    try:
        _, meta = pyreadstat.read_sas7bdat(fpath, metadataonly=True)

        cols = set(meta.column_names)
        all_cols |= cols
        present = [v for v in REQUIRED_VARS if v in cols]
        missing = [v for v in REQUIRED_VARS if v not in cols]

        logger.info(
            "Rows=%s | TotalCols=%s | RequiredPresent=%s",
            meta.number_rows,
            len(cols),
            len(present),
        )

        if present:
            logger.info("  Present vars: %s", present)
        else:
            logger.warning("  No required variables found")

        if missing:
            logger.debug("  Missing vars: %s", missing)

        results.append({
            "file": fname,
            "rows": meta.number_rows,
            "n_columns": len(cols),
            "n_required_present": len(present),
            "present_vars": ", ".join(present),
            "missing_vars": ", ".join(missing),
        })

    except Exception as e:
        logger.error("Failed to read %s: %s", fname, e)
        results.append({
            "file": fname,
            "rows": None,
            "n_columns": None,
            "n_required_present": 0,
            "present_vars": "",
            "missing_vars": "ERROR",
            "error": str(e),
        })
        
# =============================
# UNIQUE COLUMNS ACROSS ALL FILES
# =============================
unique_cols_sorted = sorted(all_cols)

out_unique_csv = os.path.join(DATA_DIR, "sas_unique_columns_all_files.csv")
pd.DataFrame({"column": unique_cols_sorted}).to_csv(out_unique_csv, index=False)

out_unique_txt = os.path.join(DATA_DIR, "sas_unique_columns_all_files.txt")
with open(out_unique_txt, "w", encoding="utf-8") as f:
    for c in unique_cols_sorted:
        f.write(c + "\n")

logger.info("Unique columns across all files: %d", len(unique_cols_sorted))
logger.info("Saved unique columns CSV: %s", out_unique_csv)
logger.info("Saved unique columns TXT: %s", out_unique_txt)

# =============================
# SUMMARY
# =============================

df_report = pd.DataFrame(results).sort_values(
    by=["n_required_present", "rows"],
    ascending=[False, False]
)

logger.info("=== FILE SUMMARY ===")
logger.info("\n%s", df_report[["file", "rows", "n_columns", "n_required_present"]])

# Optional: save full diagnostics
out_csv = os.path.join(DATA_DIR, "sas_variable_audit.csv")
df_report.to_csv(out_csv, index=False)

print("\n✔ Full audit saved to:", out_csv)