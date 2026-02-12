"""
excel_metadata_extractor.py

Extracts sheet names and full tabular metadata from an Excel workbook:
- Sheet names
- Row headers (index)
- Column headers
- Shape
- Data types
- Missing value counts
- Sample data

Author: Mithun
"""

import json
import pandas as pd
from pathlib import Path


# =========================
# CONFIGURATION
# =========================
EXCEL_FILE = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\SEM_TECHNIQUE_COMPARISON.xlsx"
OUTPUT_DIR = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project"
ROW_HEADER_SAMPLE = 10                # how many row headers to store
DATA_PREVIEW_ROWS = 5                 # rows to preview per sheet


# =========================
# LOAD WORKBOOK
# =========================
xls = pd.ExcelFile(EXCEL_FILE)

output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

all_metadata = []


# =========================
# PROCESS EACH SHEET
# =========================
for sheet_name in xls.sheet_names:
    df = xls.parse(sheet_name)

    sheet_metadata = {
        "sheet_name": sheet_name,

        # ---- shape ----
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),

        # ---- headers ----
        "column_headers": df.columns.astype(str).tolist(),
        "row_headers_sample": df.index.astype(str).tolist()[:ROW_HEADER_SAMPLE],

        # ---- schema ----
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isna().sum().to_dict(),

        # ---- data preview ----
        "data_preview": df.head(DATA_PREVIEW_ROWS).to_dict(orient="records"),
    }

    all_metadata.append(sheet_metadata)


# =========================
# SAVE OUTPUTS
# =========================

# JSON (best for pipelines / audits)
json_path = output_path / "excel_metadata.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, indent=2)

# CSV (flattened summary)
summary_rows = []
for s in all_metadata:
    summary_rows.append({
        "sheet_name": s["sheet_name"],
        "n_rows": s["n_rows"],
        "n_columns": s["n_columns"],
        "columns": ", ".join(s["column_headers"])
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(output_path / "excel_metadata_summary.csv", index=False)

print("Metadata extraction complete.")
print(f"- JSON: {json_path}")
print(f"- CSV:  {output_path / 'excel_metadata_summary.csv'}")
