import pandas as pd

xlsx_path = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Datasets\TIMSS 2023\T23_Codebook_G8.xlsx"  
out_csv   = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Datasets\TIMSS 2023\unique_first_col_values.csv"

# Load workbook
xls = pd.ExcelFile(xlsx_path)

all_vals = []

for sheet in xls.sheet_names:
    # Read ONLY the first column (A) from each sheet, no header assumption
    s = pd.read_excel(xlsx_path, sheet_name=sheet, usecols=[0], header=None).iloc[:, 0]

    # Drop blanks
    s = s.dropna()

    # Convert to string & normalize (optional but recommended)
    s = s.astype(str).str.strip()

    # If you want case-insensitive uniqueness, uncomment:
    # s = s.str.lower()

    # Remove empty strings after strip
    s = s[s != ""]

    all_vals.append(s)

# Combine and unique
combined = pd.concat(all_vals, ignore_index=True)
unique_vals = pd.Series(pd.unique(combined)).sort_values(kind="stable").reset_index(drop=True)

# Save
pd.DataFrame({"unique_first_col_value": unique_vals}).to_csv(out_csv, index=False)

print("Sheets scanned:", len(xls.sheet_names))
print("Total values:", len(combined))
print("Unique values:", len(unique_vals))
print("Saved:", out_csv)
