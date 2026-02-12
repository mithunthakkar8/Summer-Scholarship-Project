import pyreadstat
import psycopg2
import pandas as pd
import io
import os
import time

# -----------------------------------------------------------
# PostgreSQL connection
# -----------------------------------------------------------
conn = psycopg2.connect(
    dbname="PISA_2022",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
DATA_DIR = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Datasets\PISA 2022"
school_path = os.path.join(DATA_DIR, "CY08MSP_SCH_QQQ.SAS7BDAT")

# -----------------------------------------------------------
# Step 1: Read metadata only (to get column names)
# -----------------------------------------------------------
_, meta = pyreadstat.read_sas7bdat(school_path, metadataonly=True)
cols = meta.column_names
print("School columns detected:", len(cols))

# -----------------------------------------------------------
# Step 2: Create PostgreSQL table with all TEXT columns
# -----------------------------------------------------------
column_definitions = ", ".join([f'"{c}" TEXT' for c in cols])

cur.execute(f"""
DROP TABLE IF EXISTS pisa_schools_full;
CREATE TABLE pisa_schools_full (
    {column_definitions}
);
""")
conn.commit()
print("Table pisa_schools_full created.")

# -----------------------------------------------------------
# Step 3: Chunked load using COPY
# -----------------------------------------------------------
chunk_size = 5000
offset = 0
total_rows = 0
start_time = time.time()

print("\n=== BEGIN SCHOOL DATA LOADING ===")

while True:
    df, meta = pyreadstat.read_sas7bdat(
        school_path,
        row_offset=offset,
        row_limit=chunk_size
    )

    if df.empty:
        break

    total_rows += len(df)

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False, header=False)
    csv_buf.seek(0)

    cur.copy_expert("COPY pisa_schools_full FROM STDIN CSV", csv_buf)
    conn.commit()

    print(f"Inserted {total_rows} rows...", end="\r")

    offset += chunk_size

elapsed = time.time() - start_time

print("\n=== SCHOOL FILE LOAD COMPLETE ===")
print("Total rows inserted:", total_rows)
print(f"Time taken: {elapsed/60:.2f} minutes")

cur.close()
conn.close()
