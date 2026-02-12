import pyreadstat
import psycopg2
import pandas as pd
import io
import os
import time

conn = psycopg2.connect(
    dbname="PISA_2022",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

DATA_DIR = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Datasets\PISA 2022"
student_path = os.path.join(DATA_DIR, "CY08MSP_STU_QQQ.SAS7BDAT")

_, meta = pyreadstat.read_sas7bdat(student_path, metadataonly=True)
cols = meta.column_names

print("Total columns:", len(cols))


ID_COLS = ["CNT", "CNTSCHID", "CNTSTUID"]

OTHER_COLS = [c for c in cols if c not in ID_COLS]

N_GROUPS = 5
group_size = len(OTHER_COLS) // N_GROUPS + 1

column_groups = [
    OTHER_COLS[i:i + group_size]
    for i in range(0, len(OTHER_COLS), group_size)
]

print([len(g) for g in column_groups])


# 1. Create key table
cur.execute("""
DROP TABLE IF EXISTS pisa_students_keys;
CREATE TABLE pisa_students_keys (
    CNT TEXT,
    CNTSCHID TEXT,
    CNTSTUID TEXT
);
""")

# 2. Create 5 subtables
for i, group in enumerate(column_groups, start=1):
    col_defs = ", ".join([f'"{c}" TEXT' for c in group])
    sql = f"""
    DROP TABLE IF EXISTS pisa_students_part{i};
    CREATE TABLE pisa_students_part{i} (
        CNT TEXT,
        CNTSCHID TEXT,
        CNTSTUID TEXT,
        {col_defs}
    );
    """
    cur.execute(sql)

conn.commit()
print("Tables created.")


chunk_size = 10000
offset = 0
start_time = time.time()
total_rows = 0

print("=== BEGIN LOADING SAS INTO 5 SUBTABLES ===")

while True:
    df, meta = pyreadstat.read_sas7bdat(
        student_path,
        row_offset=offset,
        row_limit=chunk_size
    )

    if df.empty:
        break

    total_rows += len(df)

    # Insert keys
    keys_df = df[ID_COLS]
    buf = io.StringIO()
    keys_df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    cur.copy_expert("COPY pisa_students_keys FROM STDIN CSV", buf)

    # Insert each part
    for i, group in enumerate(column_groups, start=1):
        sub_df = pd.concat([df[ID_COLS], df[group]], axis=1)
        sub_buf = io.StringIO()
        sub_df.to_csv(sub_buf, index=False, header=False)
        sub_buf.seek(0)
        cur.copy_expert(f"COPY pisa_students_part{i} FROM STDIN CSV", sub_buf)

    conn.commit()

    print(f"Inserted {total_rows} rows...", end="\r")

    offset += chunk_size

elapsed = time.time() - start_time
print("\n=== FINISHED ===")
print("Total student rows processed:", total_rows)
print(f"Time taken: {elapsed/60:.2f} minutes")

cur.close()
conn.close()




