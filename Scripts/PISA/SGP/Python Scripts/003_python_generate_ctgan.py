# python_step3_generate_ctgan.py

import os
import random
import numpy as np
import pandas as pd
import torch
import logging
import time

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

DATA_DIR = "/nesi/project/vuw04485/pisa_sem_pipeline/data/"
CNT = "SGP"
INPUT_CSV = os.path.join(DATA_DIR, f"df_core{CNT}.csv")

LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


logger.info("=== Step 3: CTGAN multi-run synthetic data ===")
df_core = pd.read_csv(INPUT_CSV)
logger.info("Loaded input CSV: %s", INPUT_CSV)
logger.info("df_core shape: %s", df_core.shape)


categorical_columns = [
    "ST001D01T", 
    "ST004D01T",
    "ST268Q01JA",
    "ST268Q04JA",
    "ST268Q07JA",
    "IMMIG",
    "MISCED"  
]

logger.info("Categorical columns (%d): %s", len(categorical_columns), categorical_columns)

metadata = SingleTableMetadata()


num_cat = 0
num_num = 0

for col in df_core.columns:
    if col in categorical_columns:
        metadata.add_column(col, sdtype="categorical")
        num_cat += 1
    else:
        metadata.add_column(col, sdtype="numerical")
        num_num += 1

logger.info("Metadata constructed: %d categorical, %d numerical columns", num_cat, num_num)

metadata.validate()
logger.info("Metadata validation successful")


seeds = [42, 101, 202, 303, 404]


logger.info("Starting CTGAN multi-run with %d seeds", len(seeds))

for i, seed in enumerate(seeds):
    logger.info("CTGAN run %d/%d | seed=%d", i + 1, len(seeds), seed)
    run_start = time.time()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ctgan = CTGANSynthesizer(
        metadata,
        enforce_min_max_values=True,
        epochs=500,
        batch_size=64,
        pac=1
    )
    logger.info("Fitting CTGAN (epochs=%d, batch_size=%d, pac=%d)", 500, 64, 1)
    ctgan.fit(df_core)
    logger.info("CTGAN fit complete")

    logger.info("Sampling synthetic data: n_rows=%d", len(df_core))
    synth_ctgan = ctgan.sample(num_rows=len(df_core)).copy()
    logger.info("Sampling complete")

    out_csv     = os.path.join(DATA_DIR, f"synthetic_ctgan_seed_{CNT}_{seed}.csv")

    synth_ctgan.to_csv(out_csv, index=False)
    logger.info("Saved CTGAN synthetic CSV: %s", out_csv)

    elapsed = time.time() - run_start
    logger.info("Run %d completed in %.2f seconds", i + 1, elapsed)


logger.info("All CTGAN runs complete successfully")

