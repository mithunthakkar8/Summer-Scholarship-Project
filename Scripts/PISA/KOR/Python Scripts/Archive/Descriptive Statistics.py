from ydata_profiling import ProfileReport
import os 
import pandas as pd

DATA_DIR = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Datasets\PISA 2022"
CNT = "SGP"
INPUT_CSV = os.path.join(DATA_DIR, f"df_core{CNT}.csv")

df_core = pd.read_csv(INPUT_CSV)

ProfileReport(df_core, title="Dataset Summary", explorative=True).\
to_file(os.path.join(DATA_DIR, f"Descriptive_Stats_Real_{CNT}.html"))


Technique = 'CTGAN'
Synthetic_DIR = r"C:\Users\mithu\Documents\MEGA\VUW\Summer Research Project\Experiments"
CTGAN_CSV = os.path.join(Synthetic_DIR, rf"CTGAN\synthetic_ctgan_seed_SGP_42.csv")

df = pd.read_csv(CTGAN_CSV)

ProfileReport(df, title="Dataset Summary", explorative=True).\
to_file(os.path.join(Synthetic_DIR, f"Descriptive_Stats_{Technique}.html"))


Technique = 'GC'
GC_CSV = os.path.join(Synthetic_DIR, f"Synthetic_gcSGP.csv")

df = pd.read_csv(GC_CSV)

ProfileReport(df, title="Dataset Summary", explorative=True).\
to_file(os.path.join(Synthetic_DIR, f"Descriptive_Stats_{Technique}.html"))


Technique = 'DistilGPT2'
DistilGPT2_CSV = os.path.join(Synthetic_DIR, rf"GReaT\Baseline\synthetic_great_basic_42.csv")

df = pd.read_csv(DistilGPT2_CSV)

ProfileReport(df, title="Dataset Summary", explorative=True).\
to_file(os.path.join(Synthetic_DIR, f"Descriptive_Stats_{Technique}.html"))



Technique = 'GPT2'
DistilGPT2_CSV = os.path.join(Synthetic_DIR, rf"GReaT\Baseline\synthetic_great_basic_gpt2.csv")

df = pd.read_csv(DistilGPT2_CSV)

ProfileReport(df, title="Dataset Summary", explorative=True).\
to_file(os.path.join(Synthetic_DIR, f"Descriptive_Stats_{Technique}.html"))



Technique = 'GPT2Large'
GPT2Large_CSV = os.path.join(Synthetic_DIR, rf"GReaT\Baseline\synthetic_great_basic_gpt2large.csv")

df = pd.read_csv(GPT2Large_CSV)

ProfileReport(df, title="Dataset Summary", explorative=True).\
to_file(os.path.join(Synthetic_DIR, f"Descriptive_Stats_{Technique}.html"))


