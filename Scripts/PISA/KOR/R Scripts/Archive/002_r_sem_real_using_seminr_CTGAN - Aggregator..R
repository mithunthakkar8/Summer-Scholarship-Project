rm(list = ls())

library(dplyr)
library(readr)
library(stringr)
library(glue)
library(tidyr)
library(purrr)

# ================================================================
# CONFIG
# ================================================================
DATA_DIR <- "C:/Users/mithu/Documents/MEGA/VUW/Summer Research Project/Datasets/PISA 2022/CTGAN"
CNT <- "SGP"
seeds <- c(42, 101, 202, 303, 404)

# Helper to build filenames
seed_tag <- function(seed) glue("SGP_CTGAN_seed{seed}")

# ================================================================
# 1. STANDARDIZED PATH COEFFICIENTS
# ================================================================
paths_all <- map_dfr(seeds, function(seed) {
  f <- file.path(DATA_DIR,
                 glue("sem_full_std_paths_{seed_tag(seed)}.csv"))
  read_csv(f, show_col_types = FALSE) %>%
    mutate(seed = seed)
})

paths_summary <- paths_all %>%
  group_by(Path, IV, DV) %>%
  summarise(
    mean_std_B = mean(Std_B, na.rm = TRUE),
    sd_std_B   = sd(Std_B, na.rm = TRUE),
    min_std_B  = min(Std_B, na.rm = TRUE),
    max_std_B  = max(Std_B, na.rm = TRUE),
    pass_p_05  = mean(Std_p < 0.05, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(paths_summary,
          file.path(DATA_DIR, glue("agg_std_paths_{CNT}.csv")))

# ================================================================
# 2. R-SQUARED
# ================================================================
r2_all <- map_dfr(seeds, function(seed) {
  f <- file.path(DATA_DIR,
                 glue("sem_full_rsquared_{seed_tag(seed)}.csv"))
  read_csv(f, show_col_types = FALSE) %>%
    mutate(seed = seed)
})

r2_summary <- r2_all %>%
  group_by(Construct) %>%
  summarise(
    mean_R2 = mean(R2, na.rm = TRUE),
    sd_R2   = sd(R2, na.rm = TRUE),
    min_R2  = min(R2, na.rm = TRUE),
    max_R2  = max(R2, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(r2_summary,
          file.path(DATA_DIR, glue("agg_r_squared_{CNT}.csv")))

# ================================================================
# 3. RELIABILITY METRICS
# ================================================================
rel_all <- map_dfr(seeds, function(seed) {
  f <- file.path(DATA_DIR,
                 glue("sem_full_reliability_metrics_labeled_{seed_tag(seed)}.csv"))
  read_csv(f, show_col_types = FALSE) %>%
    mutate(seed = seed)
})

rel_summary <- rel_all %>%
  group_by(Construct) %>%
  summarise(
    mean_alpha = mean(alpha, na.rm = TRUE),
    mean_rhoC  = mean(rhoC, na.rm = TRUE),
    mean_AVE   = mean(AVE, na.rm = TRUE),
    pass_AVE   = mean(AVE >= 0.50, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(rel_summary,
          file.path(DATA_DIR, glue("agg_reliability_{CNT}.csv")))

# ================================================================
# 4. HTMT
# ================================================================
htmt_all <- map_dfr(seeds, function(seed) {
  f <- file.path(DATA_DIR,
                 glue("sem_full_htmt_{seed_tag(seed)}.csv"))
  read_csv(f, show_col_types = FALSE) %>%
    mutate(seed = seed)
})

htmt_long <- htmt_all %>%
  pivot_longer(
    cols = -c(Construct, seed),
    names_to = "Other_Construct",
    values_to = "HTMT"
  ) %>%
  filter(Construct != Other_Construct)

htmt_summary <- htmt_long %>%
  group_by(Construct, Other_Construct) %>%
  summarise(
    mean_HTMT = mean(HTMT, na.rm = TRUE),
    max_HTMT  = max(HTMT, na.rm = TRUE),
    pass_090  = mean(HTMT <= 0.90, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(htmt_summary,
          file.path(DATA_DIR, glue("agg_htmt_{CNT}.csv")))

# ================================================================
# 5. INDIRECT EFFECT (MEDIATION)
# ================================================================
indirect_all <- map_dfr(seeds, function(seed) {
  f <- file.path(DATA_DIR,
                 glue("sem_full_indirect_SPI_SMS_SMP_{seed_tag(seed)}.csv"))
  read_csv(f, show_col_types = FALSE) %>%
    mutate(seed = seed)
})

indirect_summary <- indirect_all %>%
  summarise(
    mean_indirect = mean(Indirect_std, na.rm = TRUE),
    sd_indirect   = sd(Indirect_std, na.rm = TRUE),
    min_indirect  = min(Indirect_std, na.rm = TRUE),
    max_indirect  = max(Indirect_std, na.rm = TRUE),
    pass_p_05     = mean(p_value < 0.05, na.rm = TRUE)
  )

write_csv(indirect_summary,
          file.path(DATA_DIR, glue("agg_indirect_SPI_SMS_SMP_{CNT}.csv")))

# ================================================================
# DONE
# ================================================================
cat("\n✅ Seed aggregation completed successfully.\n")
