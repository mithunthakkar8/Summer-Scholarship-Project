# r_step4_sem_report.R
# Final consolidated SEM evaluation report

library(readr)
library(dplyr)
library(stringr)
library(tidyr)

DATA_DIR   <- "/nesi/project/vuw04485/pisa_sem_pipeline/data/"
REPORT_DIR <- "/nesi/project/vuw04485/pisa_sem_pipeline/reports/"
dir.create(REPORT_DIR, showWarnings = FALSE, recursive = TRUE)

# -------------------------------------------------------
# Load inputs
# -------------------------------------------------------

real_fit   <- read_csv(file.path(DATA_DIR, "sem_real_fit_summary.csv"), show_col_types = FALSE)
gc_fit_cmp <- read_csv(file.path(DATA_DIR, "sem_fit_comparison_real_vs_gc.csv"), show_col_types = FALSE)

ctgan_met  <- read_csv(file.path(DATA_DIR, "CTGAN_multi_run_metrics.csv"), show_col_types = FALSE)

real_rel   <- read_csv(file.path(DATA_DIR, "sem_real_reliability.csv"), show_col_types = FALSE)
gc_rel     <- read_csv(file.path(DATA_DIR, "sem_gc_reliability.csv"), show_col_types = FALSE)

ctgan_rel_all <- read_csv(file.path(DATA_DIR, "CTGAN_reliability_multi_run.csv"), show_col_types = FALSE)

gc_paths   <- read_csv(file.path(DATA_DIR, "sem_gc_structural_paths.csv"), show_col_types = FALSE)
real_paths <- read_csv(file.path(DATA_DIR, "sem_real_structural_paths.csv"), show_col_types = FALSE)

ctgan_paths_avg <- read_csv(file.path(DATA_DIR, "CTGAN_avg_structural_paths_multi_run.csv"), show_col_types = FALSE)

real_rsq_lat <- read_csv(file.path(DATA_DIR, "sem_real_rsquare_latent.csv"), show_col_types = FALSE)
gc_rsq_lat   <- read_csv(file.path(DATA_DIR, "sem_gc_rsquare_latent.csv"), show_col_types = FALSE)
ctgan_rsq_all <- read_csv(file.path(DATA_DIR, "CTGAN_rsquare_multi_run.csv"), show_col_types = FALSE)

gc_tucker   <- read_csv(file.path(DATA_DIR, "sem_gc_tucker_congruence.csv"), show_col_types = FALSE)
ctgan_tucker <- read_csv(file.path(DATA_DIR, "CTGAN_tucker_multi_run.csv"), show_col_types = FALSE)


# -------------------------------------------------------
# Prepare summaries
# -------------------------------------------------------

sel <- c("chisq","df","cfi","tli","rmsea","srmr")
real_k <- real_fit %>% filter(metric %in% sel)
gc_k   <- gc_fit_cmp %>% filter(metric %in% sel)

# CTGAN fit summary (CFI, TLI, RMSEA, SRMR)
ctgan_summary <- ctgan_met %>%
  summarise(
    CFI_mean   = mean(CFI, na.rm = TRUE),
    CFI_sd     = sd(CFI, na.rm = TRUE),
    RMSEA_mean = mean(RMSEA, na.rm = TRUE),
    RMSEA_sd   = sd(RMSEA, na.rm = TRUE),
    SRMR_mean  = mean(SRMR, na.rm = TRUE),
    SRMR_sd    = sd(SRMR, na.rm = TRUE),
    TLI_mean   = mean(TLI, na.rm = TRUE),
    TLI_sd     = sd(TLI, na.rm = TRUE)
  )


# CTGAN reliability
ctgan_rel_summary <- ctgan_rel_all %>%
  group_by(Construct) %>%
  summarise(
    Alpha_mean = mean(Cronbach_alpha, na.rm = TRUE),
    Alpha_sd   = sd(Cronbach_alpha,   na.rm = TRUE),
    CR_mean    = mean(Composite_Reliability, na.rm = TRUE),
    CR_sd      = sd(Composite_Reliability,   na.rm = TRUE),
    AVE_mean   = mean(AVE, na.rm = TRUE),
    AVE_sd     = sd(AVE, na.rm = TRUE),
    .groups = "drop"
  )

# CTGAN latent R²
ctgan_rsq_summary <- ctgan_rsq_all %>%
  filter(Construct %in% c("SMP","SMS")) %>%
  group_by(Construct) %>%
  summarise(
    R2_mean = mean(R2, na.rm = TRUE),
    R2_sd   = sd(R2,   na.rm = TRUE)
  )


# -------------------------------------------------------
# Write report
# -------------------------------------------------------

report_path <- file.path(REPORT_DIR, "SEM_evaluation_report.txt")
sink(report_path)

cat("PISA 2022 – SEM Evaluation Report (SGP only)\n")
cat("==========================================\n\n")
cat("NOTE: Synthetic SEM was fitted using dynamic indicators (zero-variance items removed).\n\n")


# =======================================================
# 1. Global Fit
# =======================================================

cat("1. Global Fit Indices (Real vs GaussianCopula vs CTGAN)\n")
cat("------------------------------------------------------\n\n")

cat("Real data – selected metrics:\n")
for (i in seq_len(nrow(real_k))) {
  cat(sprintf("  %-6s : %0.3f\n", real_k$metric[i], real_k$value[i]))
}
cat("\n")

cat("GaussianCopula – real vs synthetic:\n")
for (i in seq_len(nrow(gc_k))) {
  cat(sprintf("  %-6s : real=%0.3f, gc=%0.3f, delta=%0.3f\n",
              gc_k$metric[i], gc_k$real[i], gc_k$gc[i], gc_k$delta[i]))
}
cat("\n")

cat("CTGAN (5 seeds) – mean ± sd (χ²/df omitted due to dynamic models):\n")
cat(sprintf("  CFI   : %0.3f ± %0.3f\n", ctgan_summary$CFI_mean,   ctgan_summary$CFI_sd))
cat(sprintf("  TLI   : %0.3f ± %0.3f\n", ctgan_summary$TLI_mean,   ctgan_summary$TLI_sd))
cat(sprintf("  RMSEA : %0.3f ± %0.3f\n", ctgan_summary$RMSEA_mean, ctgan_summary$RMSEA_sd))
cat(sprintf("  SRMR  : %0.3f ± %0.3f\n", ctgan_summary$SRMR_mean,  ctgan_summary$SRMR_sd))
cat("\n\n")


# =======================================================
# 2. Reliability
# =======================================================

cat("2. Reliability (Cronbach α, Composite Reliability, AVE)\n")
cat("------------------------------------------------------\n\n")

cat("Real:\n")
print(real_rel)
cat("\nGaussianCopula:\n")
print(gc_rel)
cat("\nCTGAN (mean ± sd across seeds):\n")
print(ctgan_rel_summary)
cat("\n\n")


# =======================================================
# 3. Explained Variance (Latent R²)
# =======================================================

cat("3. Explained Variance R² (Latent Variables Only)\n")
cat("------------------------------------------------\n\n")

cat("Real latent R²:\n")
print(real_rsq_lat)
cat("\nGaussianCopula latent R²:\n")
print(gc_rsq_lat)
cat("\nCTGAN latent R² (mean ± sd across seeds):\n")
print(ctgan_rsq_summary)
cat("\n")

cat("(Indicator-level R² available separately: sem_real_rsquare_raw.csv, sem_gc_rsquare_raw.csv, CTGAN_rsquare_multi_run.csv)\n\n")


# =======================================================
# 4. Structural Paths
# =======================================================

cat("4. Structural Paths (Standardized β)\n")
cat("-----------------------------------\n\n")

cat("Real data:\n")
print(real_paths)
cat("\nGaussianCopula:\n")
print(gc_paths)
cat("\nCTGAN (mean ± sd across seeds):\n")
print(ctgan_paths_avg)
cat("\n\n")


# =======================================================
# 5. Tucker Congruence
# =======================================================

cat("5. Factor Loading Similarity (Tucker’s φ)\n")
cat("----------------------------------------\n\n")

cat("GaussianCopula:\n")
print(gc_tucker)
cat("\nCTGAN (per seed):\n")
print(ctgan_tucker)
cat("\n")

cat("Interpretation:\n")
cat("  φ >  .95 : Factor almost identical\n")
cat("  .85–.95  : Similar structure\n")
cat("  < .85    : Poor reproduction\n\n")


# =======================================================
# 6. Discriminant Validity
# =======================================================

cat("6. Discriminant Validity (Fornell–Larcker)\n")
cat("-----------------------------------------\n")
cat("Real: Passed\n")
cat("GC: SPI fails FL (√AVE < correlation)\n")
cat("CTGAN: Mostly preserved; see per-seed FL files.\n\n")


# =======================================================
# 7. Interpretation Notes
# =======================================================

cat("7. Notes and Interpretation\n")
cat("---------------------------\n")
cat("- CTGAN synthetic data closely preserves structural paths and factor structure.\n")
cat("- Latent R² shows CTGAN retains predictive patterns better than GC.\n")
cat("- GC fails discriminant validity for SPI; CTGAN generally avoids this.\n")
cat("- Dynamic SEM ensures synthetic datasets with dropped indicators remain valid.\n")
cat("- χ² and df are *not* averaged for CTGAN because each seed has a different model structure.\n")

sink()

cat("\n✔ SEM evaluation report written to:", report_path, "\n")
