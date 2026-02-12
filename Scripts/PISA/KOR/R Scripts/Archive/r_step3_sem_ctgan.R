# r_step3_sem_ctgan.R
# Dynamic SEM for CTGAN synthetic datasets across multiple seeds

library(lavaan)
library(psych)
library(dplyr)
library(readr)
library(tidyr)

DATA_DIR <- "/nesi/project/vuw04485/pisa_sem_pipeline/data/"

cat("\n=== Step 3: CTGAN SEM (Dynamic) ===\n")

# --------------------------------------------------
# 1. Load REAL data and fit REAL SEM (fixed model)
# --------------------------------------------------

df_real <- read.csv(file.path(DATA_DIR, "df_core.csv"))

model_real <- '
SMP =~ PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH +
       PV6MATH + PV7MATH + PV8MATH + PV9MATH + PV10MATH
SMS =~ ST268Q01JA + ST268Q04JA + ST268Q07JA
SPI =~ SC064Q01TA + SC064Q02TA + SC064Q03TA + SC064Q04NA +
       SC064Q05WA + SC064Q06WA

SMS ~ a*SPI + c1*female + c2*ESCS_z + c3*ST001D01T + c4*MISCED + c5*SCHSIZE_z
SMP ~ b*SMS + cprime*SPI + d1*female + d2*ESCS_z + d3*ST001D01T + d4*MISCED + d5*SCHSIZE_z

indirect := a*b
total := cprime + (a*b)
'

cat("\n--- Fitting REAL SEM (fixed model) ---\n")
fit_real <- sem(model_real, data = df_real, std.lv = TRUE, fixed.x = FALSE, missing = "fiml")
pe_real  <- parameterEstimates(fit_real, standardized = TRUE)

# Extract REAL loadings for Tucker congruence comparison
load_tbl <- function(std_solution) {
  std_solution[std_solution$op == "=~", c("lhs","rhs","std.all")] %>%
    rename(latent = lhs, indicator = rhs, loading = std.all)
}

real_load <- load_tbl(pe_real)

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

reliability_table <- function(std_solution, df_for_alpha) {
  loads <- std_solution[std_solution$op == "=~", c("lhs","rhs","std.all")]
  colnames(loads) <- c("latent","indicator","loading")
  
  out <- lapply(unique(loads$latent), function(lat){
    sub <- loads[loads$latent == lat, ]
    lam <- as.numeric(sub$loading)
    CR  <- (sum(lam)^2) / ((sum(lam)^2) + sum(1 - lam^2))
    AVE <- sum(lam^2) / length(lam)
    alpha_val <- tryCatch({
      psych::alpha(df_for_alpha[, sub$indicator])$total$raw_alpha
    }, error = function(e) NA_real_)
    
    data.frame(
      Construct = lat,
      Cronbach_alpha = round(alpha_val, 3),
      Composite_Reliability = round(CR, 3),
      AVE = round(AVE, 3)
    )
  })
  bind_rows(out)
}

tucker_congruence <- function(load_real, load_syn) {
  merged <- merge(load_real, load_syn,
                  by = c("latent","indicator"),
                  suffixes = c("_real","_syn"))
  out <- lapply(unique(merged$latent), function(lat){
    sub <- merged[merged$latent == lat, ]
    num <- sum(sub$loading_real * sub$loading_syn)
    den <- sqrt(sum(sub$loading_real^2) * sum(sub$loading_syn^2))
    data.frame(
      Construct = lat,
      Tucker_Congruence = ifelse(den > 0, round(num / den, 3), NA_real_)
    )
  })
  bind_rows(out)
}

structural_paths_table <- function(std_solution) {
  std_solution[std_solution$op == "~", ] %>%
    select(lhs, rhs, est, se, pvalue, std.all) %>%
    rename(DV = lhs, IV = rhs, B_unstd = est, B_std = std.all)
}

# --------------------------------------------------
# 2. CTGAN settings
# --------------------------------------------------

SMP_items_full <- paste0("PV", 1:10, "MATH")
SMS_items_full <- c("ST268Q01JA", "ST268Q04JA", "ST268Q07JA")
SPI_items_full <- c("SC064Q01TA","SC064Q02TA","SC064Q03TA",
                    "SC064Q04NA","SC064Q05WA","SC064Q06WA")

model_syn_template <- '
SMP =~ {SMP_ITEMS}
SMS =~ {SMS_ITEMS}
SPI =~ {SPI_ITEMS}

SMS ~ a*SPI + c1*female + c2*ESCS_z + c3*ST001D01T + c4*MISCED + c5*SCHSIZE_z
SMP ~ b*SMS + cprime*SPI + d1*female + d2*ESCS_z + d3*ST001D01T + d4*MISCED + d5*SCHSIZE_z

indirect := a*b
total := cprime + (a*b)
'

seeds <- c(42, 101, 202, 303, 404)

results_list  <- list()
paths_list    <- list()
rsq_list      <- list()
reliab_list   <- list()
tucker_list   <- list()

# --------------------------------------------------
# 3. Loop through CTGAN seeds
# --------------------------------------------------

for (s in seeds) {
  cat("\n--- SEM on CTGAN seed", s, "---\n")
  fn <- file.path(DATA_DIR, paste0("synthetic_ctgan_seed", s, ".csv"))
  
  if (!file.exists(fn)) {
    warning("Missing synthetic dataset for seed ", s)
    next
  }
  
  df_syn <- read.csv(fn)
  
  # Remove zero-variance columns
  zero_var <- names(which(sapply(df_syn, function(x) var(x, na.rm=TRUE)) == 0))
  if (length(zero_var) > 0) {
    cat("Removed zero-variance columns:", paste(zero_var, collapse=", "), "\n")
    df_syn <- df_syn[, !(colnames(df_syn) %in% zero_var)]
  }
  
  # Keep only present indicators
  SMP_items <- SMP_items_full[SMP_items_full %in% colnames(df_syn)]
  SMS_items <- SMS_items_full[SMS_items_full %in% colnames(df_syn)]
  SPI_items <- SPI_items_full[SPI_items_full %in% colnames(df_syn)]
  
  if (length(SMP_items) < 2 || length(SMS_items) < 2 || length(SPI_items) < 2) {
    warning("Insufficient indicators for seed ", s, " — skipping.")
    next
  }
  
  # Build dynamic model
  model_syn <- model_syn_template
  model_syn <- sub("{SMP_ITEMS}", paste(SMP_items, collapse=" + "), model_syn, fixed=TRUE)
  model_syn <- sub("{SMS_ITEMS}", paste(SMS_items, collapse=" + "), model_syn, fixed=TRUE)
  model_syn <- sub("{SPI_ITEMS}", paste(SPI_items, collapse=" + "), model_syn, fixed=TRUE)
  
  # Fit SEM
  fit_syn <- sem(model_syn, data=df_syn, std.lv=TRUE, fixed.x=FALSE, missing="fiml")
  fm_syn  <- fitMeasures(fit_syn)
  pe_syn  <- parameterEstimates(fit_syn, standardized=TRUE)
  
  # Extract
  syn_load <- load_tbl(pe_syn)
  tc       <- tucker_congruence(real_load, syn_load)
  rel_syn  <- reliability_table(pe_syn, df_syn)
  paths    <- structural_paths_table(pe_syn)
  paths$Seed <- s
  
  rsq_syn <- lavInspect(fit_syn, "rsquare")
  rsq_df <- data.frame(
    Seed = s,
    Construct = names(rsq_syn),
    R2 = as.numeric(rsq_syn)
  )
  
  rel_syn$Seed <- s
  tc$Seed <- s
  
  # Collect
  results_list[[as.character(s)]] <- data.frame(
    Seed = s,
    CFI = fm_syn["cfi"],
    TLI = fm_syn["tli"],
    RMSEA = fm_syn["rmsea"],
    SRMR = fm_syn["srmr"]
  )
  
  paths_list[[as.character(s)]]  <- paths
  rsq_list[[as.character(s)]]    <- rsq_df
  reliab_list[[as.character(s)]] <- rel_syn
  tucker_list[[as.character(s)]] <- tc
}

# --------------------------------------------------
# 4. Save outputs
# --------------------------------------------------

results_df <- bind_rows(results_list)
paths_df   <- bind_rows(paths_list)
rsq_df_all <- bind_rows(rsq_list)
reliab_df_all <- bind_rows(reliab_list)
tucker_df_all <- bind_rows(tucker_list)

write_csv(results_df,    file.path(DATA_DIR, "CTGAN_multi_run_metrics.csv"))
write_csv(paths_df,      file.path(DATA_DIR, "CTGAN_structural_paths_multi_run.csv"))
write_csv(rsq_df_all,    file.path(DATA_DIR, "CTGAN_rsquare_multi_run.csv"))
write_csv(reliab_df_all, file.path(DATA_DIR, "CTGAN_reliability_multi_run.csv"))
write_csv(tucker_df_all, file.path(DATA_DIR, "CTGAN_tucker_multi_run.csv"))

# Average CTGAN paths
avg_paths <- paths_df %>%
  group_by(DV, IV) %>%
  summarise(
    B_std_mean  = mean(B_std, na.rm=TRUE),
    B_std_sd    = sd(B_std, na.rm=TRUE),
    pvalue_mean = mean(pvalue, na.rm=TRUE),
    .groups = "drop"
  )

write_csv(avg_paths, file.path(DATA_DIR, "CTGAN_avg_structural_paths_multi_run.csv"))

cat("\n✔ SEM CTGAN multi-run dynamic analysis COMPLETE.\n")
