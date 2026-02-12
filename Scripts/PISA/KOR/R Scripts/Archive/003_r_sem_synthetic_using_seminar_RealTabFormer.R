rm(list = ls())

# ================================================================
# r_step1_sem_real_seminr.R
# PLS-SEM analysis using SEMinR for structural coherence
# ================================================================

library(seminr)
library(dplyr)
library(readr)
library(tidyr)
library(psych)
library(glue)
library(lavaan)  # for CB-SEM fit indices and parameter tables

#DATA_DIR <- "/nesi/project/vuw04485/pisa_sem_pipeline/data"
#EXPERIMENTS_DIR <- "C:/Users/mithu/Documents/MEGA/VUW/Summer Research Project/Experiments/GReaT"
EXPERIMENTS_DIR <- "C:/Users/mithu/Documents/MEGA/VUW/Summer Research Project/Experiments/RealTabFormer/Relational"
DATA_DIR <- "C:/Users/mithu/Documents/MEGA/VUW/Summer Research Project/Datasets/PISA 2022"
#MODEL <- "synthetic_realtabformer_rep_1"
MODEL <- "synthetic_realtabformer_relational"


# ------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------
#df <- read.csv(file.path(EXPERIMENTS_DIR, "synthetic_sem_full_COMBO_C_High_Diversity_Synthetic_Universe.csv"))
input_file <- glue("{MODEL}.csv")

df <- read.csv(
  file.path(EXPERIMENTS_DIR, input_file)
) %>% 
  tidyr::drop_na()


mapping <- read.csv(
  file.path(DATA_DIR, "pisa_variable_mapping.csv"),
  stringsAsFactors = FALSE
)

# Build maps
# safe_short -> code (for converting the incoming dataset to code names)
safe_to_code <- setNames(mapping$code, mapping$safe_short)

# code -> safe_short (you already need this later for exporting)
code_to_safe <- setNames(mapping$safe_short, mapping$code)

# Rename columns: safe_short -> code (only where mapping exists)
old_names <- names(df)
new_names <- ifelse(old_names %in% names(safe_to_code), safe_to_code[old_names], old_names)
names(df) <- new_names

# Optional: warn if some columns were not mapped (helps catch issues)
unmapped <- setdiff(old_names, names(safe_to_code))
if (length(unmapped) > 0) {
  message("Note: these input columns were not found in mapping (left unchanged):")
  message(paste(unmapped, collapse = ", "))
}

# Sanity check for required variables
stopifnot(all(c(
  "ST268Q01JA","ST268Q04JA","ST268Q07JA",
  "SC064Q01TA","SC064Q02TA","SC064Q03TA",
  "SC064Q05WA","SC064Q06WA","SC064Q04NA", "SC064Q07WA",
  paste0("PV", 1:10, "MATH")
) %in% names(df)))


# ------------------------------------------------
# 2. MEDIAN IMPUTATION FOR MISSING DATA
# ------------------------------------------------
control_vars <- c("ST004D01T", "ST001D01T", "MISCED", "ESCS", 
                  "AGE", "IMMIG", "MCLSIZE", "SCHSIZE")

print("Missing data before imputation:")
missing_before <- sapply(df[control_vars], function(x) sum(is.na(x)))
print(missing_before)

median_impute <- function(x) {
  if (any(is.na(x))) {
    x_imputed <- x
    x_imputed[is.na(x)] <- median(x, na.rm = TRUE)
    return(x_imputed)
  } else {
    return(x)
  }
}

df_clean <- df
for (var in control_vars) {
  df_clean[[var]] <- median_impute(df_clean[[var]])
}

print("\nMissing data after median imputation:")
missing_after <- sapply(df_clean[control_vars], function(x) sum(is.na(x)))
print(missing_after)


# ------------------------------------------------
# EXPORT MISSING VALUE SUMMARY
# ------------------------------------------------

missing_summary <- data.frame(
  Variable = control_vars,
  Missing_Before = as.numeric(missing_before),
  Missing_After  = as.numeric(missing_after)
)

print("\nMissing-value summary (before vs after imputation):")
print(missing_summary)

write_csv(
  missing_summary,
  file.path(DATA_DIR, glue("sem_missing_summary_{MODEL}.csv"))
)


# ------------------------------------------------
# 3. DEFINE SEMinR MEASUREMENT & STRUCTURAL MODELS
# ------------------------------------------------
mm <- constructs(
  composite("SMP", c(
    "PV1MATH","PV2MATH","PV3MATH","PV4MATH","PV5MATH",
    "PV6MATH","PV7MATH","PV8MATH","PV9MATH","PV10MATH"
  )),
  
  composite("SMS", c(
    "ST268Q01JA","ST268Q04JA","ST268Q07JA"
  )),
  
  composite("SPI", c(
    "SC064Q01TA","SC064Q02TA","SC064Q03TA",
    "SC064Q04NA","SC064Q05WA","SC064Q06WA", "SC064Q07WA"
  )),
  
  composite("gender",      "ST004D01T"),
  composite("grade",       "ST001D01T"),
  composite("motherEdu",   "MISCED"),
  composite("ESCS",        "ESCS"),
  composite("age",         "AGE"),
  composite("immig",       "IMMIG"),
  composite("classSize",   "MCLSIZE"),
  composite("schoolSize",  "SCHSIZE")
)

sm <- relationships(
  paths(
    from = c("SPI", "gender", "grade", "motherEdu", "ESCS", 
             "age", "immig", "classSize", "schoolSize"),
    to = "SMS"
  ),
  
  paths(
    from = c("SPI", "SMS", "gender", "grade", "motherEdu", "ESCS", 
             "age", "immig", "classSize", "schoolSize"),
    to = "SMP"
  )
)

# ------------------------------------------------
# 4. ESTIMATE PLS-SEM + PLSc ADJUSTMENT
# ------------------------------------------------
pls_model <- estimate_pls(
  data = df_clean,
  measurement_model = mm,
  structural_model = sm,
  inner_weights = path_weighting
)

print("Model estimated successfully!")

pls_model <- PLSc(pls_model)
print("PLSc correction applied successfully!")
# 
sum_pls <- summary(pls_model)

# ------------------------------------------------
# 5. BOOTSTRAP FOR SIGNIFICANCE
# ------------------------------------------------
print("Starting bootstrap with 1000 iterations...")
boot_pls <- bootstrap_model(
  seminr_model = pls_model,
  nboot = 1000,
  cores = 4,
  seed = 131
)


print("Bootstrap completed!")
boot_sum <- summary(boot_pls)

# ------------------------------------------------
# 6. SPECIFIC INDIRECT EFFECT
# ------------------------------------------------
indirect_spi_sms_smp <- tryCatch({
  specific_effect_significance(
    boot_pls,
    from = "SPI",
    through = "SMS",
    to = "SMP",
    alpha = 0.05
  )
}, error = function(e) {
  print(paste("Indirect effect calculation failed:", e$message))
  NULL
})

if (!is.null(indirect_spi_sms_smp)) {
  print("Indirect effect calculated successfully:")
  print(indirect_spi_sms_smp)
  
  indirect_tbl <- data.frame(
    IV = "SPI",
    Mediator = "SMS",
    DV = "SMP",
    Indirect_B = indirect_spi_sms_smp["Original Est."],
    Bootstrap_Mean = indirect_spi_sms_smp["Bootstrap Mean"],
    Std_err = indirect_spi_sms_smp["Bootstrap SD"],
    t_stat = indirect_spi_sms_smp["T Stat."],
    CI_95L = indirect_spi_sms_smp["2.5% CI"],
    CI_95U = indirect_spi_sms_smp["97.5% CI"]
  )
  
  print(indirect_tbl)
  
  latent_sds <- apply(sum_pls$composite_scores, 2, sd)
  indirect_std <- indirect_spi_sms_smp["Original Est."] * 
    (latent_sds["SPI"] / latent_sds["SMP"])
  
  indirect_tbl$Indirect_std <- indirect_std
  
  # ---- Add p-value for indirect effect ----
  if (!is.null(indirect_spi_sms_smp)) {
    t_val <- indirect_tbl$t_stat
    p_val <- 2 * (1 - pnorm(abs(t_val)))
    indirect_tbl$p_value <- p_val
  }
  
  
  write_csv(
    indirect_tbl,
    file.path(DATA_DIR, glue("sem_full_indirect_SPI_SMS_SMP_{MODEL}.csv"))
  )
}

# ------------------------------------------------
# 7. STANDARDIZED PATH COEFFICIENTS
# ------------------------------------------------
latent_df <- as.data.frame(sum_pls$composite_scores)
lv_sds <- apply(latent_df, 2, sd)
print("\nStandard deviations of latent variables:")
print(lv_sds)

print("\nGetting path coefficients from bootstrapped summary...")

paths_std <- NULL

if (!is.null(boot_sum$bootstrapped_paths)) {
  boot_paths_df <- as.data.frame(boot_sum$bootstrapped_paths)
  
  structural_paths <- boot_paths_df[grepl("->", rownames(boot_paths_df)), ]
  
  if (nrow(structural_paths) > 0) {
    paths_std <- structural_paths %>%
      mutate(
        Path = rownames(.),
        IV = gsub("\\s*->.*", "", Path),
        DV = gsub(".*->\\s*", "", Path),
        IV = trimws(IV),
        DV = trimws(DV),
        
        Original_Est = `Original Est.`,
        Bootstrap_SD = `Bootstrap SD`,
        
        Std_B = ifelse(IV %in% names(lv_sds) & DV %in% names(lv_sds),
                       Original_Est * (lv_sds[IV] / lv_sds[DV]),
                       Original_Est),
        Std_SE = ifelse(IV %in% names(lv_sds) & DV %in% names(lv_sds),
                        Bootstrap_SD * (lv_sds[IV] / lv_sds[DV]),
                        Bootstrap_SD),
        Std_t = Std_B / Std_SE,
        Std_p = 2 * (1 - pnorm(abs(Std_t)))
      ) %>%
      select(Path, DV, IV, Original_Est, Bootstrap_SD, Std_B, Std_SE, Std_t, Std_p)
    
    print("\nStandardized path coefficients:")
    print(paths_std)
    
    write_csv(
      paths_std,
      file.path(DATA_DIR, glue("sem_full_std_paths_{MODEL}.csv"))
    )
    
    write_csv(
      structural_paths,
      file.path(DATA_DIR, glue("sem_full_unstd_paths_{MODEL}.csv"))
    )
  }
}

# ------------------------------------------------
# 8. R-SQUARED VALUES (LATENT R²)
# ------------------------------------------------
print("\nExtracting R-squared values...")

if (!is.null(sum_pls$paths)) {
  r_squared <- data.frame(
    Construct = colnames(sum_pls$paths),
    R2 = sum_pls$paths[1, ],
    Adj_R2 = sum_pls$paths[2, ]
  )
  
  print("R-squared values:")
  print(r_squared)
  
  write_csv(
    r_squared,
    file.path(DATA_DIR, glue("sem_full_rsquared_{MODEL}.csv"))
  )
}

# ------------------------------------------------
# 9. TOTAL EFFECTS
# ------------------------------------------------
print("\nExtracting total effects...")

if (!is.null(boot_sum$bootstrapped_total_paths)) {
  total_effects <- as.data.frame(boot_sum$bootstrapped_total_paths)
  
  print("Total effects:")
  print(total_effects)
  
  write_csv(
    total_effects,
    file.path(DATA_DIR, glue("sem_full_total_effects_{MODEL}.csv"))
  )
}

# ------------------------------------------------
# 10. COMPREHENSIVE STRUCTURAL COHERENCE METRICS
# ------------------------------------------------
print("\nComputing comprehensive structural coherence metrics...")

# ------------------------------------------------
# 10.1 Global PLS-SEM fit: SRMR (manual computation)
# ------------------------------------------------

srmr_vals <- tryCatch({
  
  # empirical correlation matrix
  Sigma_hat <- cor(df_clean, use = "pairwise.complete.obs")
  
  # model-implied correlation matrix from PLS
  model_cov <- pls_model$construct_scores
  Sigma_model <- cor(model_cov, use = "pairwise.complete.obs")
  
  # ensure dimensions match by selecting only constructs
  common_constructs <- intersect(colnames(Sigma_model), colnames(Sigma_hat))
  Sigma_hat_c <- Sigma_hat[common_constructs, common_constructs]
  Sigma_model_c <- Sigma_model[common_constructs, common_constructs]
  
  # SRMR = sqrt( mean( (population minus model)^2 ) )
  srmr_val <- sqrt(mean((Sigma_hat_c - Sigma_model_c)^2, na.rm = TRUE))
  
  return(c(SRMR = srmr_val))
  
}, error = function(e) {
  print(paste("SRMR computation failed:", e$message))
  NULL
})

if (!is.null(srmr_vals)) {
  srmr_df <- data.frame(
    Metric = names(srmr_vals),
    Value  = as.numeric(srmr_vals)
  )
  
  print("\nPLS-SEM SRMR (manual):")
  print(srmr_df)
  
  write_csv(
    srmr_df,
    file.path(DATA_DIR, glue("sem_full_pls_srmr_{MODEL}.csv"))
  )
}


# 10.3 Reliability & Validity Metrics (α, CR, AVE, Fornell-Larcker, HTMT)

# 10.3.1 Reliability table (alpha, rhoC, AVE, rhoA)
if (!is.null(sum_pls$reliability)) {
  reliability_metrics <- as.data.frame(sum_pls$reliability)
  print("\nReliability metrics (alpha, rhoC, AVE, rhoA):")
  print(reliability_metrics)
  
  # ---- Add construct names to reliability table ----
  reliability_metrics$Construct <- rownames(sum_pls$reliability)
  reliability_metrics <- reliability_metrics %>% 
    relocate(Construct)
  
  # Print clean table
  print("\nCleaned reliability table with construct names:")
  print(reliability_metrics)
  
  # Save to CSV
  write_csv(
    reliability_metrics,
    file.path(DATA_DIR, glue("sem_full_reliability_metrics_labeled_{MODEL}.csv"))
  )
  
}

# 10.3.2 Fornell-Larcker criterion
if (!is.null(sum_pls$validity$fl_criteria)) {
  fl_mat <- sum_pls$validity$fl_criteria
  fl_df <- as.data.frame(fl_mat)
  fl_df$Construct <- rownames(fl_mat)
  fl_df <- fl_df %>% relocate(Construct)
  
  print("\nFornell-Larcker criterion matrix:")
  print(fl_df)
  
  write_csv(
    fl_df,
    file.path(DATA_DIR, glue("sem_full_fornell_larcker_{MODEL}.csv"))
  )
}

# 10.3.3 HTMT matrix
if (!is.null(sum_pls$validity$htmt)) {
  htmt_mat <- sum_pls$validity$htmt
  htmt_df <- as.data.frame(htmt_mat)
  htmt_df$Construct <- rownames(htmt_mat)
  htmt_df <- htmt_df %>% relocate(Construct)
  
  print("\nHTMT matrix:")
  print(htmt_df)
  
  write_csv(
    htmt_df,
    file.path(DATA_DIR, glue("sem_full_htmt_{MODEL}.csv"))
  )
}

# ------------------------------------------------
# LOAD VARIABLE NAME MAPPING
# ------------------------------------------------
name_map <- code_to_safe


latent_scores <- as.data.frame(sum_pls$composite_scores)



# ------------------------------------------------
# REPLACE COUNTRY CODES WITH FULL COUNTRY NAMES
# ------------------------------------------------





# # 10.4.2 Structural residuals for SMS and SMP (on latent scores)
# sms_formula <- SMS ~ SPI + gender + grade + motherEdu + ESCS + 
#   age + immig + classSize + schoolSize
# 
# smp_formula <- SMP ~ SPI + SMS + gender + grade + motherEdu + ESCS + 
#   age + immig + classSize + schoolSize
# 
# sms_lm <- lm(sms_formula, data = latent_scores)
# smp_lm <- lm(smp_formula, data = latent_scores)
# 
# struct_resid <- data.frame(
#   row_id   = seq_len(nrow(latent_scores)),
#   SMS_resid = resid(sms_lm),
#   SMP_resid = resid(smp_lm)
# )
# 
# write_csv(
#   struct_resid,
#   file.path(DATA_DIR, glue("sem_full_structural_residuals_SMS_SMP_{MODEL}.csv"))
# )

# 10.4.3 Latent correlation matrix between SMP, SMS, SPI
if (all(c("SMP","SMS","SPI") %in% colnames(latent_scores))) {
  latent_corr <- cor(latent_scores[, c("SMP","SMS","SPI")],
                     use = "pairwise.complete.obs")
  latent_corr_df <- as.data.frame(latent_corr)
  latent_corr_df$Construct <- rownames(latent_corr)
  latent_corr_df <- latent_corr_df %>% relocate(Construct)
  
  print("\nLatent correlations among SMP, SMS, SPI:")
  print(latent_corr_df)
  
  write_csv(
    latent_corr_df,
    file.path(DATA_DIR, glue("sem_full_latent_correlations_SMP_SMS_SPI_{MODEL}.csv"))
  )
}

# 10.5 Indicator R² and factor loadings (measurement layer)

loadings_mat <- as.data.frame(sum_pls$loadings)
loadings_mat$Indicator <- rownames(sum_pls$loadings)

loadings_long <- loadings_mat %>%
  pivot_longer(
    cols = -Indicator,
    names_to = "Construct",
    values_to = "Loading"
  ) %>%
  filter(!is.na(Loading))

loadings_long <- loadings_long %>%
  mutate(Indicator_R2 = Loading^2)

print("\nFirst few indicator loadings and R²:")
print(head(loadings_long))


# ---- Create clean loading table (only true non-zero loadings) ----
clean_loadings <- loadings_long %>%
  filter(Loading != 0) %>%
  arrange(Construct, Indicator)

print("\nClean indicator loadings (non-zero):")
print(clean_loadings)

write_csv(
  clean_loadings,
  file.path(DATA_DIR, glue("sem_full_indicator_loadings_R2_clean_{MODEL}.csv"))
)

# 10.7 Mediation summary (SPI -> SMS -> SMP) using standardized paths

if (!is.null(paths_std)) {
  get_std_path <- function(iv, dv) {
    row <- paths_std %>% dplyr::filter(IV == iv, DV == dv)
    if (nrow(row) == 1) row$Std_B[1] else NA_real_
  }
  
  a_std      <- get_std_path("SPI", "SMS")
  b_std      <- get_std_path("SMS", "SMP")
  cprime_std <- get_std_path("SPI", "SMP")
  
  if (!any(is.na(c(a_std, b_std, cprime_std)))) {
    indirect_ab_std <- a_std * b_std
    total_std       <- cprime_std + indirect_ab_std
    
    med_df <- data.frame(
      IV              = "SPI",
      Mediator        = "SMS",
      DV              = "SMP",
      a_std           = a_std,
      b_std           = b_std,
      cprime_std      = cprime_std,
      Indirect_ab_std = indirect_ab_std,
      Total_std       = total_std
    )
    
    print("\nStandardized mediation summary (SPI -> SMS -> SMP):")
    print(med_df)
    
    write_csv(
      med_df,
      file.path(DATA_DIR, glue("sem_full_mediation_SPI_SMS_SMP_{MODEL}.csv"))
    )
  }
}

print("\nAll requested metrics have been computed and exported.")



