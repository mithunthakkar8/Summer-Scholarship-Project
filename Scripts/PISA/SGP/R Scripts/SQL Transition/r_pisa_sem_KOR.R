rm(list = ls())

# ================================================================
# r_step1_sem_real_seminr.R
# PLS-SEM analysis using SEMinR for structural coherence
# ================================================================

library(DBI)
library(RPostgres)   # or RPostgreSQL
library(seminr)
library(dplyr)
library(readr)
library(tidyr)
library(psych)
library(glue)
library(lavaan)  # for CB-SEM fit indices and parameter tables

library(openxlsx)

# ------------------------------------------------
# Central results registry (DRY)
# ------------------------------------------------
results <- list()

add_result <- function(name, df) {
  if (!is.null(df) && nrow(df) > 0) {
    results[[name]] <<- as.data.frame(df)
  }
}


CNT <- "KOR"

# ------------------------------------------------
# DATA SOURCE SWITCH
# ------------------------------------------------
DATA_SOURCE <- "real"
# Allowed values:
# "real"
# "synthetic_great_distilgpt2"
# "synthetic_great_gpt2"
# "synthetic_tabula_distilgpt2"
# "synthetic_tabula_gpt2"
# "synthetic_taptap_distilgpt2"
# "synthetic_taptap_gpt2"
# "synthetic_predllm_distilgpt2"
# "synthetic_predllm_gpt2"
# "synthetic_tabdiff"
# "synthetic_realtabformer"

BASE_DIR <- "C:/Users/mithu/Documents/MEGA/VUW/Summer Research Project"

DATASETS_DIR   <- file.path(BASE_DIR, "Datasets", "PISA 2022")
EXPERIMENTS_DIR <- file.path(BASE_DIR, "Experiments", "PISA-SEM", "SGP")

DATA_SOURCE_PATHS <- list(
  # -------------------------
  # REAL
  # -------------------------
  real = DATASETS_DIR,
  
  # -------------------------
  # GReaT
  # -------------------------
  synthetic_great_distilgpt2 = file.path(
    EXPERIMENTS_DIR, "GReaT", "Baseline", "DistilGPT2"
  ),
  synthetic_great_gpt2 = file.path(
    EXPERIMENTS_DIR, "GReaT", "Baseline", "GPT2"
  ),
  
  # -------------------------
  # Tabula
  # -------------------------
  synthetic_tabula_distilgpt2 = file.path(
    EXPERIMENTS_DIR, "Tabula", "DistilGPT2"
  ),
  synthetic_tabula_gpt2 = file.path(
    EXPERIMENTS_DIR, "Tabula", "GPT2"
  ),
  
  # -------------------------
  # TapTap
  # -------------------------
  synthetic_taptap_distilgpt2 = file.path(
    EXPERIMENTS_DIR, "TapTap", "DistilGPT2"
  ),
  synthetic_taptap_gpt2 = file.path(
    EXPERIMENTS_DIR, "TapTap", "GPT2"
  ),
  
  # -------------------------
  # PredLLM
  # -------------------------
  synthetic_predllm_distilgpt2 = file.path(
    EXPERIMENTS_DIR, "PredLLM", "DistilGPT2"
  ),
  synthetic_predllm_gpt2 = file.path(
    EXPERIMENTS_DIR, "PredLLM", "GPT2"
  ),
  
  # -------------------------
  # TabDiff
  # -------------------------
  synthetic_tabdiff = file.path(
    EXPERIMENTS_DIR, "TabDiff"
  ),
  
  # -------------------------
  # REaLTabFormer
  # -------------------------
  synthetic_realtabformer = file.path(
    EXPERIMENTS_DIR, "REaLTabFormer"
  )
)


DATA_DIR <- DATA_SOURCE_PATHS[[DATA_SOURCE]]
stopifnot(!is.null(DATA_DIR))


name_map <- if (DATA_SOURCE == "real") {
  mapping <- read.csv(file.path(DATA_DIR, "pisa_variable_mapping.csv"))
  setNames(mapping$safe_short, mapping$code)
} else NULL


# ------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------


load_real_data <- function(CNT) {
  con <- dbConnect(
    RPostgres::Postgres(),
    dbname   = "PISA_2022",
    host     = "localhost",
    port     = 5432,
    user     = "postgres",
    password = "postgres"
  )
  
  df <- dbGetQuery(
    con,
    glue("SELECT * FROM pisa.real_sem_{CNT}")
  )
  
  dbDisconnect(con)
  df
}

load_synthetic_files <- function(DATA_DIR) {
  list.files(
    DATA_DIR,
    pattern = "\\.csv$",
    full.names = TRUE
  )
}



run_sem_pipeline <- function(df, CNT, OUTPUT_DIR, RUN_ID, name_map = NULL) {

names(df) <- toupper(names(df))
  
# ------------------------------------------------
# SAFETY: Drop latent-only columns (TapTap hygiene)
# ------------------------------------------------
latent_prefixes <- c("^lv_", "^LV_", "^latent_", "^LATENT_")

latent_cols <- grep(
  paste(latent_prefixes, collapse = "|"),
  colnames(df),
  value = TRUE
)

if (length(latent_cols) > 0) {
  message(
    "Dropping latent-only columns (not used in SEM): ",
    paste(latent_cols, collapse = ", ")
  )
  df <- df %>% select(-all_of(latent_cols))
}


# ------------------------------------------------
# 1B. FORCE SEM INDICATORS TO NUMERIC (SEMinR REQUIREMENT)
# ------------------------------------------------

sem_numeric_cols <- unique(c(
  # Measurement indicators
  paste0("PV", 1:10, "MATH"),
  "ST268Q01JA","ST268Q04JA","ST268Q07JA",
  "SC064Q01TA","SC064Q02TA","SC064Q03TA",
  "SC064Q04NA","SC064Q05WA","SC064Q06WA","SC064Q07WA",
  
  # Controls used in structural paths
  "ST004D01T","ST001D01T","MISCED","IMMIG",
  "AGE","ESCS","MCLSIZE","SCHSIZE"
))

sem_numeric_cols <- intersect(sem_numeric_cols, names(df))

for (col in sem_numeric_cols) {
  if (is.factor(df[[col]])) {
    df[[col]] <- as.numeric(as.character(df[[col]]))
  } else if (is.character(df[[col]])) {
    df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
  }
}


# Sanity check for required variables
stopifnot(all(c(
  "ST268Q01JA","ST268Q04JA","ST268Q07JA",
  "SC064Q01TA","SC064Q02TA","SC064Q03TA",
  "SC064Q05WA","SC064Q06WA","SC064Q04NA", "SC064Q07WA",
  paste0("PV", 1:10, "MATH")
) %in% names(df)))

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
  data = df,
  measurement_model = mm,
  structural_model = sm,
  inner_weights = path_weighting
)

print("Model estimated successfully!")

pls_model <- PLSc(pls_model)
print("PLSc correction applied successfully!")

# ------------------------------------------------
# FIX: Guard against NA path coefficients (synthetic instability)
# ------------------------------------------------
if (any(is.na(pls_model$path_coef))) {
  warning("NA values detected in PLSc path coefficients – replacing with 0")
  pls_model$path_coef[is.na(pls_model$path_coef)] <- 0
}

sum_pls <- summary(pls_model)



# ------------------------------------------------
# 5. BOOTSTRAP FOR SIGNIFICANCE
# ------------------------------------------------
print("Starting bootstrap with 1000 iterations...")
boot_pls <- bootstrap_model(
  seminr_model = pls_model,
  nboot = 1000,
  cores = 4,  
  seed = 123
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
  
  
  add_result(glue("pls_sem_full_indirect_{CNT}"), 
             indirect_tbl
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
    
    add_result(glue("sem_full_std_paths_{CNT}"), 
               paths_std
    )
    
    # add_result(glue("sem_full_unstd_paths_{CNT}"),
    #   structural_paths
    # )
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
  
  add_result(glue("sem_full_rsquared_{CNT}"),
             r_squared
  )
}

# ------------------------------------------------
# 9. TOTAL EFFECTS
# ------------------------------------------------
print("\nExtracting total effects...")

if (!is.null(boot_sum$bootstrapped_total_paths)) {
  
  total_effects <- as.data.frame(boot_sum$bootstrapped_total_paths) %>%
    mutate(
      Path = rownames(.),
      IV = gsub("\\s*->.*", "", Path),
      DV = gsub(".*->\\s*", "", Path),
      IV = trimws(IV),
      DV = trimws(DV)
    ) %>%
    relocate(Path, DV, IV)
  
  print("Total effects:")
  print(total_effects)
  
  add_result(glue("sem_full_total_effects_{CNT}"), 
             total_effects
  )
}

# ------------------------------------------------
# 10. COMPREHENSIVE STRUCTURAL COHERENCE METRICS
# ------------------------------------------------
print("\nComputing comprehensive structural coherence metrics...")


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
  add_result(glue("pls_sem_reliability_{CNT}"),
             reliability_metrics
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
  
  add_result(glue("pls_sem_fornell_larcker_{CNT}"),
             fl_df
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
  
  add_result(glue("pls_sem_full_htmt_{CNT}"),
             htmt_df
  )
}

# ------------------------------------------------
# LOAD VARIABLE NAME MAPPING
# ------------------------------------------------



latent_scores <- as.data.frame(sum_pls$composite_scores)


# ================================================================
# ====================== LAVAAN BLOCK =============================
# ================================================================

# ------------------------------------------------
# LAVAAN MODEL (CB-SEM)
# ------------------------------------------------
lavaan_model <- '
  SMP =~ PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH +
         PV6MATH + PV7MATH + PV8MATH + PV9MATH + PV10MATH
  SMS =~ ST268Q01JA + ST268Q04JA + ST268Q07JA
  SPI =~ SC064Q01TA + SC064Q02TA + SC064Q03TA + SC064Q04NA +
         SC064Q05WA + SC064Q06WA + SC064Q07WA
  
  SMS ~ a*SPI + ST004D01T + ST001D01T + MISCED + ESCS + AGE + IMMIG + MCLSIZE + SCHSIZE
  SMP ~ b*SMS + cprime*SPI + ST004D01T + ST001D01T + MISCED + ESCS + AGE + IMMIG + MCLSIZE + SCHSIZE
  
  '

fit_cb <- sem(
  lavaan_model,
  data     = df,
  std.lv   = TRUE,
  fixed.x  = FALSE,
  missing  = "fiml"
)



# # ------------------------------------------------
# # LAVAAN LATENT SCORES
# # ------------------------------------------------
# latent_scores_cb <- lavPredict(fit_cb, type = "lv")
# latent_scores_cb <- as.data.frame(latent_scores_cb)
# 
# add_result(glue("sem_cb_latent_scores_{CNT}"),
#   latent_scores_cb
# )
# 
# # ------------------------------------------------
# # LAVAAN STRUCTURAL RESIDUALS
# # ------------------------------------------------
# pe <- parameterEstimates(fit_cb)
# 
# coef_of <- function(lhs, rhs) {
#   pe$est[pe$lhs == lhs & pe$rhs == rhs & pe$op == "~"]
# }
# 
# int_SMS <- pe$est[pe$lhs == "SMS" & pe$op == "~1"]
# int_SMP <- pe$est[pe$lhs == "SMP" & pe$op == "~1"]
# 
# pred_SMS <- int_SMS +
#   coef_of("SMS","SPI") * latent_scores_cb$SPI
# 
# pred_SMP <- int_SMP +
#   coef_of("SMP","SMS") * latent_scores_cb$SMS +
#   coef_of("SMP","SPI") * latent_scores_cb$SPI
# 
# resid_df <- data.frame(
#   resid_SMS = latent_scores_cb$SMS - pred_SMS,
#   resid_SMP = latent_scores_cb$SMP - pred_SMP
# )
# 
# add_result(glue("sem_cb_structural_residuals_{CNT}"), 
#   resid_df
# )

# ------------------------------------------------
# LAVAAN FIT & RELIABILITY
# ------------------------------------------------
fit_measures <- fitMeasures(fit_cb)

add_result(glue("sem_cb_fit_measures_{CNT}"),
           data.frame(metric = names(fit_measures), value = fit_measures)
)

rsq_cb <- lavInspect(fit_cb, "rsquare")

add_result(glue("sem_cb_rsquare_{CNT}"), 
           data.frame(Latent = names(rsq_cb), R2 = rsq_cb)
)

cor_lv_cb <- lavInspect(fit_cb, "cor.lv")
cor_lv_df <- as.data.frame(cor_lv_cb)
cor_lv_df$Latent <- rownames(cor_lv_df)

add_result(glue("sem_cb_latent_correlations_{CNT}"), 
           cor_lv_df
)



# ------------------------------------------------
# EXPORT FULL DATA + SELECTED LATENT SCORES (RENAMED USING SAFE_SHORT)
# ------------------------------------------------

latent_scores_export <- latent_scores[, c("SMP", "SMS", "SPI")]
latent_score_smp_export <- latent_scores[, c("SMP"), drop = FALSE]

if (DATA_SOURCE == "real") {
  colnames(latent_scores_export) <- c("LV_SMP", "LV_SMS", "LV_SPI")
  colnames(latent_score_smp_export) <- c("LV_SMP")
}

# 4. Combine original df + selected latent scores
combined_full <- cbind(df, latent_scores_export)

# 4. Combine original df + selected latent scores
combined_smp <- cbind(df, latent_score_smp_export)

# 7. Write final file
if (DATA_SOURCE == "real") {
  output_file_all_latents <- file.path(
    DATA_DIR,
    glue("df_core_with_latent_scores_{CNT}.csv")
  )
  write_csv(combined_full, output_file_all_latents)
}




if (DATA_SOURCE == "real") {
  # 7. Write final file
  output_file_smp <- file.path(
    DATA_DIR,
    glue("df_core_with_smp_latent_{CNT}.csv")
  )
  write_csv(combined_smp, output_file_smp)
  
  
  
  
  # 5. Rename columns using name_map (safe_short)
  new_names <- ifelse(
    colnames(combined_full) %in% names(name_map),
    name_map[colnames(combined_full)],
    colnames(combined_full)
  )
  colnames(combined_full) <- new_names
  
  # 6. Now rename the three latent score columns to human-readable labels
  colnames(combined_full)[colnames(combined_full) == "LV_SMP"] <- 
    "latent_score_sem_student_math_performance"
  
  colnames(combined_full)[colnames(combined_full) == "LV_SPI"] <- 
    "latent_score_sem_school_based_parental_involvement"
  
  colnames(combined_full)[colnames(combined_full) == "LV_SMS"] <- 
    "latent_score_sem_student_math_self_efficacy"
  
  # 7. Write final file
  output_file <- file.path(
    DATA_DIR,
    glue("sem_full_dataset_raw_plus_selected_latent_scores_{CNT}.csv")
  )
}
# ------------------------------------------------
# REPLACE COUNTRY CODES WITH FULL COUNTRY NAMES
# ------------------------------------------------

# combined_full$country_code_3_character <- recode(
#   combined_full$country_code_3_character,
#   "JPN" = "Japan",
#   "KOR" = "Korea",
#   "SGP" = "Singapore",
#   "TAP" = "Chinese Taipei",
#   "HKG" = "Hong Kong (China)",
#   "MAC" = "Macao (China)"
# )


# Drop identifiers
combined_full <- combined_full %>%
  select(-any_of(c("intl_school_id", "intl_student_id")))

if (DATA_SOURCE == "real") {
  
  write_csv(combined_full, output_file)
  print(paste("Exported dataset saved to:", output_file))
  
}






# 10.4.3 Latent correlation matrix between SMP, SMS, SPI
if (all(c("SMP","SMS","SPI") %in% colnames(latent_scores))) {
  latent_corr <- cor(latent_scores[, c("SMP","SMS","SPI")],
                     use = "pairwise.complete.obs")
  latent_corr_df <- as.data.frame(latent_corr)
  latent_corr_df$Construct <- rownames(latent_corr)
  latent_corr_df <- latent_corr_df %>% relocate(Construct)
  
  print("\nLatent correlations among SMP, SMS, SPI:")
  print(latent_corr_df)
  
  add_result(glue("pls_sem_latent_correlations_{CNT}"),
             latent_corr_df
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

add_result(glue("pls_sem_loadings_R2_{CNT}"), 
           clean_loadings
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
    
    add_result(glue("pls_sem_mediation_{CNT}"), 
               med_df
    )
  }
}

# ------------------------------------------------
# WRITE ALL RESULTS TO SINGLE EXCEL WORKBOOK
# ------------------------------------------------
output_xlsx <- file.path(
  OUTPUT_DIR,
  glue("sem_results_{RUN_ID}.xlsx")
)


wb <- createWorkbook()

for (sheet in names(results)) {
  addWorksheet(wb, sheet)
  writeData(wb, sheet, results[[sheet]])
}

saveWorkbook(wb, output_xlsx, overwrite = TRUE)


cat("\n✔ All SEM results written to:\n", output_xlsx, "\n")


print("\nAll requested metrics have been computed and exported.")

return(invisible(TRUE))
}

if (DATA_SOURCE == "real") {

  df <- load_real_data(CNT)

  run_sem_pipeline(
    df = df,
    CNT = CNT,
    OUTPUT_DIR = DATASETS_DIR,
    RUN_ID = glue("real_{CNT}")
  )

} else {

  csv_files <- load_synthetic_files(DATA_DIR)
  stopifnot(length(csv_files) > 0)

  for (csv_path in csv_files) {

    df <- read.csv(csv_path)

    run_sem_pipeline(
      df = df,
      CNT = CNT,
      OUTPUT_DIR = dirname(csv_path),
      RUN_ID = tools::file_path_sans_ext(basename(csv_path))
    )
  }
}

  
  


