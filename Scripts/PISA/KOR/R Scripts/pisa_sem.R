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

library(openxlsx)

# ================================================================
# LOGGING & DIAGNOSTICS
# ================================================================

LOG_LEVEL <- "DEBUG"  
# Allowed: "ERROR", "WARN", "INFO", "DEBUG"

.log_levels <- c(ERROR = 1, WARN = 2, INFO = 3, DEBUG = 4)

log_msg <- function(level, ...) {
  if (.log_levels[[level]] <= .log_levels[[LOG_LEVEL]]) {
    ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
    cat(sprintf("[%s] [%s] %s\n", ts, level, paste(..., collapse = " ")))
  }
}

log_info  <- function(...) log_msg("INFO",  ...)
log_warn  <- function(...) log_msg("WARN",  ...)
log_error <- function(...) log_msg("ERROR", ...)
log_debug <- function(...) log_msg("DEBUG", ...)

# Structured snapshot helper
log_snapshot <- function(name, x, max_items = 10) {
  log_debug(name, "=>",
            paste(utils::head(capture.output(str(x)), max_items), collapse = " | "))
}


# ------------------------------------------------
# Central results registry (DRY)
# ------------------------------------------------
results <- list()

add_result <- function(name, df) {
  if (!is.null(df) && nrow(df) > 0) {
    results[[name]] <<- as.data.frame(df)
  }
}


CNT <- "SGP"

DATA_SOURCES <- c(
  # "real",
  # "synthetic_ctgan",
  # "synthetic_great_distilgpt2",
  # "synthetic_great_gpt2",
  "synthetic_tabula_distilgpt2"
  # "synthetic_tabula_gpt2",
  # "synthetic_taptap_distilgpt2",
  # "synthetic_taptap_gpt2",
  # "synthetic_predllm_distilgpt2",
  # "synthetic_predllm_gpt2",
  # "synthetic_tabsyn",
  # "synthetic_tabdiff",
  # "synthetic_realtabformer"
)


BASE_DIR <- "C:/Users/mithu/Documents/MEGA/VUW/Summer Research Project"

DATASETS_DIR   <- file.path(BASE_DIR, "Datasets", "PISA 2022")
EXPERIMENTS_DIR <- file.path(BASE_DIR, "Experiments", "PISA-SEM", CNT)

DATA_SOURCE_PATHS <- list(
  # -------------------------
  # REAL
  # -------------------------
  real = DATASETS_DIR,
  
  # -------------------------
  # CTGAN
  # -------------------------
  synthetic_ctgan = file.path(
    EXPERIMENTS_DIR, "CTGAN"
  ),
  
  # -------------------------
  # GReaT
  # -------------------------
  synthetic_great_distilgpt2 = file.path(
    EXPERIMENTS_DIR, "GReaT", "DistilGPT2"
  ),
  synthetic_great_gpt2 = file.path(
    EXPERIMENTS_DIR, "GReaT", "GPT2"
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
  # TabSyn
  # -------------------------
  synthetic_tabsyn = file.path(
    EXPERIMENTS_DIR, "TabSyn"
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


for (DATA_SOURCE in DATA_SOURCES) {
  
  log_info("===================================================")
  log_info("Starting DATA SOURCE:", DATA_SOURCE)
  log_info("===================================================")
  
  DATA_DIR <- DATA_SOURCE_PATHS[[DATA_SOURCE]]
  stopifnot(!is.null(DATA_DIR))
  
  if (!dir.exists(DATA_DIR)) {
    log_warn("Directory does not exist, skipping:", DATA_DIR)
    next
  }
  
  if (DATA_SOURCE == "real") {
    csv_files <- file.path(DATA_DIR, glue("df_core{CNT}.csv"))
  } else {
    csv_files <- list.files(
      DATA_DIR,
      pattern = "\\.csv$",
      full.names = TRUE
    )
    
    csv_files <- csv_files[!grepl("log", basename(csv_files), ignore.case = TRUE)]
  }
  
  if (length(csv_files) == 0) {
    log_warn("No CSV files found for:", DATA_SOURCE)
    next
  }
  
  log_info("Found", length(csv_files), "files for", DATA_SOURCE)
  
  for (csv_path in csv_files) {
    results <- list()
    message("\n========================================")
    message("Running SEM on:", DATA_SOURCE, "->", basename(csv_path))
    message("========================================\n")


    
    df <- read.csv(csv_path)
    OUTPUT_DIR <- file.path(dirname(csv_path), "sem_outputs")
    if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR)
    RUN_ID <- tools::file_path_sans_ext(basename(csv_path))
    
    log_info("Loaded data:", nrow(df), "rows x", ncol(df), "cols")
    log_debug("Column names:", paste(colnames(df), collapse = ", "))
    
    
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
    log_debug("Detected latent-only columns:", 
              ifelse(length(latent_cols) == 0, "<none>", paste(latent_cols, collapse = ", ")))
    
    
    # ------------------------------------------------
    # 1A. IMPUTE MISSING VALUES (BEFORE ANY OTHER PROCESSING)
    #   - continuous: median
    #   - categorical: mode
    #   - identifiers: do NOT impute
    # ------------------------------------------------
    
    
    
    categorical_cols <- c(
      # Identifiers (do NOT impute)
      "CNT", "CNTSCHID", "CNTSTUID",
      
      # Student background (categorical / ordinal)
      "ST001D01T",     # grade
      "ST004D01T",     # gender
      "MISCED",        # mother's education
      "IMMIG",         # immigration status
      
      # SMS (Likert-type)
      "ST268Q01JA", "ST268Q04JA", "ST268Q07JA",
      
      # SPI items (ordinal)
      "SC064Q01TA", "SC064Q02TA", "SC064Q03TA",
      "SC064Q04NA", "SC064Q05WA", "SC064Q06WA", "SC064Q07WA"
    )
    
    continuous_cols <- c("AGE", "ESCS", "MCLSIZE", "SCHSIZE", paste0("PV", 1:10, "MATH"))
    
    id_cols <- c("CNT", "CNTSCHID", "CNTSTUID")
    
    # Helper: mode (most frequent non-NA)
    get_mode <- function(x) {
      x_non_na <- x[!is.na(x)]
      if (length(x_non_na) == 0) return(NA)
      tab <- table(x_non_na)
      names(tab)[which.max(tab)]
    }
    
    # Only impute columns that exist in this df
    cont_present <- intersect(continuous_cols, names(df))
    cat_present  <- intersect(setdiff(categorical_cols, id_cols), names(df))
    
    log_info("Continuous columns present:", paste(cont_present, collapse = ", "))
    log_info("Categorical columns present:", paste(cat_present, collapse = ", "))
    
    # Coerce continuous to numeric (safe for median)
    for (col in cont_present) {
      na_before <- sum(is.na(df[[col]]))
      
      if (!is.numeric(df[[col]])) {
        df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
      }
      if (anyNA(df[[col]])) {
        med <- median(df[[col]], na.rm = TRUE)
        # If a column is entirely NA, median() becomes NA; leave as-is in that case
        if (!is.na(med)) df[[col]][is.na(df[[col]])] <- med
      }
      na_after <- sum(is.na(df[[col]]))
      log_debug("Imputed numeric:", col, "| NA before:", na_before, "| NA after:", na_after)
      
    }
    
    
    
    
    # Categorical: mode imputation (treat as factor/character; keep original type where possible)
    for (col in cat_present) {
      
      
      if (anyNA(df[[col]])) {
        m <- get_mode(df[[col]])
        log_debug("Imputed categorical:", col, "| mode:", m)
        if (!is.na(m)) {
          df[[col]][is.na(df[[col]])] <- m
          # If factor, ensure the level exists
          if (is.factor(df[[col]]) && !(m %in% levels(df[[col]]))) {
            levels(df[[col]]) <- c(levels(df[[col]]), m)
          }
        }
      }
    }
    
    log_info("Total remaining NA count:", sum(is.na(df)))
    
    
    
    
    # Optional: quick audit printout
    # cat("Imputation done. Remaining NA count:", sum(is.na(df)), "\n")
    
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
    
    non_numeric <- sem_numeric_cols[!sapply(df[sem_numeric_cols], is.numeric)]
    if (length(non_numeric) > 0) {
      log_warn("Non-numeric SEM columns after coercion:", paste(non_numeric, collapse = ", "))
    }
    
    for (col in sem_numeric_cols) {
      if (is.factor(df[[col]])) {
        df[[col]] <- as.numeric(as.character(df[[col]]))
      } else if (is.character(df[[col]])) {
        df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
      }
    }
    
    
    required_vars <- c(
      "ST268Q01JA","ST268Q04JA","ST268Q07JA",
      "SC064Q01TA","SC064Q02TA","SC064Q03TA",
      "SC064Q05WA","SC064Q06WA","SC064Q04NA","SC064Q07WA",
      paste0("PV", 1:10, "MATH")
    )
    
    missing_vars <- setdiff(required_vars, names(df))
    
    if (length(missing_vars) > 0) {
      log_error("Missing required SEM variables:", paste(missing_vars, collapse = ", "))
      stop("Aborting SEM run due to missing variables.")
    }
    
    log_info("All required SEM variables present.")
    
    
    # ------------------------------------------------
    # TABLE 4: COVARIATE CORRELATION MATRIX (PRE-SEM)
    # ------------------------------------------------
    log_info("Computing Table 4: Covariate correlation matrix")
    
    # ---- Construct composite scores (as in paper) ----
    df$SPI_comp <- rowMeans(df[, c(
      "SC064Q01TA", "SC064Q02TA", "SC064Q03TA",
      "SC064Q04NA", "SC064Q05WA", "SC064Q06WA", "SC064Q07WA"
    )], na.rm = TRUE)
    
    df$SMS_comp <- rowMeans(df[, c(
      "ST268Q01JA", "ST268Q04JA", "ST268Q07JA"
    )], na.rm = TRUE)
    
    df$SMP_comp <- rowMeans(df[, paste0("PV", 1:10, "MATH")], na.rm = TRUE)
    
    # ---- Assemble covariates for Table 4 ----
    vars <- df[, c(
      "SMP_comp",
      "SPI_comp",
      "SMS_comp",
      "ST004D01T",   # gender
      "ST001D01T",   # grade
      "AGE",
      "ESCS",
      "MISCED",     # motherEdu
      "IMMIG",
      "MCLSIZE",    # classSize
      "SCHSIZE"     # schoolSize
    )]
    
    # Rename columns to publication-friendly labels
    colnames(vars) <- c(
      "SMP",
      "SPI",
      "SMS",
      "gender",
      "grade",
      "age",
      "ESCS",
      "motherEdu",
      "immig",
      "classSize",
      "schoolSize"
    )
    
    # ---- Pearson correlation matrix (pairwise deletion) ----
    # ---- Pearson correlation matrix ----
    corr_mat <- cor(
      vars,
      use = "pairwise.complete.obs",
      method = "pearson"
    )
    
    # Round
    corr_mat <- round(corr_mat, 4)
    
    # ---- Mask upper triangle (keep diagonal + lower triangle) ----
    corr_mat[upper.tri(corr_mat)] <- NA
    
    # Convert to data frame for export
    corr_df <- as.data.frame(corr_mat)
    corr_df$Variable <- rownames(corr_df)
    corr_df <- corr_df %>% relocate(Variable)
    
    
    print("\nTable 4: Covariate correlation matrix")
    print(corr_df)
    
    # ---- Store for Excel export ----
    add_result(glue("covariate_corr_{CNT}"),
               corr_df)
    
    
    
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
    
    log_info("Estimating PLS-SEM model...")
    log_snapshot("Measurement model", mm)
    log_snapshot("Structural model", sm)
    
    pls_model <- estimate_pls(
      data = df,
      measurement_model = mm,
      structural_model = sm,
      inner_weights = path_weighting
    )
    
    print("Model estimated successfully!")
    
    
    log_info("PLS model estimated.")
    log_debug("Path coefficients (raw):")
    log_snapshot("path_coef", pls_model$path_coef)
    
    pls_model <- PLSc(pls_model)
    print("PLSc correction applied successfully!")
    
    
    if (any(is.na(pls_model$path_coef))) {
      log_warn("NA detected in PLSc path coefficients.")
    }
    
    
    # ------------------------------------------------
    # FIX: Guard against NA path coefficients (synthetic instability)
    # ------------------------------------------------
    if (any(is.na(pls_model$path_coef))) {
      warning("NA values detected in PLSc path coefficients – replacing with 0")
      pls_model$path_coef[is.na(pls_model$path_coef)] <- 0
    }
    
    sum_pls <- summary(pls_model)
    
    # ------------------------------------------------
    # DEFINE TRUE LATENT CONSTRUCTS (FOR MEASUREMENT DIAGNOSTICS ONLY)
    # ------------------------------------------------
    latent_constructs <- c("SPI", "SMS", "SMP")
    
    log_info("Latent constructs used for measurement validity:",
             paste(latent_constructs, collapse = ", "))
    
    
    
    # ------------------------------------------------
    # 5. BOOTSTRAP FOR SIGNIFICANCE
    # ------------------------------------------------
    
    
    print("Starting bootstrap with 1000 iterations...")
    
    log_info("Starting bootstrap:", "nboot=1000", "cores=4", "seed=123")
    
    boot_pls <- bootstrap_model(
      seminr_model = pls_model,
      nboot = 1000,
      cores = 4,  
      seed = 123
    )
    
    print("Bootstrap completed!")
    boot_sum <- summary(boot_pls)
    
    log_info("Bootstrap completed successfully.")
    log_debug("Bootstrapped paths rows:", nrow(boot_sum$bootstrapped_paths))
    
    
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
    }, error = function(e) {
      log_warn("Indirect effect failed:", e$message)
      NULL
    }
    )
    
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
    # 10.3.1 Reliability table (latent constructs only)
    
    if (!is.null(sum_pls$reliability)) {
      
      reliability_all <- as.data.frame(sum_pls$reliability)
      reliability_all$Construct <- rownames(sum_pls$reliability)
      
      reliability_latent <- reliability_all %>%
        filter(Construct %in% latent_constructs) %>%
        relocate(Construct)
      
      print("\nReliability metrics (LATENT CONSTRUCTS ONLY):")
      print(reliability_latent)
      
      add_result(glue("pls_sem_reliability_{CNT}"),
                 reliability_latent)
    }
    
    
    # 10.3.2 Fornell-Larcker (LATENT CONSTRUCTS ONLY)
    
    if (!is.null(sum_pls$validity$fl_criteria)) {
      
      fl_mat <- sum_pls$validity$fl_criteria
      
      fl_latent <- fl_mat[latent_constructs, latent_constructs, drop = FALSE]
      
      fl_df <- as.data.frame(fl_latent)
      fl_df$Construct <- rownames(fl_latent)
      fl_df <- fl_df %>% relocate(Construct)
      
      print("\nFornell-Larcker (LATENT CONSTRUCTS ONLY):")
      print(fl_df)
      
      add_result(glue("pls_sem_fornell_larcker_{CNT}"),
                 fl_df)
    }
    
    
    # 10.3.3 HTMT (LATENT CONSTRUCTS ONLY)
    
    if (!is.null(sum_pls$validity$htmt)) {
      
      htmt_mat <- sum_pls$validity$htmt
      
      htmt_latent <- htmt_mat[latent_constructs, latent_constructs, drop = FALSE]
      
      htmt_df <- as.data.frame(htmt_latent)
      htmt_df$Construct <- rownames(htmt_latent)
      htmt_df <- htmt_df %>% relocate(Construct)
      
      print("\nHTMT (LATENT CONSTRUCTS ONLY):")
      print(htmt_df)
      
      add_result(glue("pls_sem_htmt_{CNT}"),
                 htmt_df)
    }
    
    log_info("Measurement validity evaluated only for:",
             paste(latent_constructs, collapse = ", "),
             "| controls excluded from validity metrics")
    
    
    # ------------------------------------------------
    # LOAD VARIABLE NAME MAPPING
    # ------------------------------------------------
    if (DATA_SOURCE == "real") {
      
      mapping <- read.csv(
        file.path(DATA_DIR, "pisa_variable_mapping.csv"),
        stringsAsFactors = FALSE
      )
      
      # ---- VALIDATION ----
      required_cols <- c("code", "canonical_name")
      missing_cols  <- setdiff(required_cols, names(mapping))
      
      if (length(missing_cols) > 0) {
        stop(
          "Variable mapping file is missing required columns: ",
          paste(missing_cols, collapse = ", ")
        )
      }
      
      if (nrow(mapping) == 0) {
        stop("Variable mapping file is empty.")
      }
      
      log_debug("Variable mapping columns:", paste(names(mapping), collapse = ", "))
      log_debug("First rows of mapping:")
      log_snapshot("mapping_head", head(mapping))
      log_debug(
        "canonical_name NULL?",
        is.null(mapping$canonical_name),
        "| code NULL?",
        is.null(mapping$code)
      ) 
      
      name_map <- setNames(mapping$canonical_name, mapping$code)
      
      
    } else {
      
      name_map <- NULL
    }
    
    
    
    latent_scores <- as.data.frame(sum_pls$composite_scores)
    
    
    # ================================================================
    # ====================== LAVAAN BLOCK =============================
    # ================================================================
    
    # # ------------------------------------------------
    # # LAVAAN MODEL (CB-SEM)
    # # ------------------------------------------------
    # lavaan_model <- '
    # SMP =~ PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH +
    #        PV6MATH + PV7MATH + PV8MATH + PV9MATH + PV10MATH
    # SMS =~ ST268Q01JA + ST268Q04JA + ST268Q07JA
    # SPI =~ SC064Q01TA + SC064Q02TA + SC064Q03TA + SC064Q04NA +
    #        SC064Q05WA + SC064Q06WA + SC064Q07WA
    # 
    # SMS ~ a*SPI + ST004D01T + ST001D01T + MISCED + ESCS + AGE + IMMIG + MCLSIZE + SCHSIZE
    # SMP ~ b*SMS + cprime*SPI + ST004D01T + ST001D01T + MISCED + ESCS + AGE + IMMIG + MCLSIZE + SCHSIZE
    # 
    # '
    # 
    # fit_cb <- sem(
    #   lavaan_model,
    #   data     = df,
    #   std.lv   = TRUE,
    #   fixed.x  = FALSE,
    #   missing  = "fiml"
    # )
    # 
    # 
    # if (!inspect(fit_cb, "converged")) {
    #   log_warn("Lavaan model did NOT converge.")
    # } else {
    #   log_info("Lavaan model converged successfully.")
    # }
    # 
    # log_debug("Lavaan fit measures snapshot:")
    # log_snapshot("fitMeasures", fitMeasures(fit_cb))
    # 
    # 
    # 
    # # ------------------------------------------------
    # # LAVAAN FIT & RELIABILITY
    # # ------------------------------------------------
    # fit_measures <- fitMeasures(fit_cb)
    # 
    # add_result(glue("sem_cb_fit_measures_{CNT}"),
    #            data.frame(metric = names(fit_measures), value = fit_measures)
    # )
    # 
    # rsq_cb <- lavInspect(fit_cb, "rsquare")
    # 
    # add_result(glue("sem_cb_rsquare_{CNT}"), 
    #            data.frame(Latent = names(rsq_cb), R2 = rsq_cb)
    # )
    # 
    # cor_lv_cb <- lavInspect(fit_cb, "cor.lv")
    # cor_lv_df <- as.data.frame(cor_lv_cb)
    # cor_lv_df$Latent <- rownames(cor_lv_df)
    # 
    # add_result(glue("sem_cb_correlations_{CNT}"), 
    #            cor_lv_df
    # )
    # 
    # 
    
    # ------------------------------------------------
    # EXPORT FULL DATA + SELECTED LATENT SCORES (RENAMED USING canonical_name)
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
    
    
    
    # Remove all columns with suffix "_comp"
    combined_smp <- combined_smp %>%
      dplyr::select(-dplyr::ends_with("_comp"))
    
    
    if (DATA_SOURCE == "real") {
      # 7. Write final file
      output_file_smp <- file.path(
        DATA_DIR,
        glue("df_core_with_smp_latent_{CNT}.csv")
      )
      write_csv(combined_smp, output_file_smp)
      
      
      
      
      # 5. Rename columns using name_map (canonical_name)
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
      
      add_result(glue("pls_sem_correlations_{CNT}"),
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
    
    
    log_info("Writing Excel workbook:", output_xlsx)
    log_debug("Result sheets:", paste(names(results), collapse = ", "))
    
    
    
    wb <- createWorkbook()
    
    for (sheet in names(results)) {
      addWorksheet(wb, sheet)
      writeData(wb, sheet, results[[sheet]])
    }
    
    saveWorkbook(wb, output_xlsx, overwrite = TRUE)
    
    log_info("Excel export completed.")
    cat("\n✔ All SEM results written to:\n", output_xlsx, "\n")
    
    
    print("\nAll requested metrics have been computed and exported.")
  } 
}

log_info("R SEM script completed successfully.")
# quit(status = 0)
