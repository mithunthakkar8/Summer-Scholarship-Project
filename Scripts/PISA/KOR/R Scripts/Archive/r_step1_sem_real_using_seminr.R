rm(list = ls())

# ================================================================
# r_step1_sem_real_seminr.R
# PLS(-like) SEM version of your lavaan model using SEMinR
# ================================================================

library(seminr)
library(dplyr)
library(readr)
library(tidyr)
library(psych)
library(glue)

DATA_DIR <- "C:/Users/mithu/Documents/MEGA/VUW/Summer Research Project/Datasets/PISA 2022"
#CNT <- "KOR"
CNT <- ""

# ------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------
df <- read.csv(file.path(DATA_DIR, glue("df_core{CNT}.csv")))

# (Optional sanity check)
stopifnot(all(c(
  "ST268Q01JA","ST268Q04JA","ST268Q07JA",
  "SC064Q01TA","SC064Q02TA","SC064Q03TA",
  "SC064Q04NA","SC064Q05WA","SC064Q06WA",
  paste0("PV", 1:10, "MATH")
) %in% names(df)))

# Keep only complete cases on SEM indicators (like in your cSEM code)
sem_vars <- c(
  "ST268Q01JA","ST268Q04JA","ST268Q07JA",
  "SC064Q01TA","SC064Q02TA","SC064Q03TA",
  "SC064Q04NA","SC064Q05WA","SC064Q06WA",
  paste0("PV", 1:10, "MATH")
)
df_clean <- df[complete.cases(df[, sem_vars]), ]


# ------------------------------------------------
# 2. DEFINE SEMinR MEASUREMENT & STRUCTURAL MODELS
# ------------------------------------------------

# Measurement model (reflective composites)
smp_items <- paste0("PV", 1:10, "MATH")

mm <- constructs(
  composite("SMP", smp_items),  # Student Math Performance
  composite("SMS", c("ST268Q01JA","ST268Q04JA","ST268Q07JA")),  # Self-efficacy
  composite("SPI", c("SC064Q01TA","SC064Q02TA","SC064Q03TA",
                     "SC064Q04NA","SC064Q05WA","SC064Q06WA"))   # Parental involvement
)

# Structural model: SMS ~ SPI ; SMP ~ SMS + SPI
sm <- relationships(
  paths(from = "SPI", to = c("SMS", "SMP")),
  paths(from = "SMS", to = "SMP")
)

# ------------------------------------------------
# 3. ESTIMATE PLS-SEM + (OPTIONAL) PLSc ADJUSTMENT
# ------------------------------------------------

pls_model <- estimate_pls(
  data             = df_clean,
  measurement_model = mm,
  structural_model  = sm,
  inner_weights     = path_weighting   # path-weighting scheme, like SmartPLS default
)

# If you want Consistent PLS (PLSc) to approximate SmartPLS "PLSc" results,
# uncomment the following line:
pls_model <- PLSc(pls_model)

sum_pls <- summary(pls_model)

# ------------------------------------------------
# 4. BOOTSTRAP FOR SIGNIFICANCE (1000 resamples)
# ------------------------------------------------

boot_pls <- bootstrap_model(
  seminr_model = pls_model,
  nboot        = 1000,
  cores        = NULL,   # or parallel::detectCores()
  seed         = 123
)

boot_sum <- summary(boot_pls)

# ================================================================
# STANDARDIZED PATH COEFFICIENTS (SmartPLS-style)
# ================================================================

# ================================================================
# MANUAL STANDARDIZATION OF PATH COEFFICIENTS (SmartPLS-style)
# ================================================================

# 1. Latent variable scores from SEMinR
latent_df <- as.data.frame(sum_pls$composite_scores)

# Compute SDs of latent variables
lv_sds <- apply(latent_df, 2, sd)

# Extract unstandardized bootstrap table
boot_unstd <- as.data.frame(boot_sum$bootstrapped_paths)
boot_unstd$Name <- rownames(boot_unstd)

# Parse DV and IV
paths_std <- boot_unstd %>%
  separate(Name, into = c("IV", "DV"), sep = "->") %>%
  mutate(
    IV = trimws(IV),
    DV = trimws(DV),
    
    # Standardize coefficient
    Std_B = `Original Est.` * (lv_sds[IV] / lv_sds[DV]),
    
    # Standardize SE (approximate)
    Std_SE = `Bootstrap SD` * (lv_sds[IV] / lv_sds[DV]),
    
    Std_t = Std_B / Std_SE,
    Std_p = 2 * (1 - pnorm(abs(Std_t)))
  ) %>%
  select(DV, IV, Std_B, Std_SE, Std_t, Std_p)


# ================================================================
# 5. LATENT (COMPOSITE) SCORES
# ================================================================

latent_scores <- sum_pls$composite_scores   # matrix/data.frame: rows = obs, cols = constructs
df_latent <- cbind(df_clean, latent_scores)

# Quick dimension checks (similar to your cSEM script)
cat("dim(df_clean):      ", dim(df_clean), "\n")
cat("dim(latent_scores): ", dim(latent_scores), "\n")
cat("dim(df_latent):     ", dim(df_latent), "\n")


# ================================================================
# 6. STRUCTURAL RESIDUALS (SMS & SMP)
# ================================================================

# sum_pls$paths is a matrix:
#   rows: R^2, AdjR^2, then IV constructs
#   cols: DV constructs
# e.g. sum_pls$paths["SPI","SMS"] gives beta(SMS ~ SPI)

coef_of <- function(dv, iv) {
  sum_pls$paths[iv, dv]
}

# PLS constructs are standardized; intercept ~ 0
int_SMS <- 0
pred_SMS <- coef_of("SMS", "SPI") * df_latent$SPI
df_latent$resid_SMS <- df_latent$SMS - pred_SMS

int_SMP <- 0
pred_SMP <-
  coef_of("SMP", "SPI") * df_latent$SPI +
  coef_of("SMP", "SMS") * df_latent$SMS

df_latent$resid_SMP <- df_latent$SMP - pred_SMP


# ================================================================
# 7. RELIABILITY, AVE, FORNELL–LARCKER, R²
# ================================================================

# Reliability table: alpha, rhoC, AVE, rhoA
reliab_full <- sum_pls$reliability

# Simple 2-column version (like your cSEM reliability_df)
# Using composite reliability (rhoC), which is the standard in PLS-SEM
reliability_df <- data.frame(
  construct  = rownames(reliab_full),
  reliability = reliab_full[, "rhoC"]
)

# AVE as its own data frame
ave_df <- data.frame(
  construct = rownames(reliab_full),
  AVE       = reliab_full[, "AVE"]
)

# Fornell–Larcker criteria
fl_matrix <- sum_pls$validity$fl_criteria

# R² from paths table
rsq_df <- data.frame(
  construct = colnames(sum_pls$paths),
  R2        = as.numeric(sum_pls$paths["R^2", ]),
  AdjR2     = as.numeric(sum_pls$paths["AdjR^2", ])
)


# ================================================================
# 8. STRUCTURAL PATHS TABLE (WITH BOOTSTRAP)
# ================================================================

boot_paths <- as.data.frame(boot_sum$bootstrapped_paths)
boot_paths$Name <- rownames(boot_sum$bootstrapped_paths)

paths_df <- boot_paths %>%
  separate(Name, into = c("DV", "IV"), sep = "->") %>%
  mutate(
    DV = trimws(DV),
    IV = trimws(IV),
    B_unstd = `Original Est.`,
    Std_err = `Bootstrap SD`,
    t_stat  = `T Stat.`,
    p_value = 2 * (1 - pnorm(abs(`T Stat.`))),   # normal approx
    CI_percentile.95L = `2.5% CI`,
    CI_percentile.95U = `97.5% CI`
  ) %>%
  select(DV, IV, B_unstd, Std_err, t_stat, p_value,
         CI_percentile.95L, CI_percentile.95U)


# ================================================================
# 9. LOADINGS, HTMT
# ================================================================

# Loadings (point estimates)
loadings_df <- as.data.frame(sum_pls$loadings)
loadings_df$item <- rownames(sum_pls$loadings)
# rearrange to long format if you like later

# HTMT matrix
htmt_mat <- sum_pls$validity$htmt

# Tidy HTMT into long format (similar spirit to your cSEM htmt_df)
htmt_df <- htmt_mat %>%
  as.data.frame() %>%
  mutate(Construct1 = rownames(htmt_mat)) %>%
  pivot_longer(
    cols      = -Construct1,
    names_to  = "Construct2",
    values_to = "HTMT"
  ) %>%
  filter(!is.na(HTMT), Construct1 != Construct2)


# ================================================================
# 10. RENAME COLUMNS FOR GReaT (FULL NAMES + LATENTS)
# ================================================================

shortmap <- read.csv(file.path(DATA_DIR, "pisa_shortname_mapping.csv"))
short_name_map <- setNames(shortmap$full, shortmap$old)

df_with_latents <- df_latent

# Rename observed variables using mapping
intersecting <- intersect(names(df_with_latents), names(short_name_map))
names(df_with_latents)[match(intersecting, names(df_with_latents))] <-
  short_name_map[intersecting]

# Rename latent composites to descriptive names
latent_map <- c(
  "SMP" = "Latent Factor: Student Math Performance (SMP)",
  "SMS" = "Latent Factor: Student Math self-efficacy (SMS)",
  "SPI" = "Latent Factor: School-level Parental Involvement (SPI)"
)

latent_intersect <- intersect(names(df_with_latents), names(latent_map))
names(df_with_latents)[match(latent_intersect, names(df_with_latents))] <-
  latent_map[latent_intersect]

write_csv(
  df_with_latents,
  file.path(DATA_DIR, glue("df_core_fullnames_with_latents_{CNT}.csv"))
)


# ================================================================
# 11. SAVE OUTPUTS (MIRRORING YOUR cSEM VERSION)
# ================================================================

# HTMT
write_csv(htmt_df,
          file.path(DATA_DIR, glue("sem_real_htmt_{CNT}.csv")))

# Loadings
write_csv(loadings_df,
          file.path(DATA_DIR, glue("sem_real_loadings_{CNT}.csv")))

# Indirect & total effects (from SEMinR summary)
# - indirect effects via SMS: SPI -> SMS -> SMP
# - total effects from summary object
total_effects_df <- as.data.frame(sum_pls$total_effects)
total_indirect_df <- as.data.frame(sum_pls$total_indirect_effects)

write_csv(total_indirect_df,
          file.path(DATA_DIR, glue("sem_real_indirect_effects_{CNT}.csv")))
write_csv(total_effects_df,
          file.path(DATA_DIR, glue("sem_real_total_effects_{CNT}.csv")))

# Structural paths
write_csv(paths_df,
          file.path(DATA_DIR, glue("sem_real_structural_paths_{CNT}.csv")))

#write_csv(std_paths_df,
        #  file.path(DATA_DIR, glue("sem_real_structural_paths_standardized_{CNT}.csv")))


# Reliability (rhoC only)
write_csv(reliability_df,
          file.path(DATA_DIR, glue("sem_real_reliability_{CNT}.csv")))

# AVE
write_csv(ave_df,
          file.path(DATA_DIR, glue("sem_real_ave_{CNT}.csv")))

# Fornell-Larcker
write_csv(as.data.frame(fl_matrix),
          file.path(DATA_DIR, glue("sem_real_fornell_larcker_{CNT}.csv")))

# R-squared
write_csv(rsq_df,
          file.path(DATA_DIR, glue("sem_real_rsquare_raw_{CNT}.csv")))

cat("\n✔ PLS-SEM REAL complete using SEMinR.\n")
cat("✔ Generated df_core_fullnames_with_latents (GReaT-ready)\n")
