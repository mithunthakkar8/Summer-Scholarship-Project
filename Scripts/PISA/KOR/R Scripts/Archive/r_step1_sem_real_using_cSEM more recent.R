
rm(list = ls())

# ================================================================
# r_step1_sem_real_csem.R
# PLS(-like) SEM version of your lavaan model using cSEM
# ================================================================

library(cSEM)
library(dplyr)
library(readr)
library(tidyr)
library(psych)
library(glue)
options(future.globals.maxSize = 2 * 1024^3)   # 2 GB
library(future)
plan(multisession)   # parallel bootstrap


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
  paste0("PV", 1:10, "MATH"),
  "ST004D01T","ESCS","ST001D01T","MISCED","SCHSIZE"
) %in% names(df)))


# ------------------------------------------------
# 2. DEFINE MODEL (single lavaan-like string)
#    - *no* labels (a, b, c1, etc.) and *no* := effects
#      We'll get indirect/total via `assess()`.
# ------------------------------------------------

model_csem <- "
# Measurement model (reflective)
SMP =~ PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH +
       PV6MATH + PV7MATH + PV8MATH + PV9MATH + PV10MATH

SMS =~ ST268Q01JA + ST268Q04JA + ST268Q07JA   # or ST268Q7JA depending on your data

SPI =~ SC064Q01TA + SC064Q02TA + SC064Q03TA +
       SC064Q04NA + SC064Q05WA + SC064Q06WA


# Structural model (as in the paper)
SMS ~ SPI
SMP ~ SMS + SPI
"


# ------------------------------------------------
# 3. FIT PLS-SEM + BOOTSTRAP
# ------------------------------------------------

sem_vars <- c(
  "ST268Q01JA","ST268Q04JA","ST268Q07JA",
  "SC064Q01TA","SC064Q02TA","SC064Q03TA",
  "SC064Q04NA","SC064Q05WA","SC064Q06WA",
  paste0("PV", 1:10, "MATH")
)

df_clean <- df[df_complete <- complete.cases(df[, sem_vars]), ]


fit <- csem(
  .data  = df_clean,
  .model = model_csem,
  .resample_method = "bootstrap",
  .R = 1000,
  .approach_weights = "PLS-PM",
  .handle_inadmissibles = "replace"
)



# Quick overview
summary(fit)


# ================================================================
# 4. LATENT (COMPOSITE) SCORES
# ================================================================

# 1. Grab construct scores from the fit object
latent_scores <- fit$Estimates$Construct_scores
# or equivalently:
# latent_scores <- getConstructScores(fit)$Estimates

# 2. Combine with df_clean (same number of rows)
df_latent <- cbind(df_clean, latent_scores)

dim(df_clean)      # should be 4290 x 24
dim(latent_scores) # should be 4290 x 8
dim(df_latent)     # should be 4290 x 32


# ================================================================
# 5. STRUCTURAL RESIDUALS (SMS & SMP)
# ================================================================

summ <- cSEM::summarize(fit)


paths <- summ$Estimates$Path_estimates

# helper: coefficient lookup
coef_of <- function(dv, iv) {
  out <- paths$Estimate[paths$DV == dv & paths$IV == iv]
  if(length(out) == 0) return(0)   # if path missing, return 0
  out
}


# PLS constructs are centered → intercept = 0
int_SMS <- 0

pred_SMS <- coef_of("SMS","SPI") * df_latent$SPI


df_latent$resid_SMS <- df_latent$SMS - pred_SMS

# Predicted SMP
int_SMP <- 0

pred_SMP <- 
  coef_of("SMP","SPI") * df_latent$SPI +
  coef_of("SMP","SMS") * df_latent$SMS

df_latent$resid_SMP <- df_latent$SMP - pred_SMP

# ================================================================
# 6. RELIABILITY, AVE, FORNELL–LARCKER, R²
# ================================================================
ass <- assess(fit)

reliability <- ass$Reliability
ave         <- ass$AVE
fl_matrix   <- ass$Fornell_Larcker
rsq         <- ass$R2

# ================================================================
# 7. STRUCTURAL PATHS TABLE
# ================================================================
paths_df <- summ$Estimates$Path_estimates %>%
  separate(Name, into = c("DV", "IV"), sep = " ~ ") %>% 
  rename(
    B_unstd = Estimate  # this exists
    # cSEM does NOT output std estimates unless requested;
    # your object does not contain Std_estimate
  )

# ================================================================
# 8. RENAME COLUMNS FOR GReaT
# ================================================================
shortmap <- read.csv(file.path(DATA_DIR, "pisa_shortname_mapping.csv"))
short_name_map <- setNames(shortmap$full, shortmap$old)

df_with_latents <- df_latent

# Rename observed variables using mapping
intersecting <- intersect(names(df_with_latents), names(short_name_map))
names(df_with_latents)[match(intersecting, names(df_with_latents))] <- short_name_map[intersecting]

# Rename latent composites to descriptive names
latent_map <- c(
  "SMP" = "Latent Factor: Student Math Performance (SMP)",
  "SMS" = "Latent Factor: Student Math self-efficacy (SMS)",
  "SPI" = "Latent Factor: School-level Parental Involvement (SPI)"
)

latent_intersect <- intersect(names(df_with_latents), names(latent_map))
names(df_with_latents)[match(latent_intersect, names(df_with_latents))] <- latent_map[latent_intersect]

write_csv(df_with_latents,
          file.path(DATA_DIR, glue("df_core_fullnames_with_latents_{CNT}.csv")))

reliability <- summ$Estimates$Reliabilities
reliability_df <- data.frame(
  construct = names(reliability),
  reliability = as.numeric(reliability)
)

indirect_df <- summ$Effect_estimates$Indirect_effect
indirect_df <- as.data.frame(indirect_df)


total_df    <- summ$Effect_estimates$Total_effect
total_df <- as.data.frame(total_df)

loadings_df <- summ$Estimates$Loading_estimates

htmt <- assess(fit)$HTMT
htmt <- lapply(htmt, function(x) { x[is.nan(x)] <- NA; x })

# Convert HTMT list into a tidy data frame
htmt_df <- do.call(rbind, lapply(names(htmt), function(name) {
  x <- htmt[[name]]
  
  # convert to data frame only if non-empty
  if(length(x) > 0) {
    df <- as.data.frame(x)
    df$Construct <- name
    return(df)
  } else {
    return(NULL)
  }
}))




# ================================================================
# 9. SAVE OUTPUTS
# ================================================================

write_csv(htmt_df, file.path(DATA_DIR, glue("sem_real_htmt_{CNT}.csv")))

write_csv(loadings_df, file.path(DATA_DIR, glue("sem_real_loadings_{CNT}.csv")))


write_csv(indirect_df, file.path(DATA_DIR, glue("sem_real_indirect_effects_{CNT}.csv")))
write_csv(total_df,    file.path(DATA_DIR, glue("sem_real_total_effects_{CNT}.csv")))

write_csv(paths_df, file.path(DATA_DIR, glue("sem_real_structural_paths_{CNT}.csv")))
write_csv(reliability_df,
          file.path(DATA_DIR, glue("sem_real_reliability_{CNT}.csv")))

write_csv(as.data.frame(ave), file.path(DATA_DIR, glue("sem_real_ave_{CNT}.csv")))
write_csv(as.data.frame(fl_matrix), file.path(DATA_DIR, glue("sem_real_fornell_larcker_{CNT}.csv")))
write_csv(as.data.frame(rsq), file.path(DATA_DIR, glue("sem_real_rsquare_raw_{CNT}.csv")))

cat("\n✔ PLS-SEM REAL complete using cSEM.\n")
cat("✔ Generated df_core_fullnames_with_latents (GReaT-ready)\n")

