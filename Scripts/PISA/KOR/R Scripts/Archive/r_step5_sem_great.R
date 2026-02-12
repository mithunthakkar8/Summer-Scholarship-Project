rm(list = ls())

library(lavaan)
library(psych)
library(dplyr)
library(readr)
library(tidyr)

# Adjust path as needed
DATA_DIR <- "/nesi/project/vuw04485/pisa_sem_pipeline/data/"
#DATA_DIR <- "C:/Users/mithu/Documents/MEGA/VUW/Summer Research Project/Datasets/PISA 2022"

# -----------------------------
# 1. Load mapping file
# -----------------------------
mapping <- read_csv(file.path(DATA_DIR, "pisa_shortname_mapping.csv"))

# Python-style shortname (already correct, but we recreate it to be safe)
make_safe_shortname <- function(col) {
  s <- tolower(col)
  s <- gsub("[^a-z0-9]", "_", s)
  s <- gsub("_+", "_", s)
  s <- gsub("^_+|_+$", "", s)
  if (nchar(s) == 0) s <- "col"
  s
}

# R-style column name (mimic what read.csv does, but normalized)
make_r_safe <- function(x) {
  x <- gsub("[^A-Za-z0-9.]", ".", x)
  x <- gsub("[.]+", ".", x)    # collapse multiple dots
  x
}

mapping <- mapping %>%
  mutate(
    safe_short = sapply(full, make_safe_shortname),
    rsafe_full = make_r_safe(full)
  )

# (optional) save updated mapping
write_csv(mapping, file.path(DATA_DIR, "pisa_shortname_mapping_with_safe_short.csv"))

# -----------------------------
# 2. Load Real + Synthetic
# -----------------------------
real  <- read.csv(file.path(DATA_DIR, "df_core_fullnames_with_latents.csv"))
synth <- read.csv(file.path(DATA_DIR, "synthetic_sem_conditioned_on_latents_shortnames.csv"))

# Normalize REAL column names with the SAME rule used for rsafe_full
colnames(real) <- make_r_safe(colnames(real))

# -----------------------------
# 3. Build rename vectors
# -----------------------------

# REAL: rsafe_full (current name) -> old (target canonical name)
real_rename <- mapping %>%
  filter(rsafe_full %in% colnames(real))

real_rename_vector <- setNames(real_rename$rsafe_full, real_rename$old)
# This creates a vector like:
#   c(PV1MATH = "Plausible.Value.1.in.Mathematics", ...)

# SYNTH: safe_short (current name) -> old (target canonical name)
synth_rename <- mapping %>%
  filter(safe_short %in% colnames(synth))

synth_rename_vector <- setNames(synth_rename$safe_short, synth_rename$old)
#   c(PV1MATH = "plausible_value_1_in_mathematics", ...)

# -----------------------------
# 4. Apply renaming (ONCE)
# -----------------------------
real  <- dplyr::rename(real,  !!!real_rename_vector)
synth <- dplyr::rename(synth, !!!synth_rename_vector)

# Quick sanity checks
intersect(colnames(real), mapping$old)
intersect(colnames(synth), mapping$old)


# ---------------------------------------------------------------
# 5. SEM Model (uses OLD canonical names)
# ---------------------------------------------------------------
model <- '
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

# ---------------------------------------------------------------
# 6. Fit SEM using OLD name system
# ---------------------------------------------------------------
fit_real  <- sem(model, real, std.lv = TRUE, fixed.x = FALSE, missing = "fiml")
fit_synth <- sem(model, synth, std.lv = TRUE, fixed.x = FALSE, missing = "fiml")

# ---------------------------------------------------------------
# 3. FIT INDICES — Combined
# ---------------------------------------------------------------
fm_real  <- fitMeasures(fit_real)
fm_synth <- fitMeasures(fit_synth)

fit_compare <- data.frame(
  metric = names(fm_real),
  real   = as.numeric(fm_real),
  synth  = as.numeric(fm_synth)
)

write_csv(fit_compare,
          file.path(DATA_DIR, "SEM_compare_fit_indices_combined.csv"))

# ---------------------------------------------------------------
# 4. LOADINGS — Combined
# ---------------------------------------------------------------
pe_real  <- parameterEstimates(fit_real, standardized = TRUE)
pe_synth <- parameterEstimates(fit_synth, standardized = TRUE)

load_real  <- pe_real  %>% filter(op == "=~") %>% select(lhs, rhs, std.all)
load_synth <- pe_synth %>% filter(op == "=~") %>% select(lhs, rhs, std.all)

colnames(load_real)  <- c("latent", "indicator", "loading_real")
colnames(load_synth) <- c("latent", "indicator", "loading_synth")

load_combined <- merge(load_real, load_synth, by = c("latent","indicator"))
write_csv(load_combined,
          file.path(DATA_DIR, "SEM_compare_loadings_combined.csv"))

# ---------------------------------------------------------------
# 5. TUCKER’S CONGRUENCE — Single combined table
# ---------------------------------------------------------------
tucker_df <- load_combined %>%
  group_by(latent) %>%
  summarise(
    tucker_congruence =
      sum(loading_real * loading_synth) /
      sqrt(sum(loading_real^2) * sum(loading_synth^2))
  )

write_csv(tucker_df,
          file.path(DATA_DIR, "SEM_tucker_congruence.csv"))

# ---------------------------------------------------------------
# 6. LATENT CORRELATIONS — Combined long format
# ---------------------------------------------------------------
cor_real  <- lavInspect(fit_real,  "cor.lv")
cor_synth <- lavInspect(fit_synth, "cor.lv")

lat_names <- rownames(cor_real)

cor_real_df <- as.data.frame(as.table(cor_real))
cor_synth_df <- as.data.frame(as.table(cor_synth))

colnames(cor_real_df)  <- c("latent1", "latent2", "corr_real")
colnames(cor_synth_df) <- c("latent1", "latent2", "corr_synth")

cor_combined <- merge(cor_real_df, cor_synth_df,
                      by = c("latent1", "latent2"))

write_csv(cor_combined,
          file.path(DATA_DIR, "SEM_latent_correlations_combined.csv"))

# ---------------------------------------------------------------
# 7. RELIABILITY — Combined
# ---------------------------------------------------------------
rel_fun <- function(pe, df) {
  loads <- pe %>% filter(op == "=~") %>%
    select(latent = lhs, indicator = rhs, loading = std.all)
  
  do.call(rbind, lapply(unique(loads$latent), function(lat) {
    sub <- loads %>% filter(latent == lat)
    lam <- sub$loading
    inds <- sub$indicator
    
    CR  <- (sum(lam)^2) / ((sum(lam)^2) + sum(1 - lam^2))
    AVE <- sum(lam^2) / length(lam)
    alpha <- tryCatch(psych::alpha(df[, inds])$total$raw_alpha,
                      error = function(e) NA_real_)
    
    data.frame(
      Construct = lat,
      Cronbach = alpha,
      CR = CR,
      AVE = AVE
    )
  }))
}

rel_real  <- rel_fun(pe_real,  real)
rel_synth <- rel_fun(pe_synth, synth)

rel_combined <- merge(rel_real, rel_synth,
                      by = "Construct",
                      suffixes = c("_real","_synth"))

write_csv(rel_combined,
          file.path(DATA_DIR, "SEM_reliability_combined.csv"))


# ---------------------------------------------------------------
# 8. STRUCTURAL PATHS — Combined
# ---------------------------------------------------------------
paths_real <- pe_real %>%
  filter(op == "~") %>%
  select(lhs, rhs, est, std.all) %>%
  rename(B_real = est, Beta_real = std.all)

paths_synth <- pe_synth %>%
  filter(op == "~") %>%
  select(lhs, rhs, est, std.all) %>%
  rename(B_synth = est, Beta_synth = std.all)

paths_combined <- merge(paths_real, paths_synth, by = c("lhs","rhs"))

write_csv(paths_combined,
          file.path(DATA_DIR, "SEM_structural_paths_combined.csv"))

# ---------------------------------------------------------------
# 9. R-SQUARED — Combined
# ---------------------------------------------------------------
rsq_real  <- lavInspect(fit_real, "rsquare")
rsq_synth <- lavInspect(fit_synth, "rsquare")

rsq_df <- data.frame(
  latent = names(rsq_real),
  R2_real  = as.numeric(rsq_real),
  R2_synth = as.numeric(rsq_synth)
)

write_csv(rsq_df,
          file.path(DATA_DIR, "SEM_R2_combined.csv"))



# ---------------------------------------------------------------
# 10. FORNELL–LARCKER — Combined
# ---------------------------------------------------------------
fornell_fun <- function(rel_df, cor_mat) {
  sqrtAVE <- sqrt(rel_df$AVE)
  names(sqrtAVE) <- rel_df$Construct
  latent <- rel_df$Construct
  M <- cor_mat[latent, latent]
  diag(M) <- sqrtAVE
  return(M)
}

fl_real  <- fornell_fun(rel_real,  cor_real)
fl_synth <- fornell_fun(rel_synth, cor_synth)

fl_real_df  <- as.data.frame(as.table(fl_real))
fl_synth_df <- as.data.frame(as.table(fl_synth))

colnames(fl_real_df)  <- c("Construct1","Construct2","FL_real")
colnames(fl_synth_df) <- c("Construct1","Construct2","FL_synth")

fl_combined <- merge(fl_real_df, fl_synth_df,
                     by = c("Construct1","Construct2"))

write_csv(fl_combined,
          file.path(DATA_DIR, "SEM_fornell_larcker_combined.csv"))


cat("\n✔ ALL COMBINED SEM COMPARISON FILES GENERATED.\n")


