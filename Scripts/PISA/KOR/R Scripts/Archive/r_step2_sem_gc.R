# r_step2_sem_gc.R (FINAL — indicator + latent R² separated correctly)

library(lavaan)
library(psych)
library(dplyr)
library(readr)
library(tidyr)

DATA_DIR <- "/nesi/project/vuw04485/pisa_sem_pipeline/data/"

df_real <- read.csv(file.path(DATA_DIR, "df_core.csv"))
df_gc   <- read.csv(file.path(DATA_DIR, "synthetic_gc.csv"))

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

reliability_table <- function(std_solution, df_for_alpha) {
  loads <- std_solution[std_solution$op == "=~", c("lhs","rhs","std.all")]
  colnames(loads) <- c("latent", "indicator", "loading")
  out <- lapply(unique(loads$latent), function(lat){
    sub <- loads[loads$latent == lat, ]
    inds <- sub$indicator
    lam  <- as.numeric(sub$loading)
    CR   <- (sum(lam)^2) / ((sum(lam)^2) + sum(1 - lam^2))
    AVE  <- sum(lam^2) / length(lam)
    alpha_val <- tryCatch({
      psych::alpha(df_for_alpha[, inds])$total$raw_alpha
    }, error = function(e) NA_real_)
    data.frame(
      Construct = lat,
      Cronbach_alpha = round(alpha_val, 3),
      Composite_Reliability = round(CR, 3),
      AVE = round(AVE, 3)
    )
  })
  do.call(rbind, out)
}

fornell_larcker_table <- function(reliab_df, lat_cor_mat) {
  sqrt_ave <- sqrt(reliab_df$AVE)
  names(sqrt_ave) <- reliab_df$Construct
  latents <- reliab_df$Construct
  corr <- lat_cor_mat[latents, latents, drop = FALSE]
  for (l in latents) corr[l, l] <- sqrt_ave[l]
  corr
}

structural_paths_table <- function(std_solution) {
  std_solution[std_solution$op == "~", ] %>%
    select(lhs, rhs, est, se, pvalue, std.all) %>%
    rename(
      DV      = lhs,
      IV      = rhs,
      B_unstd = est,
      B_std   = std.all
    )
}

tucker_congruence <- function(load_real, load_synth) {
  merged <- merge(
    load_real, load_synth,
    by = c("latent","indicator"),
    suffixes = c("_real","_synth")
  )
  out <- lapply(unique(merged$latent), function(lat){
    sub <- merged[merged$latent == lat, ]
    num <- sum(sub$loading_real * sub$loading_synth)
    den <- sqrt(sum(sub$loading_real^2) * sum(sub$loading_synth^2))
    data.frame(
      Construct = lat,
      Tucker_Congruence = ifelse(den > 0, round(num / den, 3), NA_real_)
    )
  })
  do.call(rbind, out)
}

load_tbl <- function(std_sol) {
  std_sol[std_sol$op == "=~", c("lhs","rhs","std.all")] %>%
    rename(
      latent    = lhs,
      indicator = rhs,
      loading   = std.all
    )
}




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

fit_real <- sem(model, data = df_real, std.lv = TRUE, fixed.x = FALSE, missing = "fiml")
fit_gc   <- sem(model, data = df_gc,   std.lv = TRUE, fixed.x = FALSE, missing = "fiml")

# ============================================================
#   Extract indicator-level R² + latent-level R² properly
# ============================================================

get_r2_tables <- function(fit) {
  r2 <- lavInspect(fit, "rsquare")
  
  # indicator names = those that appear in =~ lines
  indicator_names <- names(r2)[!names(r2) %in% c("SMP", "SMS")]
  
  # latent names = endogenous latent variables
  latent_names <- names(r2)[names(r2) %in% c("SMP", "SMS")]
  
  list(
    indicator_r2 = data.frame(
      Construct = indicator_names,
      R2        = as.numeric(r2[indicator_names])
    ),
    latent_r2 = data.frame(
      Construct = latent_names,
      R2        = as.numeric(r2[latent_names])
    )
  )
}

real_r2 <- get_r2_tables(fit_real)
gc_r2   <- get_r2_tables(fit_gc)

# --- Save both forms ---
write_csv(real_r2$indicator_r2, file.path(DATA_DIR, "sem_real_rsquare_indicator.csv"))
write_csv(gc_r2$indicator_r2,   file.path(DATA_DIR, "sem_gc_rsquare_indicator.csv"))

write_csv(real_r2$latent_r2, file.path(DATA_DIR, "sem_real_rsquare_latent.csv"))
write_csv(gc_r2$latent_r2,   file.path(DATA_DIR, "sem_gc_rsquare_latent.csv"))

# ============================================================
# Everything below remains unchanged
# ============================================================

pe_real <- parameterEstimates(fit_real, standardized = TRUE)
pe_gc   <- parameterEstimates(fit_gc,   standardized = TRUE)

cor_lv_real <- lavInspect(fit_real, "cor.lv")
cor_lv_gc   <- lavInspect(fit_gc,   "cor.lv")

fm_real <- fitMeasures(fit_real)
fm_gc   <- fitMeasures(fit_gc)

fit_real_df <- data.frame(metric = names(fm_real), real = as.numeric(fm_real))
fit_gc_df   <- data.frame(metric = names(fm_gc),   gc   = as.numeric(fm_gc))

fit_join <- full_join(fit_real_df, fit_gc_df, by = "metric") %>%
  mutate(delta = gc - real)

# Loadings
load_tbl <- function(std_sol) {
  std_sol[std_sol$op == "=~", c("lhs","rhs","std.all")] %>%
    rename(
      latent    = lhs,
      indicator = rhs,
      loading   = std.all
    )
}

real_loadings <- load_tbl(pe_real)
gc_loadings   <- load_tbl(pe_gc)

tucker_df <- tucker_congruence(real_loadings, gc_loadings)

# Reliability
real_reliab <- reliability_table(pe_real, df_real)
gc_reliab   <- reliability_table(pe_gc,   df_gc)

# Fornell–Larcker
real_fl <- fornell_larcker_table(real_reliab, cor_lv_real)
gc_fl   <- fornell_larcker_table(gc_reliab,   cor_lv_gc)

# Save outputs
write_csv(gc_reliab, file.path(DATA_DIR, "sem_gc_reliability.csv"))

gc_fl_df <- as.data.frame(gc_fl)
gc_fl_df$Latent <- rownames(gc_fl_df)
gc_fl_df <- gc_fl_df %>% relocate(Latent, .before = 1)
write_csv(gc_fl_df, file.path(DATA_DIR, "sem_gc_fornell_larcker.csv"))

gc_paths <- structural_paths_table(pe_gc)
write_csv(gc_paths, file.path(DATA_DIR, "sem_gc_structural_paths.csv"))

write_csv(tucker_df, file.path(DATA_DIR, "sem_gc_tucker_congruence.csv"))
write_csv(fit_join,  file.path(DATA_DIR, "sem_fit_comparison_real_vs_gc.csv"))

cat("\n✔ SEM GaussianCopula + comparison complete.\n")
