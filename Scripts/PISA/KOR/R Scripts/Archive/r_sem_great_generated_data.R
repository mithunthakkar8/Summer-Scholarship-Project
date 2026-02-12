# r_step1_sem_real.R

library(lavaan)
library(psych)
library(dplyr)
library(readr)
library(tidyr)

#DATA_DIR <- "/nesi/project/vuw04485/pisa_sem_pipeline/data"
DATA_DIR <- "C:/Users/mithu/Documents/MEGA/VUW/Summer Research Project/Datasets/PISA 2022"

# ============================
# 1. LOAD CORE DATA (CODED NAMES)
# ============================
df <- read.csv(file.path(DATA_DIR, "synthetic_sem_full.csv"))

# ============================
# 2. SEM MODEL (same as before)
# ============================
model <- '
SMP =~ PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH +
       PV6MATH + PV7MATH + PV8MATH + PV9MATH + PV10MATH

SMS =~ ST268Q01JA + ST268Q04JA + ST268Q07JA

SPI =~ SC064Q01TA + SC064Q02TA + SC064Q03TA + SC064Q04NA +
       SC064Q05WA + SC064Q06WA

SMS ~ a*SPI + c1*ST004D01T + c2*ESCS + c3*ST001D01T + c4*MISCED + c5*SCHSIZE
SMP ~ b*SMS + cprime*SPI + d1*ST004D01T + d2*ESCS + d3*ST001D01T + d4*MISCED + d5*SCHSIZE

indirect := a*b
total := cprime + (a*b)
'

fit <- sem(model, data = df, std.lv = TRUE, fixed.x = FALSE, missing = "fiml")

# ============================
# 3. LATENT SCORES
# ============================
latent_scores <- lavPredict(fit, type = "lv")   # columns: SMP, SMS, SPI
df_latent <- cbind(df, as.data.frame(latent_scores))

# ============================
# 3B. COMPUTE STRUCTURAL RESIDUAL FOR SMS
# ============================

# 3B-1: Extract all regression coefficients for SMS
pe <- parameterEstimates(fit)

# Helper to extract a coefficient
coef_of <- function(lhs, rhs) {
  pe$est[pe$lhs == lhs & pe$rhs == rhs & pe$op == "~"]
}

# Structural coefficients for SMS equation
a  <- coef_of("SMS", "SPI")
c1 <- coef_of("SMS", "ST004D01T")
c2 <- coef_of("SMS", "ESCS")
c3 <- coef_of("SMS", "ST001D01T")
c4 <- coef_of("SMS", "MISCED")
c5 <- coef_of("SMS", "SCHSIZE")

# Intercept for SMS
int_SMS <- pe$est[pe$lhs == "SMS" & pe$op == "~1"]

pred_SMS <- int_SMS +
  a  * df_latent$SPI +
  c1 * df_latent$ST004D01T +
  c2 * df_latent$ESCS +
  c3 * df_latent$ST001D01T +
  c4 * df_latent$MISCED +
  c5 * df_latent$SCHSIZE

# 3B-3: Compute structural residual for SMS
resid_SMS <- df_latent$SMS - pred_SMS

# 3B-4: Add to df_latent
df_latent$resid_SMS <- resid_SMS

# ============================
# 3C. COMPUTE STRUCTURAL RESIDUAL FOR SMP
# ============================

# Extract regression coefficients for SMP
d1 <- coef_of("SMP", "ST004D01T")
d2 <- coef_of("SMP", "ESCS")
d3 <- coef_of("SMP", "ST001D01T")
d4 <- coef_of("SMP", "MISCED")
d5 <- coef_of("SMP", "SCHSIZE")
b  <- coef_of("SMP", "SMS")
cprime <- coef_of("SMP", "SPI")

# Intercept for SMP
int_SMP <- pe$est[pe$lhs == "SMP" & pe$op == "~1"]

pred_SMP <- int_SMP +
  b       * df_latent$SMS +
  cprime  * df_latent$SPI +
  d1      * df_latent$ST004D01T +
  d2      * df_latent$ESCS +
  d3      * df_latent$ST001D01T +
  d4      * df_latent$MISCED +
  d5      * df_latent$SCHSIZE

# Structural residual for SMP
resid_SMP <- df_latent$SMP - pred_SMP

# Add to dataset
df_latent$resid_SMP <- resid_SMP

# ============================
# 4. RENAME COLUMNS FOR GReaT
#    - use SHORT NAMES for all observed/derived vars
#    - LONG DESCRIPTIVE NAMES for latent factors
# ============================

# This mapping is written by python_step1_prepare_df_core.py
# with columns: old_name, short_name, full_label
shortmap <- read.csv(file.path(DATA_DIR, "pisa_shortname_mapping.csv"))

# old_name (e.g. PV1MATH) -> short_name (e.g. PV1MATH or ESCS_z)
short_name_map <- setNames(shortmap$full, shortmap$old)

df_with_latents <- df_latent

# Apply short names to any columns that appear in the mapping
intersecting <- intersect(names(df_with_latents), names(short_name_map))
names(df_with_latents)[match(intersecting, names(df_with_latents))] <- short_name_map[intersecting]

# Now rename latent score columns to long descriptive names
latent_map <- c(
  "SMP" = "Latent Factor: Student Math Performance (SMP)",
  "SMS" = "Latent Factor: Student Math self-efficacy (SMS)",
  "SPI" = "Latent Factor: School-level Parental Involvement (SPI)"
)

latent_intersect <- intersect(names(df_with_latents), names(latent_map))
names(df_with_latents)[match(latent_intersect, names(df_with_latents))] <- latent_map[latent_intersect]

# Final GReaT-ready file: short/compact names for all observed vars,
# and long descriptive names for the 3 latent factors.
write_csv(df_with_latents,
          file.path(DATA_DIR, "df_core_fullnames_with_latents.csv"))


# ============================
# 5. SAVE SEM OUTPUTS (unchanged)
# ============================

fm <- fitMeasures(fit)
fit_df <- data.frame(
  metric = names(fm),
  value  = as.numeric(fm)
)

pe   <- parameterEstimates(fit, standardized = TRUE)
cor_lv <- lavInspect(fit, "cor.lv")
rsq  <- lavInspect(fit, "rsquare")

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
      AVE = round(AVE, 3),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, out)
}

real_reliab <- reliability_table(pe, df)

fornell_larcker_table <- function(reliab_df, lat_cor_mat) {
  sqrt_ave <- sqrt(reliab_df$AVE)
  names(sqrt_ave) <- reliab_df$Construct
  latents <- reliab_df$Construct
  corr <- lat_cor_mat[latents, latents, drop = FALSE]
  for (l in latents) {
    corr[l, l] <- sqrt_ave[l]
  }
  corr
}

real_fl <- fornell_larcker_table(real_reliab, cor_lv)

structural_paths_table <- function(std_solution) {
  paths <- std_solution[std_solution$op == "~", ]
  paths %>%
    select(lhs, rhs, est, se, pvalue, std.all) %>%
    rename(
      DV    = lhs,
      IV    = rhs,
      B_unstd = est,
      B_std   = std.all
    )
}

real_paths <- structural_paths_table(pe)

rsq_df <- data.frame(
  latent = names(rsq),
  R2 = as.numeric(rsq)
)

cor_df <- as.data.frame(cor_lv)

cor_df$latent1 <- rownames(cor_lv)
cor_df <- cor_df[, c("latent1", colnames(cor_df)[1:3])]


write_csv(pe,        file.path(DATA_DIR, "sem_real_parameterEstimates.csv"))
write_csv(fit_df,    file.path(DATA_DIR, "sem_real_fitMeasures.csv"))
write_csv(cor_df, file.path(DATA_DIR, "sem_real_latentCorrelations.csv"))
write_csv(rsq_df,    file.path(DATA_DIR, "sem_real_rsquare_raw.csv"))
write_csv(real_reliab,           file.path(DATA_DIR, "sem_real_reliability.csv"))



real_fl_df <- as.data.frame(real_fl)
real_fl_df$Latent <- rownames(real_fl)
real_fl_df <- real_fl_df %>% relocate(Latent, .before = 1)
write_csv(real_fl_df, file.path(DATA_DIR, "sem_real_fornell_larcker.csv"))

write_csv(real_paths, file.path(DATA_DIR, "sem_real_structural_paths.csv"))

sel_metrics <- c("cfi","tli","rmsea","srmr","chisq","df")
summary_df <- fit_df %>% filter(metric %in% sel_metrics)
write_csv(summary_df, file.path(DATA_DIR, "sem_real_fit_summary.csv"))

cat("\n✔ SEM REAL complete.\n")
cat("✔ Generated df_core_fullnames_with_latents.csv (GReaT-ready: short names + latent labels)\n")
