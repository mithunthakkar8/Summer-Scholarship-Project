# r_step1_sem_real.R

library(lavaan)
library(psych)
library(dplyr)
library(readr)
library(tidyr)

DATA_DIR <- "/nesi/project/vuw04485/pisa_sem_pipeline/data/"

#DATA_DIR = "C:/Users/mithu/Documents/MEGA/VUW/Summer Research Project/Datasets/PISA 2022/"

# ============================
# 1. LOAD CORE DATA
# ============================
df <- read.csv(paste0(DATA_DIR, "df_core.csv"))

# ============================
# 2. LOAD VARIABLE → FULL NAME MAPPING
# ============================
mapping <- read.csv(file.path(DATA_DIR, "pisa_variable_mapping.csv"))

name_map <- setNames(mapping$full_name, mapping$code)

# We will rename columns LATER, after SEM is done.

# ============================
# 3. SEM MODEL
# ============================
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

fit <- sem(model, data = df, std.lv = TRUE, fixed.x = FALSE, missing = "fiml")

# ============================
# 4. LATENT SCORES
# ============================
latent_scores <- lavPredict(fit, type = "lv")   # SMP, SMS, SPI

df_latent <- cbind(df, as.data.frame(latent_scores))

# ============================
# 5. RENAME COLUMNS → FULL TEXT
# ============================

# FULL NAME MAPPING
mapping <- read.csv(file.path(DATA_DIR, "pisa_variable_mapping.csv"))

name_map <- setNames(mapping$full_name, mapping$code)

get_label <- function(var) {
  if (var %in% names(name_map)) {
    return(name_map[[var]])
  } else {
    return(var)
  }
}

derived_map <- c(
  "female"       = paste0("Gender (derived from ", get_label("ST004D01T"), "; 1 = Female, 0 = Male)"),
  "ESCS_z"       = paste0(get_label("ESCS"), " (standardized z-score)"),
  "ST001D01T_z"  = paste0(get_label("ST001D01T"), " (standardized z-score)"),
  "SCHSIZE_z"    = paste0(get_label("SCHSIZE"), " (standardized z-score)")
)

latent_map <- c(
  "SMP" = "Latent Factor: Math Performance (SMP)",
  "SMS" = "Latent Factor: Socio-Math Self-Concept (SMS)",
  "SPI" = "Latent Factor: School Climate (SPI)"
)

# combine all renaming maps
full_name_map <- c(name_map, derived_map, latent_map)


# Apply mapping AFTER adding latent scores
df_with_latents <- df_latent
intersecting <- intersect(names(df_with_latents), names(full_name_map))
names(df_with_latents)[match(intersecting, names(df_with_latents))] <- full_name_map[intersecting]

write_csv(df_with_latents, file.path(DATA_DIR, "df_core_fullnames_with_latents.csv"))

# ============================
# 6. SAVE ORIGINAL OUTPUTS
# ============================

fm <- fitMeasures(fit)
fit_df <- data.frame(
  metric = names(fm),
  value  = as.numeric(fm)
)

pe <- parameterEstimates(fit, standardized = TRUE)
cor_lv <- lavInspect(fit, "cor.lv")
rsq <- lavInspect(fit, "rsquare")

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

write_csv(pe, file.path(DATA_DIR, "sem_real_parameterEstimates.csv"))
write_csv(fit_df, file.path(DATA_DIR, "sem_real_fitMeasures.csv"))

write_csv(as.data.frame(cor_lv), file.path(DATA_DIR, "sem_real_latentCorrelations.csv"))
write_csv(as.data.frame(rsq),    file.path(DATA_DIR, "sem_real_rsquare_raw.csv"))
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
cat("✔ Generated df_core_fullnames_with_latents.csv (GReaT Ready)\n")
