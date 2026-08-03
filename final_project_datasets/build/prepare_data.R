#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Build the student-facing CSVs for the final-project dataset repository.
#
# Each dataset is rebuilt FROM THE RAW SOURCE (not from datasets_staging/cleaned),
# because several of the staged CSVs carry decoding bugs:
#   * cimt_ra      : `sex` lost all women (0 = woman was dropped -> 34% "complete")
#   * pam13        : questionnaire items keep SPSS's 99 = missing sentinel
#   * pam13        : Disease_severity shifted from 1/2/3 to 0/1/2
#   * hiv_6mwt     : smoking left as 0/1/2, which reads as never/former/current
#                    but is actually never/CURRENT/FORMER (verified vs Table 1)
#
# Output: data/<slug>.csv  (one tidy CSV per dataset, snake_case, factors decoded)
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(readxl); library(haven); library(purrr)
})

# Run from the repository folder root (final_project_datasets_fable5_d3n8/).
stopifnot(dir.exists("build"))
RAW     <- "../datasets_staging/raw"
CLEANED <- "../datasets_staging/cleaned"
OUT     <- "data"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

lab <- function(x, from, to) factor(to[match(x, from)], levels = to)
zap99 <- function(x) { x[x %in% 99] <- NA; x }

written <- list()
put <- function(slug, df) {
  df <- as.data.frame(df)
  write_csv(df, file.path(OUT, paste0(slug, ".csv")), na = "")
  written[[slug]] <<- df
  cat(sprintf("  %-24s %5d x %3d\n", slug, nrow(df), ncol(df)))
}

cat("Building student CSVs\n")

# ---------------------------------------------------------------------------
# 1. hiv_lung_6mwt -- Frasca et al. PLOS ONE 2019 (Pittsburgh HIV Lung Cohort)
# ---------------------------------------------------------------------------
d <- read_excel(file.path(RAW, "11_hiv_6mwt/Data for HIV 6MW PLOS ONE_2.xlsx"),
                sheet = "Data", .name_repair = "minimal")
put("hiv_lung_6mwt", tibble(
  hiv_positive          = d$hiv_st,                                  # codebook: 1 = Pos
  age_yrs               = d$age,
  sex                   = lab(d$gender, c(1, 0), c("female", "male")),  # codebook: 1 = female
  smoking_status        = lab(d$smokingstatus, c(0, 1, 2),
                              c("never", "current", "former")),      # verified vs paper Table 1
  pack_years            = d$pack_years,
  systolic_bp           = d$pre_bp_sys,
  diastolic_bp          = d$pre_bp_dia,
  haemoglobin_g_dl      = d$hgb,
  six_min_walk_m        = d$dist_meters,
  mmrc_dyspnoea         = d$mmrc_score,
  sgrq_symptoms         = d$symptoms_score,
  sgrq_activity         = d$activity_score,
  sgrq_impacts          = d$impacts_score,
  sgrq_total            = d$sgrq_total_score,
  fvc_pct_predicted     = d$post_fvcppp,
  fev1_pct_predicted    = d$post_fev1ppp,
  fev1_fvc_ratio        = d$post_fev1fvcpp,
  dlco_pct_predicted    = d$dlcopp,
  cd4_count             = d$cd4,                                     # PLWH only
  on_antiretroviral     = d$a4hivmed_current,                        # PLWH only
  viral_load_detectable = d$vldet                                    # PLWH only
))

# ---------------------------------------------------------------------------
# 2. ggt_arterial_stiffness -- BMJ Open 2014 (Japanese health-check cohort)
# ---------------------------------------------------------------------------
d <- read_excel(file.path(RAW, "12_ggt_atherosclerosis/data bmjopen dryad.xls"),
                sheet = 1, .name_repair = "minimal")
g <- read_csv(file.path(CLEANED, "12_ggt_atherosclerosis.csv"), show_col_types = FALSE)
put("ggt_arterial_stiffness", tibble(
  id                = d$ID,
  sex               = g$sex,                # already decoded in staging, verified 592 m / 320 f
  age_yrs           = d$Age,
  bmi               = d$BMI,
  systolic_bp       = d$sBP,
  diastolic_bp      = d$dBP,
  smoking_status    = factor(case_when(
                         d$`current smoker` == 1 ~ "current",
                         d$exsmoker == 1 ~ "former",
                         TRUE ~ "never"
                       ), levels = c("never", "former", "current")),
  alcohol_g_per_week = d$Alcohol,
  regular_exercise  = d$exercise,
  ast               = d$AST,
  alt               = d$ALT,
  ggt               = d$`γGTP`,
  log2_ggt          = d$log2ggt,
  fasting_glucose   = d$FBS,
  uric_acid         = d$`Uric acid`,
  total_cholesterol = d$TC,
  triglycerides     = d$TG,
  hdl_cholesterol   = d$HDL,
  ldl_cholesterol   = d$LDL3,
  egfr              = d$eGFR,
  fatty_liver       = d$`fatty liver`,
  # 36 rows (IDs 491-526) carry a second PWV reading in the ABImax column of the
  # deposited file instead of an ankle-brachial index. Set to NA rather than
  # shipping values ~1000x too large.
  abi_max           = ifelse(d$ABImax >= 3, NA, d$ABImax),
  pwv_max           = d$PWVmax
))
# NB: "posr menopausal state" dropped -- defined only for the 320 women (35% complete)

# ---------------------------------------------------------------------------
# 3. cimt_rheumatoid -- Ozen et al. PLOS ONE 2015
#    Trimmed to the variables measured on ALL participants. The RA-specific
#    treatment/disease-activity columns are structurally missing for the 166
#    controls, so they are dropped rather than shipped as 65%-complete traps.
# ---------------------------------------------------------------------------
d <- read_sav(file.path(RAW, "13_cimt_ra/traditional CVRF in relation to cIMT_PLOSONE23aug15.sav"))
put("cimt_rheumatoid", tibble(
  id                 = paste0(ifelse(as.integer(d$RA) %in% c(0, 1), "ra_", "control_"),
                              ifelse(is.na(d$idPatient), paste0("row", seq_len(nrow(d))),
                                     as.integer(d$idPatient))),
  sex                = lab(as.integer(d$gender), c(1, 0), c("male", "female")),  # 0 = woman, 1 = man
  age_yrs            = d$age,
  has_ra             = as.integer(as.integer(d$RA) %in% c(0, 1)),
  # `RA`: 0 = RA no HT/HC, 1 = RA with HT/HC, 2 = control no HT/HC, 3 = control with HT/HC
  hypertension_or_hyperchol = as.integer(as.integer(d$RA) %in% c(1, 3)),
  # `length` mixes units in the deposited file: 161 rows are in metres (1.55-1.95),
  # 307 are in centimetres. Rescaled to cm; verified against the file's own BMI.
  height_cm          = ifelse(d$length < 3, d$length * 100, d$length),
  weight_kg          = d$weight,
  bmi                = d$BMI,
  waist_cm           = d$waist,
  systolic_bp        = d$systolicBP,
  diastolic_bp       = d$diastoligbp,
  current_smoker     = as.integer(d$Smoking),                 # 0 = nee, 1 = ja
  antihypertensives  = as.integer(d$Anti_Hypertensives),
  statins            = as.integer(d$Statines),
  prednisone         = as.integer(d$Prednison),
  glucose            = d$Glucose,
  total_cholesterol  = d$Cholesterol,
  hdl_cholesterol    = d$HDL_Chol,
  ldl_cholesterol    = d$LDL_berekend,
  triglycerides      = d$Triglyceriden,
  crp                = d$CRP,
  apo_a              = d$ApoA,
  apo_b              = d$ApoB,
  cimt_total         = d$CIMT_total,
  carotid_plaque     = as.integer(d$Plaques)
))

# ---------------------------------------------------------------------------
# 4. patient_activation -- Bos-Touwen et al. PLOS ONE 2015
#    Derived scale scores + patient characteristics only. The 60 item-level
#    columns are dropped (they use 99 as a missing sentinel and add nothing
#    for a regression project). Disease-specific severity markers (eGFR, NYHA,
#    GOLD) are dropped: each is defined for one disease group only.
# ---------------------------------------------------------------------------
d <- read_sav(file.path(RAW, "15_pam13/Datafile Patient activation for self-management.sav"))
num <- function(v) { v <- zap99(as.numeric(v)); v }
put("patient_activation", tibble(
  id                     = as.integer(d$Number),
  sex                    = lab(as.integer(d$Gender), c(1, 2), c("male", "female")),
  age_yrs                = num(d$Age),
  bmi                    = num(d$BMI),
  height_cm              = num(d$Length),
  weight_kg              = num(d$Bodyweight),
  education_level        = lab(as.integer(d$Education_level), 1:3, c("lower", "middle", "higher")),
  living_alone           = as.integer(as.integer(d$Living_situation) == 1),
  financial_distress     = lab(as.integer(d$Financial_distress), 1:3, c("none", "low", "high")),
  smoking_status         = lab(as.integer(d$Smoking), 1:3, c("never", "former", "current")),
  ethnicity_dutch        = as.integer(as.integer(d$Ethnicity) == 1),
  care_allowance         = as.integer(as.integer(d$Care_allowance) == 1),
  disease                = lab(as.integer(d$Disease), 1:4,
                               c("diabetes_t2", "copd", "heart_failure", "renal_disease")),
  # NB: Disease_severity is deliberately NOT shipped. The SPSS value labels say
  # 1 = mild / 2 = moderate / 3 = severe but the stored values are 0/1/2, so the
  # mapping is ambiguous and would be guesswork.
  disease_duration       = lab(as.integer(d$Disease_duration), 0:2,
                               c("under_2_yrs", "2_to_5_yrs", "over_5_yrs")),
  n_comorbidities        = num(d$Total_comorbidities),
  charlson_index         = num(d$Charlson),
  pam_score              = num(d$activation_score),
  pam_level              = as.integer(d$PAM_levels),
  sf12_physical          = num(d$SF_phys),
  sf12_mental            = num(d$SF_ment),
  hads_depression        = num(d$HADS_Depression),
  hads_anxiety           = num(d$HADS_Anxiety),
  ipq_total              = num(d$IPQ_Total_score),
  social_support_total   = num(d$SUPP_Total_score),
  support_family         = num(d$SUPP_Total_Family),
  support_friends        = num(d$SUPP_Total_Friends),
  support_significant_other = num(d$SUPP_Total_SignificantOther)
))

# ---------------------------------------------------------------------------
# 5. medstudent_qol -- Tempski et al. PLOS ONE 2015 (22 Brazilian medical schools)
# ---------------------------------------------------------------------------
d <- read_excel(file.path(RAW, "16_med_student_qol/Dataset Resilience Educational Environment QoL.xlsx"),
                sheet = 1, skip = 1, .name_repair = "minimal")
put("medstudent_qol", tibble(
  id                    = d$IDR,
  sex                   = factor(tolower(d$Sex), levels = c("female", "male")),
  year_group            = factor(gsub(" ", "_", tolower(d$Group)),
                                 levels = c("basic_sciences", "clinical_sciences", "clerkship")),
  age_yrs               = as.numeric(d$Age),
  school_legal_status   = factor(tolower(d$`School legal status`), levels = c("public", "private")),
  school_location       = factor(ifelse(d$`School location` == "State capital",
                                        "state_capital", "other_city"),
                                 levels = c("state_capital", "other_city")),
  overall_qol           = as.numeric(d$`Overall QoL`),
  medical_school_qol    = as.numeric(d$`Medical school-related QoL`),
  whoqol_physical       = as.numeric(d$`WHOQOL physical health`),
  whoqol_psychological  = as.numeric(d$`WHOQOL psychological`),
  whoqol_social         = as.numeric(d$`WHOQOL social relationships`),
  whoqol_environment    = as.numeric(d$`WHOQOL environment`),
  dreem_learning        = as.numeric(d$`DREEM learning`),
  dreem_teachers        = as.numeric(d$`DREEM teachers`),
  dreem_academic_self   = as.numeric(d$`DREEM academic self-perception`),
  dreem_atmosphere      = as.numeric(d$`DREEM atmosphere`),
  dreem_social_self     = as.numeric(d$`DREEM social self-perception`),
  dreem_global          = as.numeric(d$`DREEM global score`),
  resilience_score      = as.numeric(d$`Resilience score`),
  bdi_depression        = as.numeric(d$BDI),
  state_anxiety         = as.numeric(d$`State Anxiety`),
  trait_anxiety         = as.numeric(d$`Trait anxiety`)
))

# ---------------------------------------------------------------------------
# 8. tibial_bone_strength -- Denys et al. J Musculoskelet Neuronal Interact 2022
# ---------------------------------------------------------------------------
d <- read_csv(file.path(CLEANED, "06_peak_power_bone.csv"), show_col_types = FALSE)
put("tibial_bone_strength", tibble(
  id                     = d$subject_id,
  sex                    = factor(d$sex, levels = c("female", "male")),
  age_decade             = as.integer(d$age_bin),
  body_mass_kg           = d$body_mass_kg,
  peak_power_w           = d$peak_power_w,
  relative_peak_power    = d$relative_peak_power,
  bone_strength_index    = d$bsi_compression,
  polar_strength_strain_index = d$polar_strength_strain_index
))

cat("\nDone. ", length(written), " datasets written to ", OUT, "/\n", sep = "")

# ===========================================================================
# LMIC additions. Raw files live in this repository's own raw/ folder (they are
# not part of ../datasets_staging). Sources are recorded in build/metadata.R.
# ===========================================================================
RAW2 <- "../final_project_datasets_fable5_d3n8/raw"

# ---------------------------------------------------------------------------
# 11. nepal_hypertension -- Dhungana et al., PLOS ONE 2017 (WHO STEPS, Surkhet)
# ---------------------------------------------------------------------------
d <- read_sav(file.path(RAW2, "nepal_hypertension/1rb14.sav"))
# 88 = "answer not given" is declared in the value labels but never occurs
stopifnot(!any(d$Marrital == 88), !any(d$Smoking == 88), !any(d$Tobacco == 88))
put("nepal_hypertension", tibble(
  sex               = lab(as.integer(d$Sex), c(2, 1), c("female", "male")),
  age_group         = lab(as.integer(d$Agecat), 1:4,
                          c("30_39", "40_49", "50_59", "60_plus")),
  marital_status    = lab(as.integer(d$Marrital), 1:3,
                          c("unmarried", "married", "divorced_or_widowed")),
  education         = lab(as.integer(d$Edu4), 1:4,
                          c("none_or_informal", "primary", "secondary", "higher")),
  occupation        = lab(as.integer(d$Occupation4), 1:4,
                          c("job_or_self_employed", "household_work",
                            "agriculture_or_labour", "unemployed")),
  below_poverty_line = as.integer(d$Poor),
  smoking_status    = lab(as.integer(d$Smoking), c(3, 2, 1),
                          c("never", "former", "current")),
  cigarettes_per_week = as.numeric(d$DoseSmoking),
  smokeless_tobacco = as.integer(as.integer(d$Tobacco) == 1),
  alcohol_last_month = as.integer(as.integer(d$CurAlcohol) == 1),
  fruit_veg_servings = as.numeric(d$ServingsFV),
  salt_g_per_day    = as.numeric(d$SaltGm),
  mets_min_per_week = as.numeric(d$METs),
  physical_activity = lab(as.integer(d$PA), c(0, 1), c("low", "moderate_or_high")),
  family_history_htn = as.integer(as.integer(d$FamilyHxHT) == 1),
  bmi               = as.numeric(d$BMI),
  systolic_bp       = as.numeric(d$SBP),
  diastolic_bp      = as.numeric(d$DBP),
  hypertension      = as.integer(d$HTN),
  bp_awareness      = as.integer(d$BPAwareness),
  antihypertensive_drug = as.integer(d$AntiHTNdrug)
))
# `Standerddrks` dropped: asked only of the 177 current drinkers (85% missing)

# ---------------------------------------------------------------------------
# 12. kenya_hypertension -- PLOS ONE 2025, clients at Kenyan health facilities
#     The three repeat BP readings are dropped in favour of the averages.
# ---------------------------------------------------------------------------
d <- suppressWarnings(read_excel(
  file.path(RAW2, "kenya_hypertension/journal.pone.0334255.s001.xls"),
  sheet = 1, .name_repair = "minimal"))
yn01 <- function(v) as.integer(v == "Yes")
put("kenya_hypertension", tibble(
  sex               = factor(tolower(d$Gender), levels = c("female", "male")),
  age_group         = factor(d$`Age Categories`,
                             levels = c("18-25", "26-35", "36-45", "46-55", "56-65", ">65")),
  education         = factor(d$Education, levels = c("No School", "Primary", "Secondary", "Tertiary")),
  marital_status    = factor(d$`Marital Status`),
  employment        = factor(d$Employment),
  # "Prefer not to answer" is a non-response, not an income band
  monthly_income    = factor(ifelse(d$Income == "Prefer not to answer", NA, d$Income),
                             levels = c("0  - 15,000", "16,000 - 50,000", "51,000 - 100,000",
                                        "100,000 - 200,000", "Above 200,000")),
  bmi_category      = factor(d$BMI, levels = c("Underweight", "Healthy", "Overweight", "Obese")),
  current_smoker    = yn01(d$`Current Smoker`),
  current_alcohol   = yn01(d$`Current Alcohol`),
  adequate_activity = yn01(d$`Adequate Physical Activity`),
  adequate_fruit_veg = yn01(d$`Sufficient Fruits and Vegetable Intake`),
  diabetes          = factor(tolower(d$Diabetes), levels = c("no", "yes", "unknown")),
  cardiovascular_disease = factor(tolower(d$CardiovascularDisease), levels = c("no", "yes", "unknown")),
  study_site        = factor(tolower(d$`Study Site`)),
  prior_htn_diagnosis = yn01(d$`Prior Diagnosis of Hypertension`),
  systolic_bp       = as.numeric(d$`Average SBP`),
  diastolic_bp      = as.numeric(d$`Average DBP`),
  hypertension      = yn01(d$Hypertension)
))
# `Current use of prescribed medication` dropped: asked only of the 412 people
# with a prior diagnosis, so it is 71% missing by design.

# ---------------------------------------------------------------------------
# 13. peru_child_anaemia -- PLOS Glob Public Health 2024 (Peru ENDES 2022)
#     Restricted to the paper's analytic subpopulation, which removes every
#     DHS sentinel code (9996-9999) from the haemoglobin and height-for-age
#     columns and leaves both outcomes fully observed.
# ---------------------------------------------------------------------------
d <- read_dta(file.path(RAW2, "peru_child_anaemia/journal.pgph.0002914.s002.dta"))
d <- d[as.integer(d[["Subpoblación"]]) == 1, ]
stopifnot(max(d$HW56, na.rm = TRUE) < 9990, max(d$HW70, na.rm = TRUE) < 9990)
put("peru_child_anaemia", tibble(
  child_age_months   = as.numeric(d$HW1),
  child_sex          = lab(as.integer(d[["Sexo_niño"]]), c(2, 1), c("female", "male")),
  birth_weight       = lab(as.integer(d[["Peso_nacer_niño"]]), 0:2,
                           c("low", "normal", "macrosomia")),
  caesarean_delivery = as.integer(d$Parto_cesarea),
  multiple_birth     = as.integer(d[["Característica_parto"]]),
  breastfed_immediately = as.integer(d$Lactancia_materna_inmediata),
  mother_education   = lab(as.integer(d[["Educación_madre"]]), 0:2,
                           c("primary_or_none", "secondary", "higher")),
  father_education   = lab(as.integer(d[["Educación_padre"]]), 0:2,
                           c("primary_or_none", "secondary", "higher")),
  mother_working     = as.integer(d$Madre_trabaja_actualmente),
  mother_partnered   = as.integer(as.integer(d$Estado_civil_madre) == 0),
  mother_insured     = as.integer(d$Cobertura_seguro_madre),
  mother_indigenous  = as.integer(d$Pertenencia_grupo_etnico_materno),
  residence          = lab(as.integer(d$Residencia), c(1, 2), c("urban", "rural")),
  natural_region     = lab(as.integer(d$Region_natural), 1:4,
                           c("lima_metropolitan", "rest_of_coast", "highlands", "jungle")),
  wealth_index       = lab(as.integer(d$Indice_de_riqueza), 1:5,
                           c("poorest", "poorer", "middle", "richer", "richest")),
  unimproved_water   = as.integer(d$Fuente_agua_dicotomica),
  unimproved_floor   = as.integer(d$Tipo_material_piso),
  unimproved_walls   = as.integer(d$Tipo_material_pared),
  haemoglobin_g_dl   = as.numeric(d$HW56) / 10,       # DHS stores g/dL x 10
  height_for_age_z   = as.numeric(d$HW70) / 100,      # DHS stores Z x 100
  anaemia            = as.integer(d[["Estado_anemia_dicotómico"]]),
  stunting           = as.integer(d[["Desnutrición_crónica_dicotómi"]]),
  survey_weight      = as.numeric(d$Peso_muestral)
))
# `Diversidad_minima_dieta` dropped (36% missing -- asked of a feeding subsample)

# ---------------------------------------------------------------------------
# 14. bangladesh_growth_monitoring -- PLOS ONE 2025, GMP service utilisation
# ---------------------------------------------------------------------------
d <- suppressWarnings(read_excel(
  file.path(RAW2, "bangladesh_growth_monitoring/journal.pone.0324918.s003.xls"),
  sheet = 1, .name_repair = "minimal"))
put("bangladesh_growth_monitoring", tibble(
  delivery_model     = factor(ifelse(d$Group == "home-based GMP", "home_based", "facility_based"),
                              levels = c("facility_based", "home_based")),
  # Labelled, not bare 1-6: written as integers these read back from CSV as
  # numeric and get fitted as a continuous predictor by mistake.
  sub_district       = factor(paste0("upazila_", d$SUB_DISTRICT)),
  mother_age_yrs     = as.numeric(d$`Mother's age`),
  mother_schooling_yrs = as.numeric(d$`Which highest class have you passed?`),
  mother_occupation  = factor(tolower(d$`Mother occupation`)),
  religion_muslim    = as.integer(d$Religion == "Muslim"),
  ngo_member         = yn01(d$`member of an association, or an NGO program`),
  asset_index_tertile = factor(d$`Asset index`, levels = c(1, 2, 3),
                               labels = c("lowest", "middle", "highest")),
  improved_toilet    = as.integer(d$`Toilet facility` == "Improved"),
  child_sex          = factor(tolower(d$`Child sex`), levels = c("female", "male")),
  child_age_months   = as.numeric(d$`Child's age`),
  height_for_age_z   = as.numeric(d$`Length/height-for-age Z-score`),
  weight_for_age_z   = as.numeric(d$`Weight-for-age Z-score`),
  weight_for_height_z = as.numeric(d$`Weight-for-length/height Z-score`),
  stunted            = as.integer(as.numeric(d$`Length/height-for-age Z-score`) < -2),
  heard_of_gmp       = yn01(d$`Caregivers heard about GMP or GMP card`),
  received_gmp_card  = yn01(d$`Received GMP card`),
  can_explain_chart  = yn01(d$`Mother/caregiver can explain the purpose of growth chart`),
  explains_chart_colours = yn01(d$`Caregivers correctly explained the colors in growth chart`)
))
# Dropped: drinking-water source (3,029 of 3,038 use a tubewell -- effectively constant);
# husband's schooling (30% missing, and uses 99 as a sentinel);
# "Ever attended GMP" (50% missing, asked only of those who had heard of it);
# duplicated religion / NGO-membership columns.

# ---------------------------------------------------------------------------
# Additional paper-linked teaching datasets. Download the deposited source
# files only when they are not already present in the staging area.
# ---------------------------------------------------------------------------
fetch_source <- function(url, path) {
  if (!file.exists(path)) {
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    download.file(url, path, mode = "wb", quiet = TRUE)
  }
  path
}

# ---------------------------------------------------------------------------
# 15. serbia_antibiotics -- Horvat et al., PLOS ONE 2017
# ---------------------------------------------------------------------------
p <- fetch_source(
  "https://doi.org/10.1371/journal.pone.0180799.s002",
  file.path(RAW, "29_serbia_antibiotics/serbia_antibiotics.sav"))
d <- read_sav(p)
put("serbia_antibiotics", tibble(
  id                    = seq_len(nrow(d)),
  age_group             = factor(as.character(as_factor(d$age)),
                                 levels = c("18-24", "25-34", "35-44", "45-54", "55-64", ">65")),
  sex                   = factor(tolower(as.character(as_factor(d$sex))),
                                 levels = c("female", "male")),
  marital_status        = factor(tolower(gsub("-", "_", as.character(as_factor(d$marital_status)))),
                                 levels = c("married", "single", "divorced_widowed")),
  education_level       = factor(as.character(as_factor(d$highest_education_received)),
                                 levels = c("primary", "secondary", "tertiary")),
  employment_status     = factor(tolower(as.character(as_factor(d$employment_status))),
                                 levels = c("employed", "unemployed", "pensioner")),
  household_size        = factor(as.character(as_factor(d$household_size)),
                                 levels = c("1-3", "3-5", "more than 5")),
  gp_visits_past_year   = factor(as.character(as_factor(d$No_of_visits_to_gp)),
                                 levels = c("0", "1-4", "5-10", "more than 10")),
  health_professional_in_family = as.integer(d$healthcare_professional_family_memeber == 1),
  knowledge_score       = as.numeric(d$knowledge_scor),
  # The deposited `knowledge_category` conflicts with the score and paper.
  # Recreate the published definition directly: score 9 or above is adequate.
  adequate_knowledge    = as.integer(d$knowledge_scor >= 9),
  self_medication       = as.integer(d$self_medication_category == 1)
))

# ---------------------------------------------------------------------------
# 16. workplace_protection -- Ingram et al., BMC Public Health 2022
# ---------------------------------------------------------------------------
p <- fetch_source(
  paste0("https://static-content.springer.com/esm/art%3A10.1186%2F",
         "s12889-022-12500-w/MediaObjects/12889_2022_12500_MOESM1_ESM.xlsx"),
  file.path(RAW, "30_workplace_protection/workplace_protection.xlsx"))
d <- read_excel(p, sheet = "Complete survey results", skip = 1,
                .name_repair = "minimal")
names(d) <- c("vaccination", "management", "sex", "education", "age",
              "country", "sector", "protected")
put("workplace_protection", tibble(
  id                  = seq_len(nrow(d)),
  vaccination_status  = factor(case_when(
    d$vaccination == "Vaccinated - Fully or Partially" ~ "vaccinated",
    d$vaccination == "Vaccine Refusal" ~ "vaccine_refusal",
    TRUE ~ "no_vaccine_access"
  ), levels = c("vaccinated", "vaccine_refusal", "no_vaccine_access")),
  management_role     = as.integer(d$management == "Management"),
  sex                 = factor(tolower(d$sex), levels = c("female", "male")),
  college_degree      = as.integer(d$education == "College degree or higher"),
  age_group           = factor(case_when(
    d$age == "< 35" ~ "under_35",
    d$age == "Between 35 and 45" ~ "35_to_45",
    d$age == "Between 45 and 55" ~ "45_to_55",
    TRUE ~ "over_55"
  ), levels = c("under_35", "35_to_45", "45_to_55", "over_55")),
  country             = factor(case_when(
    d$country == "United Kingdom of Great Britain and Northern Ireland" ~ "united_kingdom",
    TRUE ~ tolower(d$country)
  )),
  occupational_sector = factor(gsub(" ", "_", tolower(d$sector))),
  feels_protected     = as.integer(d$protected == "Yes")
))

# ---------------------------------------------------------------------------
# 17. sciatica_primary_care -- Stynes et al., PLOS ONE 2018
# ---------------------------------------------------------------------------
p <- fetch_source(
  "https://doi.org/10.1371/journal.pone.0191852.s002",
  file.path(RAW, "31_sciatica_primary_care/sciatica_primary_care.xlsx"))
d <- read_excel(p, sheet = "Dataset", .name_repair = "minimal")
duration_factor <- function(v) factor(v, levels = 0:2,
                                     labels = c("under_6_weeks", "6_to_12_weeks", "over_12_weeks"))
put("sciatica_primary_care", tibble(
  id                         = as.integer(d$Number),
  age_yrs                    = as.numeric(d$Age),
  sex                        = factor(tolower(d$Gender), levels = c("f", "m"),
                                      labels = c("female", "male")),
  bmi                        = as.numeric(d$BMI),
  disability_score           = as.numeric(d$`RMDQ (0-23)`),
  back_pain_intensity        = as.numeric(d$`Back pain intensity (0-10)`),
  leg_pain_intensity         = as.numeric(d$`Leg pain Intensity (0-10)`),
  back_pain_duration         = duration_factor(d$`Back pain duration`),
  leg_pain_duration          = duration_factor(d$`Leg pain duration`),
  comorbidities              = factor(d$Comorbidities, levels = 0:2,
                                      labels = c("none", "one", "two_or_more")),
  general_health             = factor(d$`General health`, levels = 1:3,
                                      labels = c("good", "fair", "poor")),
  leg_pain_worse             = as.integer(d$`Leg pain worse`),
  cough_sneeze_positive      = as.integer(d$`Cough/sneeze positive`),
  subjective_sensory_changes = as.integer(d$`Subjective sensory changes`),
  below_knee_pain            = as.integer(d$`Below knee pain`),
  neural_tension_positive    = as.integer(d$`Neural tension (yes=1)`),
  neurological_deficit       = as.integer(d$`Neurological deficit (myotome, reflex or sensory) yes=1`),
  mri_nerve_root_compression = as.integer(d$`Nerve root compression on MRI`),
  clinician_sciatica         = as.integer(d$`Clinician diagnosis sciatica (1=yes)`)
))

# ---------------------------------------------------------------------------
# 18. nigeria_malaria_households -- Onyiah et al., PLOS ONE 2018
# ---------------------------------------------------------------------------
p <- fetch_source(
  paste0("https://zenodo.org/api/records/4977228/files/",
         "LLIN%20use%20and%20malaria%20parasitemia%20in%20Abuja%20",
         "analysis%20(1)%20(1).xls/content"),
  file.path(RAW, "32_nigeria_malaria/nigeria_malaria_households.xls"))
d <- read_excel(p, sheet = "Results", .name_repair = "minimal")
clean_level <- function(v) {
  v <- gsub("[^a-z0-9]+", "_", tolower(v))
  gsub("^_|_$", "", v)
}
put("nigeria_malaria_households", tibble(
  age_group = factor(case_when(
    d$Age_RECODED == "<5" ~ "under_5",
    d$Age_RECODED == "5-9" ~ "5_to_9",
    d$Age_RECODED == "10-19" ~ "10_to_19",
    d$Age_RECODED == "20-24" ~ "20_to_24",
    d$Age_RECODED == "25-34" ~ "25_to_34",
    TRUE ~ "35_plus"
  ), levels = c("under_5", "5_to_9", "10_to_19", "20_to_24", "25_to_34", "35_plus")),
  sex                    = factor(tolower(d$Sex), levels = c("female", "male")),
  education              = factor(tolower(d$Educationallevel),
                                  levels = c("none", "primary", "secondary", "tertiary")),
  area_council           = lab(as.integer(d$AreaCouncilCode), 1:3,
                               c("abuja_municipal", "kuje", "kwali")),
  bushes_near_home       = as.integer(d$`Bushes around the house` == "yes"),
  owns_llin              = as.integer(d$`Own net` == "yes"),
  used_llin_previous_night = as.integer(d$`LLIN use_RECODED` == 1),
  house_floor            = factor(clean_level(d$Housefloor)),
  house_roof             = factor(clean_level(d$Houseroof)),
  house_type             = factor(clean_level(d$Housetype)),
  house_wall             = factor(clean_level(d$Housewall)),
  house_window           = factor(clean_level(d$Housewindow)),
  same_room_as_index_case = as.integer(d$`sleep same room with patient` == "yes"),
  uncovered_water_receptacles = as.integer(d$`Uncovered water receptacles` == "yes"),
  malaria_parasitaemia   = as.integer(d$`Microscopy result` == "pos")
))
# The deposited RDT result and parasite-density count are direct laboratory
# manifestations of the microscopy outcome, so they are not shipped as predictors.

# ---------------------------------------------------------------------------
# Ageing quality of life -- Raggi et al., PLOS ONE 2016 (COURAGE in Europe)
# ---------------------------------------------------------------------------
p <- fetch_source(
  paste0("https://journals.plos.org/plosone/article/file?type=supplementary&",
         "id=info:doi/10.1371/journal.pone.0159293.s001"),
  file.path(RAW, "33_ageing_qol_europe/ageing_qol_europe.tsv"))
d <- read_tsv(p, show_col_types = FALSE)
difficulty4 <- function(v) lab(as.integer(v), 1:4,
                               c("none", "mild", "moderate", "severe_or_extreme"))
put("ageing_qol_europe", tibble(
  residence                    = factor(tolower(d$q0104), levels = c("rural", "urban")),
  age_yrs                      = as.numeric(d$q1011),
  neighbourhood_usability     = as.numeric(d$score_8200_1),
  walkability_hindrance       = as.numeric(d$score_8200_2),
  public_buildings_usability  = as.numeric(d$score_8300),
  home_usability              = as.numeric(d$score_8400),
  social_network_index        = as.numeric(d$sns_tot_weigh),
  quality_of_life_score       = as.numeric(d$qol_totalscore),
  country                     = lab(as.integer(d$country), 1:3, c("finland", "poland", "spain")),
  sex                         = lab(as.integer(d$sex), c(0, 1), c("male", "female")),
  marital_status              = lab(as.integer(d$marital), 0:3,
                                    c("married_or_cohabiting", "never_married",
                                      "separated_or_divorced", "widowed")),
  education                   = lab(as.integer(d$educ_two), 0:2,
                                    c("none_or_informal", "primary_or_secondary",
                                      "high_school_or_higher")),
  body_pain                   = lab(as.integer(d$body_pain), 1:5,
                                    c("none", "pain_without_difficulty", "mild_difficulty",
                                      "moderate_difficulty", "severe_or_extreme_difficulty")),
  sleep_difficulty            = difficulty4(d$diff_sleep),
  tiredness                   = difficulty4(d$feel_tired),
  learning_difficulty         = difficulty4(d$diff_learn),
  concentration_difficulty    = difficulty4(d$diff_concentr),
  sadness                     = difficulty4(d$depress),
  anxiety                     = difficulty4(d$anxiety),
  health_worry                = difficulty4(d$health_worry),
  worry_interference          = difficulty4(d$worry_interfer),
  distance_vision_difficulty  = difficulty4(d$dist_vision),
  near_vision_difficulty      = difficulty4(d$near_vision),
  near_hearing_difficulty     = difficulty4(d$near_hearing),
  conversation_hearing_difficulty = difficulty4(d$nihl),
  bmi_category                = lab(as.integer(d$bmi_three), 1:3,
                                    c("normal_or_underweight", "overweight", "obese")),
  high_waist_risk             = as.integer(d$waist_risk),
  smoking_status              = lab(as.integer(d$smoke), 0:2,
                                    c("never", "former", "current")),
  alcohol_use                 = lab(as.integer(d$alcohol), 0:3,
                                    c("abstainer_or_occasional", "non_heavy",
                                      "infrequent_heavy", "frequent_heavy")),
  physical_activity           = lab(as.integer(d$physical), 1:3,
                                    c("high", "moderate", "low")),
  arthritis                   = as.integer(d$arthritis),
  stroke                      = as.integer(d$stroke),
  angina                      = as.integer(d$angina),
  diabetes                    = as.integer(d$diabetes),
  lung_disease                = as.integer(d$lung),
  asthma                      = as.integer(d$asthma),
  depression_diagnosis        = as.integer(d$depression),
  hypertension                = as.integer(d$hypertension)
))

# ---------------------------------------------------------------------------
# ESSENS adolescent screen time -- Chortatos et al., PLOS ONE 2020
# ---------------------------------------------------------------------------
p <- fetch_source(
  "https://datadryad.org/downloads/file_stream/523333",
  file.path(RAW, "34_essens_adolescents/ESSENS_study_PlosOne_2.sav"))
d <- read_sav(p)
drop_100 <- function(v) {
  v <- as.numeric(v)
  v[v == 100] <- NA
  v
}
binary_from <- function(v, yes) {
  v <- drop_100(v)
  ifelse(is.na(v), NA_integer_, as.integer(v == yes))
}
put("essens_adolescents", tibble(
  bedroom_tv = lab(drop_100(d$V101_Bedroom_TV), 1:2, c("yes", "no")),
  owns_pc = lab(drop_100(d$V102_Own_PC), 1:2, c("yes", "no")),
  sex = lab(drop_100(d$Gender), c(1, 0), c("boy", "girl")),
  homework_weekday = lab(drop_100(d$V115_compressed), 1:3,
                         c("up_to_30_min", "1_to_2_hours", "2_plus_hours")),
  hobbies_weekday = lab(drop_100(d$V117_compressed), 1:3,
                        c("up_to_30_min", "1_to_2_hours", "2_plus_hours")),
  homework_weekend = lab(drop_100(d$V119_compressed), 1:3,
                         c("up_to_30_min", "1_to_2_hours", "2_plus_hours")),
  hobbies_weekend = lab(drop_100(d$V121_compressed), 1:3,
                        c("up_to_30_min", "1_to_2_hours", "2_plus_hours")),
  pa_self_efficacy_high = binary_from(d$SE_PA2_binary_strict, 1),
  tv_self_efficacy_high = binary_from(d$new_SETV, 1),
  gaming_self_efficacy_high = binary_from(d$new_SEPC, 1),
  pcs_in_house = lab(drop_100(d$PC_in_house), 1:2, c("zero_to_two", "more_than_two")),
  active_5plus_days = binary_from(d$PAL_typical_wk, 2),
  weekday_tv = lab(drop_100(d$new_TV_cat1), 1:3,
                   c("up_to_1_5_hours", "2_to_2_5_hours", "3_plus_hours")),
  weekday_gaming = lab(drop_100(d$new_PC_cat1), 1:3,
                       c("up_to_1_hour", "1_5_to_2_hours", "2_5_plus_hours")),
  weekday_online = lab(drop_100(d$new_Net_cat1), 1:3,
                       c("up_to_1_5_hours", "2_to_2_5_hours", "3_plus_hours")),
  weekend_tv = lab(drop_100(d$new_TVWE_cat11), 1:3,
                   c("up_to_2_5_hours", "3_to_3_5_hours", "4_plus_hours")),
  weekend_gaming = lab(drop_100(d$new_PCWE_cat1), 1:3,
                       c("up_to_1_5_hours", "2_to_2_5_hours", "3_plus_hours")),
  weekend_online = lab(drop_100(d$new_NetWE_cat1), 1:3,
                       c("up_to_1_5_hours", "2_to_2_5_hours", "3_plus_hours")),
  high_weekday_tv = binary_from(d$new_TV_cat1, 3),
  high_weekday_gaming = binary_from(d$new_PC_cat1, 3)
))
# The source uses 100 as a missing-value sentinel. It is converted to NA here.

cat("\nTotal datasets: ", length(written), "\n", sep = "")
