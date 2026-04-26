#!/usr/bin/env Rscript
# Prepare health datasets for OLS linear-regression teaching.
#
# For each dataset:
#   - load (from R package or local raw file),
#   - clean/rename columns to tidy snake_case,
#   - save CSV to datasets_staging/cleaned/<slug>.csv,
#   - render a small PNG thumbnail (outcome vs. main predictor, with lm line) to
#     datasets_staging/thumbnails/<slug>.png,
#   - append a row to datasets_staging/cleaned/_summary.csv (row_id, slug, n, p,
#     columns, outcome, main_predictor, thumbnail, csv, license, load_code).

suppressPackageStartupMessages({
  library(readr); library(readxl); library(haven)
  library(dplyr); library(ggplot2); library(tibble)
})

args_file <- tryCatch(
  {
    sf <- sys.frame(1)$ofile
    if (is.null(sf)) stop("no ofile")
    sf
  },
  error = function(e) {
    # Running via Rscript: commandArgs carries --file=<path>
    m <- regmatches(commandArgs(trailingOnly = FALSE),
                    regexpr("--file=.*", commandArgs(trailingOnly = FALSE)))
    if (length(m)) sub("^--file=", "", m) else "datasets_staging/scripts/prepare_datasets.R"
  }
)
here <- normalizePath(dirname(args_file))
root <- normalizePath(file.path(here, ".."))
raw <- file.path(root, "raw")
cleaned <- file.path(root, "cleaned")
thumbs <- file.path(root, "thumbnails")
dir.create(cleaned, showWarnings = FALSE, recursive = TRUE)
dir.create(thumbs,  showWarnings = FALSE, recursive = TRUE)

message("Writing outputs to:\n  ", cleaned, "\n  ", thumbs)

# ----- helpers -------------------------------------------------------------- #

make_thumb <- function(df, x, y, slug, subtitle = NULL) {
  p <- ggplot(df, aes(x = .data[[x]], y = .data[[y]])) +
    geom_point(alpha = 0.55, size = 1.1, colour = "#2c7fb8") +
    geom_smooth(method = "lm", se = TRUE, colour = "#e34a33",
                linewidth = 0.8, fill = "#fdbb84", alpha = 0.3) +
    labs(x = x, y = y, subtitle = subtitle) +
    theme_minimal(base_size = 9) +
    theme(plot.subtitle = element_text(size = 7, colour = "grey40"),
          panel.grid.minor = element_blank())
  ggsave(file.path(thumbs, paste0(slug, ".png")), p,
         width = 3.2, height = 2.2, dpi = 140, bg = "white")
  invisible(p)
}

write_clean <- function(df, slug) {
  path <- file.path(cleaned, paste0(slug, ".csv"))
  write_csv(df, path)
  path
}

summary_rows <- list()
add_summary <- function(row_id, slug, df, outcome, main_predictor, license,
                        load_code, source_name) {
  summary_rows[[length(summary_rows) + 1]] <<- tibble(
    row_id = row_id,
    slug = slug,
    source = source_name,
    n = nrow(df),
    p = ncol(df),
    outcome = outcome,
    main_predictor = main_predictor,
    columns = paste(names(df), collapse = ", "),
    license = license,
    load_code = load_code,
    csv = file.path("datasets_staging/cleaned", paste0(slug, ".csv")),
    thumbnail = file.path("datasets_staging/thumbnails", paste0(slug, ".png"))
  )
}

# --------------------------------------------------------------------------- #
# Row 1 — Body fat (Johnson/Penrose), 252 men. Fixed-width text file.         #
# --------------------------------------------------------------------------- #
message("\n[1] bodyfat_johnson")
fat_widths <- c(7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8)
fat_cols <- c("case","bodyfat_brozek","bodyfat_siri","density",
              "age","weight_lb","height_in","adiposity_bmi",
              "fat_free_wt_lb","neck_cm","chest_cm","abdomen_cm",
              "hip_cm","thigh_cm","knee_cm","ankle_cm",
              "biceps_cm","forearm_cm","wrist_cm")
bodyfat_johnson <- read.fwf(file.path(raw, "fat.dat.txt"),
                            widths = fat_widths, col.names = fat_cols,
                            stringsAsFactors = FALSE)
bodyfat_johnson <- as_tibble(bodyfat_johnson)
write_clean(bodyfat_johnson, "01_bodyfat_johnson")
make_thumb(bodyfat_johnson, "abdomen_cm", "bodyfat_brozek",
           "01_bodyfat_johnson", "Body fat % vs abdomen circumference")
add_summary(1, "01_bodyfat_johnson", bodyfat_johnson,
            outcome = "bodyfat_brozek", main_predictor = "abdomen_cm",
            license = "Free for non-commercial use (Fisher/JSE)",
            load_code = 'read.csv("datasets_staging/cleaned/01_bodyfat_johnson.csv")',
            source_name = "JSE (Johnson 1996)")

# --------------------------------------------------------------------------- #
# Row 2 — Diabetes progression (Efron et al., 2004), n=442.                   #
# --------------------------------------------------------------------------- #
message("[2] diabetes_efron")
diabetes <- read_tsv(file.path(raw, "diabetes.tab.txt"),
                     show_col_types = FALSE) |>
  rename(age = AGE, sex = SEX, bmi = BMI, bp_mean = BP,
         total_cholesterol = S1, ldl = S2, hdl = S3,
         tc_hdl_ratio = S4, log_serum_triglycerides = S5,
         glucose = S6, progression = Y)
write_clean(diabetes, "02_diabetes_efron")
make_thumb(diabetes, "bmi", "progression", "02_diabetes_efron",
           "Disease progression vs BMI")
add_summary(2, "02_diabetes_efron", diabetes,
            outcome = "progression", main_predictor = "bmi",
            license = "Public (scikit-learn redistributes it)",
            load_code = 'read.csv("datasets_staging/cleaned/02_diabetes_efron.csv")',
            source_name = "Efron et al. 2004")

# --------------------------------------------------------------------------- #
# Row 3 — FEV in youths (Kahn/Tager), n=654. Fixed width.                     #
# --------------------------------------------------------------------------- #
message("[3] fev_kahn")
fev_widths <- c(3, 8, 7, 5, 5)
fev_cols <- c("age_yrs", "fev_l", "height_in", "sex", "smoke")
fev <- read.fwf(file.path(raw, "fev.dat.txt"), widths = fev_widths,
                col.names = fev_cols, stringsAsFactors = FALSE) |>
  as_tibble() |>
  mutate(sex = factor(sex, levels = c(0, 1), labels = c("female", "male")),
         smoke = factor(smoke, levels = c(0, 1), labels = c("non_smoker", "smoker")))
write_clean(fev, "03_fev_kahn")
make_thumb(fev, "height_in", "fev_l", "03_fev_kahn",
           "FEV (L) vs height (inches)")
add_summary(3, "03_fev_kahn", fev,
            outcome = "fev_l", main_predictor = "height_in",
            license = "Free for non-commercial use (JSE)",
            load_code = 'read.csv("datasets_staging/cleaned/03_fev_kahn.csv")',
            source_name = "JSE (Kahn 2005)")

# --------------------------------------------------------------------------- #
# Row 4 — Body fat — Spanish adults (Fuster-Parra), n≈3200.                    #
# File shipped with .xlsx extension but is actually a quoted text table.      #
# --------------------------------------------------------------------------- #
message("[4] bodyfat_spanish")
bodyfat_es <- read.table(file.path(raw, "PLOS_bodyfat_spanish.xlsx"),
                         header = TRUE, stringsAsFactors = FALSE) |>
  as_tibble() |>
  rename(age = Age, bai = BAI, bmi = BMI,
         bodyfat_pct = BodyFat, sex_code = Gender) |>
  mutate(sex = factor(sex_code, levels = c(1, 2),
                      labels = c("male", "female"))) |>
  select(age, sex, bai, bmi, bodyfat_pct)
write_clean(bodyfat_es, "04_bodyfat_spanish")
make_thumb(bodyfat_es, "bmi", "bodyfat_pct", "04_bodyfat_spanish",
           "Body fat % vs BMI (Spanish adults)")
add_summary(4, "04_bodyfat_spanish", bodyfat_es,
            outcome = "bodyfat_pct", main_predictor = "bmi",
            license = "CC-BY 4.0 (PLOS ONE)",
            load_code = 'read.csv("datasets_staging/cleaned/04_bodyfat_spanish.csv")',
            source_name = "Fuster-Parra 2015 (PLOS ONE)")

# --------------------------------------------------------------------------- #
# Row 5 — Birthweight from ultrasound (Secher), n=107. Semicolon-separated.   #
# --------------------------------------------------------------------------- #
message("[5] birthweight_secher")
bw_secher <- read_delim(file.path(raw, "BirthWeight.csv"), delim = ";",
                        show_col_types = FALSE) |>
  rename(birthweight_g = bw, biparietal_diameter_mm = bpd,
         abdominal_diameter_mm = ad, id = idnr) |>
  select(id, biparietal_diameter_mm, abdominal_diameter_mm, birthweight_g)
write_clean(bw_secher, "05_birthweight_secher")
make_thumb(bw_secher, "abdominal_diameter_mm", "birthweight_g",
           "05_birthweight_secher",
           "Birthweight (g) vs abdominal diameter (mm)")
add_summary(5, "05_birthweight_secher", bw_secher,
            outcome = "birthweight_g", main_predictor = "abdominal_diameter_mm",
            license = "Textbook supplement (public download)",
            load_code = 'read.csv("datasets_staging/cleaned/05_birthweight_secher.csv")',
            source_name = "Ekstrøm & Sørensen 2015")

# --------------------------------------------------------------------------- #
# Row 6 — Peak power & tibial bone strength (Denys & Yingling, Dryad CC0).    #
# 142 adults (79 F, 63 M). Outcomes: BSIc (compressive bone-strength index)   #
# and SSI (polar strength-strain index).                                       #
# --------------------------------------------------------------------------- #
message("[6] peak_power_bone")
denys <- read_csv(file.path(raw, "06_peak_power_bone",
                            "Peak_power_and_body_mass_data-v3.csv"),
                  show_col_types = FALSE) |>
  rename(subject_id = ID, gender_code = Gender_A, age_bin = `Age Bins`,
         body_mass_kg = BM, peak_power_w = PP, relative_peak_power = RPP,
         bsi_compression = BSIc, polar_strength_strain_index = SSI) |>
  mutate(sex = factor(gender_code, levels = c(1, 2),
                      labels = c("female", "male"))) |>
  select(subject_id, sex, age_bin, body_mass_kg, peak_power_w,
         relative_peak_power, bsi_compression, polar_strength_strain_index)
write_clean(denys, "06_peak_power_bone")
make_thumb(denys, "peak_power_w", "bsi_compression", "06_peak_power_bone",
           "Tibial bone strength (BSIc) vs peak power (W)")
add_summary(6, "06_peak_power_bone", denys,
            outcome = "bsi_compression", main_predictor = "peak_power_w",
            license = "CC0 1.0 (Dryad)",
            load_code = 'read.csv("datasets_staging/cleaned/06_peak_power_bone.csv")',
            source_name = "Denys & Yingling 2022 (MSSE) via Dryad")

# --------------------------------------------------------------------------- #
# Row 7 — Alcohol metabolism (Frezza, NEJM), n=32, Sleuth3::case1101.         #
# --------------------------------------------------------------------------- #
message("[7] alcohol_metabolism")
data(case1101, package = "Sleuth3")
alcohol <- as_tibble(case1101) |>
  rename(subject = Subject, first_pass_metabolism = Metabol,
         gastric_ad_activity = Gastric, sex = Sex, alcohol_status = Alcohol) |>
  mutate(sex = tolower(as.character(sex)),
         alcohol_status = tolower(gsub("-", "_", as.character(alcohol_status))))
write_clean(alcohol, "07_alcohol_metabolism")
make_thumb(alcohol, "gastric_ad_activity", "first_pass_metabolism",
           "07_alcohol_metabolism",
           "First-pass metabolism vs gastric AD activity")
add_summary(7, "07_alcohol_metabolism", alcohol,
            outcome = "first_pass_metabolism",
            main_predictor = "gastric_ad_activity",
            license = "GPL-2 (redistributed via Sleuth3 pkg)",
            load_code = 'data(case1101, package = "Sleuth3")',
            source_name = "Frezza 1990 (NEJM) via Sleuth3")

# --------------------------------------------------------------------------- #
# Row 8 — Cystic fibrosis lung function, n=25, ISwR::cystfibr.                #
# --------------------------------------------------------------------------- #
message("[8] cystic_fibrosis")
data(cystfibr, package = "ISwR")
cf <- as_tibble(cystfibr) |>
  rename(age_yrs = age, sex_code = sex, height_cm = height, weight_kg = weight,
         bmp_pct = bmp, fev1_pct = fev1,
         residual_volume = rv, functional_residual_capacity = frc,
         total_lung_capacity = tlc, pe_max = pemax) |>
  mutate(sex = factor(sex_code, levels = c(0, 1), labels = c("male", "female"))) |>
  select(age_yrs, sex, height_cm, weight_kg, bmp_pct, fev1_pct,
         residual_volume, functional_residual_capacity,
         total_lung_capacity, pe_max)
write_clean(cf, "08_cystic_fibrosis")
make_thumb(cf, "weight_kg", "pe_max", "08_cystic_fibrosis",
           "PEmax vs weight (kg)")
add_summary(8, "08_cystic_fibrosis", cf,
            outcome = "pe_max", main_predictor = "weight_kg",
            license = "GPL-2 (redistributed via ISwR pkg)",
            load_code = 'data(cystfibr, package = "ISwR")',
            source_name = "O'Neill 1983 via ISwR")

# --------------------------------------------------------------------------- #
# Row 9 — DXA body fat German women (Garcia), n=71, TH.data::bodyfat.         #
# --------------------------------------------------------------------------- #
message("[9] bodyfat_german")
data(bodyfat, package = "TH.data")
bf_de <- as_tibble(bodyfat) |>
  rename(age_yrs = age, dxa_fat_kg = DEXfat, waist_cm = waistcirc,
         hip_cm = hipcirc, elbow_breadth_cm = elbowbreadth,
         knee_breadth_cm = kneebreadth,
         anthro_3a = anthro3a, anthro_3b = anthro3b,
         anthro_3c = anthro3c, anthro_4 = anthro4)
write_clean(bf_de, "09_bodyfat_german")
make_thumb(bf_de, "waist_cm", "dxa_fat_kg", "09_bodyfat_german",
           "DXA body fat (kg) vs waist (cm)")
add_summary(9, "09_bodyfat_german", bf_de,
            outcome = "dxa_fat_kg", main_predictor = "waist_cm",
            license = "GPL-2 (redistributed via TH.data pkg)",
            load_code = 'data(bodyfat, package = "TH.data")',
            source_name = "Garcia 2005 via TH.data")

# --------------------------------------------------------------------------- #
# Row 10 — Baystate birth weight (Hosmer), n=189, MASS::birthwt.              #
# --------------------------------------------------------------------------- #
message("[10] birthweight_baystate")
data(birthwt, package = "MASS")
bw_ma <- as_tibble(birthwt) |>
  rename(low_birthweight = low, mother_age_yrs = age,
         mother_weight_lb = lwt, race_code = race,
         smoked_during_pregnancy = smoke,
         previous_premature_labours = ptl,
         history_hypertension = ht, uterine_irritability = ui,
         physician_visits_1st_trimester = ftv,
         birthweight_g = bwt) |>
  mutate(race = factor(race_code, levels = c(1, 2, 3),
                       labels = c("white", "black", "other")),
         smoked_during_pregnancy = as.logical(smoked_during_pregnancy),
         history_hypertension = as.logical(history_hypertension),
         uterine_irritability = as.logical(uterine_irritability),
         low_birthweight = as.logical(low_birthweight)) |>
  select(birthweight_g, low_birthweight, mother_age_yrs, mother_weight_lb,
         race, smoked_during_pregnancy, previous_premature_labours,
         history_hypertension, uterine_irritability,
         physician_visits_1st_trimester)
write_clean(bw_ma, "10_birthweight_baystate")
make_thumb(bw_ma, "mother_weight_lb", "birthweight_g",
           "10_birthweight_baystate",
           "Birthweight (g) vs mother's weight (lb)")
add_summary(10, "10_birthweight_baystate", bw_ma,
            outcome = "birthweight_g", main_predictor = "mother_weight_lb",
            license = "GPL-2 (redistributed via MASS pkg)",
            load_code = 'data(birthwt, package = "MASS")',
            source_name = "Hosmer & Lemeshow 1989 via MASS")

# --------------------------------------------------------------------------- #
# Row 11 — HIV & 6-min walk distance (Frasca 2019, Dryad, CC0). n=427.        #
# --------------------------------------------------------------------------- #
message("[11] hiv_6mwt")
hiv <- read_excel(file.path(raw, "11_hiv_6mwt",
                            "Data for HIV 6MW PLOS ONE_2.xlsx")) |>
  rename(hiv_status = hiv_st, age_yrs = age, sex_code = gender,
         cd4_count = cd4, pack_years = pack_years,
         on_antiretroviral = a4hivmed_current,
         systolic_bp_pre = pre_bp_sys, diastolic_bp_pre = pre_bp_dia,
         six_min_walk_m = dist_meters,
         mmrc_dyspnoea = mmrc_score,
         sgrq_symptoms = symptoms_score, sgrq_activity = activity_score,
         sgrq_impacts = impacts_score, sgrq_total = sgrq_total_score,
         haemoglobin = hgb, viral_load_detectable = vldet,
         post_fvc_pct_pred = post_fvcppp, dlco_pct_pred = dlcopp,
         post_fev1_pct_pred = post_fev1ppp,
         post_fev1_fvc_ratio = post_fev1fvcpp,
         viral_load_copies = vlcorr,
         smoking_status = smokingstatus, drug_use = drug) |>
  mutate(hiv_status = factor(hiv_status, levels = c(0, 1),
                             labels = c("hiv_negative", "hiv_positive")),
         sex = factor(sex_code, levels = c(0, 1),
                      labels = c("male", "female"))) |>
  select(hiv_status, age_yrs, sex, cd4_count, pack_years, on_antiretroviral,
         systolic_bp_pre, diastolic_bp_pre, six_min_walk_m,
         mmrc_dyspnoea, sgrq_symptoms, sgrq_activity, sgrq_impacts, sgrq_total,
         haemoglobin, viral_load_detectable, post_fvc_pct_pred, dlco_pct_pred,
         post_fev1_pct_pred, post_fev1_fvc_ratio, viral_load_copies,
         smoking_status, drug_use)
write_clean(hiv, "11_hiv_6mwt")
make_thumb(hiv, "age_yrs", "six_min_walk_m", "11_hiv_6mwt",
           "6-minute walk distance (m) vs age")
add_summary(11, "11_hiv_6mwt", hiv,
            outcome = "six_min_walk_m", main_predictor = "age_yrs",
            license = "CC0 1.0 (Dryad)",
            load_code = 'read.csv("datasets_staging/cleaned/11_hiv_6mwt.csv")',
            source_name = "Frasca 2019 (PLOS ONE) via Dryad")

# --------------------------------------------------------------------------- #
# Row 12 — Serum GGT & atherosclerosis, n=912, Zenodo (CC0).                  #
# --------------------------------------------------------------------------- #
message("[12] ggt_atherosclerosis")
ggt <- read_excel(file.path(raw, "12_ggt_atherosclerosis",
                            "data bmjopen dryad.xls"))
names(ggt) <- c("id", "sex_code", "age", "bmi", "systolic_bp", "diastolic_bp",
                "ast", "alt", "ggt", "log2_ggt", "fasting_glucose",
                "uric_acid", "total_cholesterol", "triglycerides",
                "hdl_cholesterol", "ldl_cholesterol", "current_smoker",
                "ex_smoker", "alcohol_use", "exercise",
                "fatty_liver", "egfr", "post_menopausal",
                "abi_max", "pwv_max")
ggt <- ggt |>
  mutate(sex = factor(sex_code, levels = c(1, 2),
                      labels = c("male", "female"))) |>
  select(-sex_code)
write_clean(ggt, "12_ggt_atherosclerosis")
make_thumb(ggt, "log2_ggt", "pwv_max", "12_ggt_atherosclerosis",
           "Pulse-wave velocity vs log2(GGT)")
add_summary(12, "12_ggt_atherosclerosis", ggt,
            outcome = "pwv_max", main_predictor = "log2_ggt",
            license = "CC0 1.0 (Zenodo)",
            load_code = 'read.csv("datasets_staging/cleaned/12_ggt_atherosclerosis.csv")',
            source_name = "BMJ Open 2014 via Zenodo")

# --------------------------------------------------------------------------- #
# Row 13 — CV risk factors & CIMT in rheumatoid arthritis, Zenodo (CC0).      #
# --------------------------------------------------------------------------- #
message("[13] cimt_ra")
cimt <- read_sav(file.path(raw, "13_cimt_ra",
                           "traditional CVRF in relation to cIMT_PLOSONE23aug15.sav"))
cimt <- cimt |>
  zap_labels() |>
  zap_label() |>
  as_tibble() |>
  rename(id = idPatient, anti_ccp = antiCCP, apo_b_vnumber = ApoBvNummer,
         plaques = Plaques, sex_code = gender, ra_4_groups = RA_4groepen,
         ra_healthy = RA_gezond, ra_ht_hc = RA_HT_HC, ra = RA,
         ra_ldl_cat = RA_LDLcat, hx_hypertension = VG_hypertensie,
         antihypertensives = Anti_Hypertensives, hx_dyslipidaemia = VG_dyslip,
         statins = Statines, age_yrs = age, height_cm = length,
         weight_kg = weight, bmi = BMI, waist_cm = waist,
         systolic_bp = systolicBP, diastolic_bp = diastoligbp,
         hypertension = Hypertension, smoking = Smoking,
         cimt_total = CIMT_total, prednisone = Prednison,
         glucose = Glucose, total_cholesterol = Cholesterol,
         hdl_cholesterol = HDL_Chol, ldl_cholesterol = LDL_berekend,
         triglycerides = Triglyceriden, crp = CRP,
         apo_a = ApoA, apo_b = ApoB, das28_bse = txtDAS28BSE,
         das28_crp = txtDAS28CRP, cv_risk = CVrisk,
         rheumatoid_factor = RF, erosive_ra = erosiveRA,
         ra_disease_duration = RA_diseasduration, ldl_cutoff_2_5 = LDL_cutoff_2_5,
         nsaid = NSAID, hydroxychloroquine = hydroxychloroquine,
         sulfasalazine = Sulfasalazin, methotrexate = methotrexate,
         leflunomide = Leflunomide, anti_tnf = anti_TNF,
         other_biologicals = other_biologicals, azathioprine = Azathioprine) |>
  mutate(sex = factor(sex_code, levels = c(1, 2),
                      labels = c("male", "female"))) |>
  select(id, sex, age_yrs, bmi, waist_cm, systolic_bp, diastolic_bp,
         total_cholesterol, hdl_cholesterol, ldl_cholesterol, triglycerides,
         crp, smoking, hypertension, statins, ra, das28_bse, das28_crp,
         cimt_total, everything(), -sex_code)
write_clean(cimt, "13_cimt_ra")
make_thumb(filter(cimt, !is.na(age_yrs), !is.na(cimt_total)),
           "age_yrs", "cimt_total", "13_cimt_ra",
           "Carotid intima-media thickness vs age")
add_summary(13, "13_cimt_ra", cimt,
            outcome = "cimt_total", main_predictor = "age_yrs",
            license = "CC0 1.0 (Zenodo)",
            load_code = 'read.csv("datasets_staging/cleaned/13_cimt_ra.csv")',
            source_name = "PLOS ONE 2015 via Zenodo")

# --------------------------------------------------------------------------- #
# Row 14 — Oral-contraceptive use & prostate cancer (Dryad, CC0). n=167.      #
# Country-level ecological data. Numeric columns arrived as text, so coerce.  #
# --------------------------------------------------------------------------- #
message("[14] oc_prostate")
oc <- read_excel(file.path(raw, "14_oc_prostate", "whole_world_ocp.xls")) |>
  rename(country = Countryorarea,
         gdp_usd = gdp,
         prostate_cancer_incidence = incidence,
         prostate_cancer_mortality = mortality,
         pill_use_pct = Pill,
         iud_use_pct = IUD,
         condom_use_pct = Condom,
         vaginal_barrier_use_pct = Vaginalbarriermethodc,
         europe = europe) |>
  mutate(across(c(gdp_usd, prostate_cancer_incidence, prostate_cancer_mortality,
                  iud_use_pct, condom_use_pct, vaginal_barrier_use_pct),
                ~ suppressWarnings(as.numeric(trimws(.x)))),
         europe = as.logical(europe))
write_clean(oc, "14_oc_prostate")
make_thumb(filter(oc, !is.na(pill_use_pct), !is.na(prostate_cancer_incidence)),
           "pill_use_pct", "prostate_cancer_incidence", "14_oc_prostate",
           "Prostate-cancer incidence vs % pill use (country-level)")
add_summary(14, "14_oc_prostate", oc,
            outcome = "prostate_cancer_incidence",
            main_predictor = "pill_use_pct",
            license = "CC0 1.0 (Dryad)",
            load_code = 'read.csv("datasets_staging/cleaned/14_oc_prostate.csv")',
            source_name = "Margel & Fleshner 2011 (BMJ Open) via Dryad")

# --------------------------------------------------------------------------- #
# Row 15 — Patient activation for self-management (Dryad, CC0). n=1154, 94    #
# cols including full PAM/SF-12/HADS/IPQ/SSUP item-level scores.              #
# --------------------------------------------------------------------------- #
message("[15] pam13")
pam <- read_sav(file.path(raw, "15_pam13",
                          "Datafile Patient activation for self-management.sav")) |>
  zap_labels() |>
  zap_label() |>
  as_tibble()
names(pam) <- tolower(gsub("[^[:alnum:]_]+", "_", names(pam)))
# Human-friendly renames for the derived/total columns; keep item-level as-is.
pam <- pam |>
  rename(subject_id = number, sex_code = gender,
         height_cm = length, weight_kg = bodyweight,
         financial_distress = financial_distress,
         smoking_code = smoking,
         nyha = nyha, gold_stage = gold_stadium,
         nyha_class = nyha_class, egfr_ml_min = gfr_mlmin,
         ethnicity_code = ethnicity,
         living_situation = living_situation, age_yrs = age, bmi = bmi,
         pam_activation_score = activation_score,
         pam_level = pam_levels,
         sf12_total = sf12_total_score,
         sf12_physical = sf_phys, sf12_mental = sf_ment,
         hads_depression = hads_depression, hads_anxiety = hads_anxiety,
         ipq_total = ipq_total_score,
         support_family = supp_total_family,
         support_friends = supp_total_friends,
         support_significant_other = supp_total_significantother,
         support_total = supp_total_score,
         n_comorbidities = total_comorbidities,
         disease = disease, education_level = education_level,
         care_allowance = care_allowance,
         disease_duration_yrs = disease_duration,
         dm_medication = medication_dm,
         disease_severity = disease_severity,
         egfr = gfr, charlson_index = charlson) |>
  mutate(sex = factor(sex_code, levels = c(1, 2),
                      labels = c("male", "female"))) |>
  select(subject_id, sex, age_yrs, bmi, height_cm, weight_kg,
         education_level, living_situation, financial_distress,
         smoking_code, disease, disease_severity, disease_duration_yrs,
         n_comorbidities, charlson_index,
         pam_activation_score, pam_level,
         sf12_total, sf12_physical, sf12_mental,
         hads_depression, hads_anxiety, ipq_total,
         support_family, support_friends, support_significant_other,
         support_total, egfr, egfr_ml_min, nyha, nyha_class, gold_stage,
         ethnicity_code, care_allowance, dm_medication,
         everything(), -sex_code)
write_clean(pam, "15_pam13")
make_thumb(filter(pam, !is.na(pam_activation_score), !is.na(sf12_mental)),
           "sf12_mental", "pam_activation_score", "15_pam13",
           "PAM-13 activation vs SF-12 mental summary")
add_summary(15, "15_pam13", pam,
            outcome = "pam_activation_score",
            main_predictor = "sf12_mental",
            license = "CC0 1.0 (Dryad)",
            load_code = 'read.csv("datasets_staging/cleaned/15_pam13.csv")',
            source_name = "Bos-Touwen 2015 (PLOS ONE) via Dryad")

# --------------------------------------------------------------------------- #
# Row 16 — Medical student resilience & QoL (Dryad, CC0). n=1350.             #
# The first spreadsheet row is a title banner; real headers live on row 2.    #
# --------------------------------------------------------------------------- #
message("[16] med_student_qol")
msq <- read_excel(file.path(raw, "16_med_student_qol",
                            "Dataset Resilience Educational Environment QoL.xlsx"),
                  skip = 1) |>
  rename(subject_id = IDR, sex = Sex, group = Group,
         overall_qol = `Overall QoL`,
         medical_school_qol = `Medical school-related QoL`,
         whoqol_physical = `WHOQOL physical health`,
         whoqol_psychological = `WHOQOL psychological`,
         whoqol_social = `WHOQOL social relationships`,
         whoqol_environment = `WHOQOL environment`,
         dreem_learning = `DREEM learning`,
         dreem_teachers = `DREEM teachers`,
         dreem_academic_self_perception = `DREEM academic self-perception`,
         dreem_atmosphere = `DREEM atmosphere`,
         dreem_social_self_perception = `DREEM social self-perception`,
         dreem_global = `DREEM global score`,
         resilience_score = `Resilience score`,
         bdi = BDI, age_yrs = Age,
         school_legal_status = `School legal status`,
         school_location = `School location`,
         state_anxiety = `State Anxiety`,
         trait_anxiety = `Trait anxiety`) |>
  mutate(across(c(overall_qol, medical_school_qol,
                  whoqol_physical, whoqol_psychological,
                  whoqol_social, whoqol_environment,
                  dreem_learning, dreem_teachers,
                  dreem_academic_self_perception, dreem_atmosphere,
                  dreem_social_self_perception, dreem_global,
                  resilience_score, bdi, age_yrs,
                  state_anxiety, trait_anxiety),
                ~ suppressWarnings(as.numeric(.x)))) |>
  mutate(sex = tolower(as.character(sex)),
         group = tolower(gsub("[^[:alnum:]]+", "_", as.character(group))))
write_clean(msq, "16_med_student_qol")
make_thumb(filter(msq, !is.na(resilience_score), !is.na(whoqol_psychological)),
           "resilience_score", "whoqol_psychological", "16_med_student_qol",
           "WHOQOL psychological vs resilience score")
add_summary(16, "16_med_student_qol", msq,
            outcome = "whoqol_psychological",
            main_predictor = "resilience_score",
            license = "CC0 1.0 (Dryad)",
            load_code = 'read.csv("datasets_staging/cleaned/16_med_student_qol.csv")',
            source_name = "Tempski 2015 (PLOS ONE) via Dryad")

# --------------------------------------------------------------------------- #
# Row 17 — Depression & anxiety in older adults (Dryad, CC0). n=5331.         #
# Many categorical variables encoded 1..k; we decode per README.              #
# --------------------------------------------------------------------------- #
message("[17] depression_anxiety")
dep <- read_csv(file.path(raw, "17_depression_anxiety",
                          "Mental_Health_Survey_of_the_Elderly.csv"),
                show_col_types = FALSE) |>
  rename(subject_code = code,
         education_level = educationcat,
         marital_status_code = marriagecat,
         has_chronic_disease_code = chronicdiseases,
         income_level = Monthlypersonalincome,
         drinking_code = drinking,
         smoking_code = Smoking,
         self_rated_health = Healthstatus,
         phq9_depression_score = PHQ9score,
         gad7_anxiety_score = GAD7score,
         isi_insomnia_score = ISIscore,
         sleep_hours = Sleepduration,
         ad8_cognitive_score = AD8score,
         csid_cognitive_score = CSIDscore,
         uls_loneliness_score = ULSscore,
         depressive_symptoms = Depressivesymptoms,
         anxiety_symptoms = Anxietysymptoms,
         mild_cognitive_impairment = Mildcognitiveimpairment,
         early_dementia = Earlydementia,
         insomnia = Insomnia) |>
  mutate(
    education_level = factor(education_level, levels = 1:5,
                             labels = c("primary_or_below", "junior_high",
                                        "high_school", "college",
                                        "master_plus")),
    marital_status = factor(marital_status_code, levels = 1:2,
                            labels = c("unmarried_div_widowed", "married")),
    has_chronic_disease = has_chronic_disease_code == 2,
    drinking_status = factor(drinking_code, levels = 1:3,
                             labels = c("non_drinker", "ex_drinker",
                                        "current_drinker")),
    smoking_status = factor(smoking_code, levels = 1:3,
                            labels = c("non_smoker", "ex_smoker",
                                       "current_smoker")),
    self_rated_health = factor(self_rated_health, levels = 1:5,
                               labels = c("good", "relatively_good", "ordinary",
                                          "relatively_poor", "poor")),
    depressive_symptoms = as.logical(depressive_symptoms),
    anxiety_symptoms = as.logical(anxiety_symptoms),
    mild_cognitive_impairment = as.logical(mild_cognitive_impairment),
    early_dementia = as.logical(early_dementia),
    insomnia = as.logical(insomnia)
  ) |>
  select(subject_code, education_level, marital_status, has_chronic_disease,
         income_level, drinking_status, smoking_status, self_rated_health,
         sleep_hours,
         phq9_depression_score, gad7_anxiety_score, isi_insomnia_score,
         ad8_cognitive_score, csid_cognitive_score, uls_loneliness_score,
         depressive_symptoms, anxiety_symptoms,
         mild_cognitive_impairment, early_dementia, insomnia,
         marital_status_code, has_chronic_disease_code,
         drinking_code, smoking_code)
write_clean(dep, "17_depression_anxiety")
make_thumb(dep, "uls_loneliness_score", "phq9_depression_score",
           "17_depression_anxiety",
           "PHQ-9 depression score vs UCLA loneliness score")
add_summary(17, "17_depression_anxiety", dep,
            outcome = "phq9_depression_score",
            main_predictor = "uls_loneliness_score",
            license = "CC0 1.0 (Dryad)",
            load_code = 'read.csv("datasets_staging/cleaned/17_depression_anxiety.csv")',
            source_name = "Shenzhen mental-health survey 2024 (BMJ Open) via Dryad")

# --------------------------------------------------------------------------- #
# Row 18 — PREVEND sample (oibiostat::prevend.samp). n=500.                   #
# Dutch cohort: predict RFFT cognitive-function score.                        #
# --------------------------------------------------------------------------- #
message("[18] prevend")
data("prevend.samp", package = "oibiostat")
prevend <- as_tibble(prevend.samp) |>
  rename(subject_id = Casenr, age_yrs = Age, gender_code = Gender,
         ethnicity_code = Ethnicity, education_code = Education,
         rfft = RFFT, vat = VAT, cvd = CVD, diabetes = DM,
         smoking_code = Smoking, hypertension = Hypertension, bmi = BMI,
         systolic_bp = SBP, diastolic_bp = DBP, mean_arterial_pressure = MAP,
         egfr = eGFR, albuminuria_v1 = Albuminuria.1,
         albuminuria_v2 = Albuminuria.2, total_cholesterol = Chol,
         hdl_cholesterol = HDL, statin_user = Statin,
         statin_solubility_code = Solubility, days_on_statin = Days,
         years_on_statin = Years, defined_daily_dose = DDD,
         framingham_risk_score = FRS, propensity_score = PS,
         propensity_quintile = PSquint, genetic_risk_score = GRS,
         match_id_1 = Match_1, match_id_2 = Match_2) |>
  mutate(sex = factor(gender_code, levels = c(0, 1),
                      labels = c("male", "female")),
         ethnicity = factor(ethnicity_code, levels = c(0, 1, 2, 3),
                            labels = c("western_european", "black",
                                       "asian", "other")),
         education = factor(education_code, levels = c(0, 1, 2, 3),
                            labels = c("primary", "lower_secondary",
                                       "higher_secondary", "university")),
         smoking_status = factor(smoking_code, levels = c(0, 1, 2),
                                 labels = c("never", "former", "current")),
         cvd = as.logical(cvd),
         diabetes = as.logical(diabetes),
         hypertension = as.logical(hypertension),
         statin_user = as.logical(statin_user)) |>
  select(subject_id, sex, age_yrs, ethnicity, education, rfft,
         bmi, systolic_bp, diastolic_bp, mean_arterial_pressure, egfr,
         total_cholesterol, hdl_cholesterol, cvd, diabetes, hypertension,
         smoking_status, statin_user, framingham_risk_score,
         everything(),
         -gender_code, -ethnicity_code, -education_code, -smoking_code)
write_clean(prevend, "18_prevend")
make_thumb(prevend, "age_yrs", "rfft", "18_prevend",
           "RFFT cognitive score vs age")
add_summary(18, "18_prevend", prevend,
            outcome = "rfft", main_predictor = "age_yrs",
            license = "No explicit licence (oibiostat pkg); PREVEND data usage governed by UMC Groningen",
            load_code = 'data(prevend.samp, package = "oibiostat")',
            source_name = "PREVEND study (Netherlands) via oibiostat")

# --------------------------------------------------------------------------- #
# Row 19 — FAMuSS (oibiostat::famuss). n=595 adults.                          #
# Percent change in non-dominant arm strength after resistance training.      #
# --------------------------------------------------------------------------- #
message("[19] famuss")
data("famuss", package = "oibiostat")
famuss_df <- as_tibble(famuss) |>
  rename(ndrm_change_pct = ndrm.ch, drm_change_pct = drm.ch,
         age_yrs = age, height_in = height, weight_lb = weight,
         actn3_genotype = actn3.r577x) |>
  mutate(sex = factor(tolower(as.character(sex)),
                      levels = c("female", "male")),
         race = factor(as.character(race)))
write_clean(famuss_df, "19_famuss")
make_thumb(famuss_df, "age_yrs", "ndrm_change_pct", "19_famuss",
           "Arm-strength % change vs age")
add_summary(19, "19_famuss", famuss_df,
            outcome = "ndrm_change_pct", main_predictor = "age_yrs",
            license = "No explicit licence (oibiostat pkg); FAMuSS original study Thompson et al. 2004",
            load_code = 'data(famuss, package = "oibiostat")',
            source_name = "FAMuSS study via oibiostat")

# --------------------------------------------------------------------------- #
# Row 20 — kidiq (rosdata::kidiq). n=434.                                     #
# Child test score on mother's IQ + whether mother finished high school.      #
# --------------------------------------------------------------------------- #
message("[20] kidiq")
data("kidiq", package = "rosdata")
kidiq_df <- as_tibble(kidiq) |>
  rename(mom_age_yrs = mom_age) |>
  mutate(mom_high_school = as.logical(mom_hs)) |>
  select(kid_score, mom_iq, mom_high_school, mom_work, mom_age_yrs)
write_clean(kidiq_df, "20_kidiq")
make_thumb(kidiq_df, "mom_iq", "kid_score", "20_kidiq",
           "Child test score vs mother's IQ")
add_summary(20, "20_kidiq", kidiq_df,
            outcome = "kid_score", main_predictor = "mom_iq",
            license = "rosdata pkg: BSD-3 code, data licence unstated; underlying NLSY data public",
            load_code = 'data(kidiq, package = "rosdata")',
            source_name = "Gelman/Hill/Vehtari Regression and Other Stories")

# --------------------------------------------------------------------------- #
# Row 21 — births14 (openintro::births14). n=1000 US NCHS 2014 births.        #
# --------------------------------------------------------------------------- #
message("[21] births14")
data("births14", package = "openintro")
births14_df <- as_tibble(births14) |>
  rename(father_age_yrs = fage, mother_age_yrs = mage,
         gestation_weeks = weeks, prenatal_visits = visits,
         weight_gain_lb = gained, birthweight_lb = weight) |>
  mutate(mother_maturity = factor(gsub(" ", "_", mature)),
         premature = premie == "premie",
         low_birthweight = lowbirthweight == "low",
         sex = factor(sex, levels = c("female", "male")),
         smoking_habit = factor(habit),
         marital_status = factor(gsub(" ", "_", marital)),
         white_mother = whitemom == "white") |>
  select(father_age_yrs, mother_age_yrs, mother_maturity, gestation_weeks,
         premature, prenatal_visits, weight_gain_lb, birthweight_lb,
         low_birthweight, sex, smoking_habit, marital_status, white_mother)
write_clean(births14_df, "21_births14")
make_thumb(filter(births14_df, !is.na(gestation_weeks), !is.na(birthweight_lb)),
           "gestation_weeks", "birthweight_lb", "21_births14",
           "Birth weight (lb) vs gestational age (weeks)")
add_summary(21, "21_births14", births14_df,
            outcome = "birthweight_lb", main_predictor = "gestation_weeks",
            license = "CC-BY-SA 4.0 (openintro)",
            load_code = 'data(births14, package = "openintro")',
            source_name = "US NCHS 2014 via openintro")

# --------------------------------------------------------------------------- #
# Row 22 — UN member states 2024 (moderndive 0.7+). n=193 countries.          #
# Loaded from an rda copied from the moderndive GitHub repo.                  #
# --------------------------------------------------------------------------- #
message("[22] un_member_states_2024")
un_env <- new.env()
load(file.path(raw, "18_un_member_states", "un_member_states_2024.rda"),
     envir = un_env)
un_df <- as_tibble(un_env$un_member_states_2024) |>
  rename(iso_code = iso,
         income_group = income_group_2024,
         population = population_2024,
         area_km2 = area_in_square_km,
         gdp_per_capita_usd = gdp_per_capita,
         obesity_rate_2016_pct = obesity_rate_2016,
         obesity_rate_2024_pct = obesity_rate_2024,
         life_expectancy_yrs = life_expectancy_2022,
         fertility_rate = fertility_rate_2022,
         hdi = hdi_2022) |>
  select(country, iso_code, continent, region, income_group,
         population, area_km2, gdp_per_capita_usd,
         obesity_rate_2016_pct, obesity_rate_2024_pct,
         life_expectancy_yrs, fertility_rate, hdi)
write_clean(un_df, "22_un_member_states_2024")
make_thumb(filter(un_df, !is.na(hdi), !is.na(life_expectancy_yrs)),
           "hdi", "life_expectancy_yrs", "22_un_member_states_2024",
           "Life expectancy (yrs) vs HDI, 193 countries")
add_summary(22, "22_un_member_states_2024", un_df,
            outcome = "life_expectancy_yrs", main_predictor = "hdi",
            license = "MIT (redistributed via moderndive pkg)",
            load_code = 'data(un_member_states_2024, package = "moderndive")',
            source_name = "UN/World Bank via moderndive")

# --------------------------------------------------------------------------- #
# Row 23 — NHANES adult sample (oibiostat::nhanes.samp.adult). n=135.         #
# US National Health and Nutrition Examination Survey (2009-12).              #
# --------------------------------------------------------------------------- #
message("[23] nhanes_adult")
data("nhanes.samp.adult", package = "oibiostat")
nhanes_df <- as_tibble(nhanes.samp.adult) |>
  rename(subject_id = ID, survey_year = SurveyYr, sex = Gender,
         age_yrs = Age, age_decade = AgeDecade, race = Race1,
         education_level = Education, marital_status = MaritalStatus,
         hh_income_midpoint = HHIncomeMid, poverty_ratio = Poverty,
         home_ownership = HomeOwn, work_status = Work,
         weight_kg = Weight, height_cm = Height, bmi = BMI,
         pulse = Pulse, systolic_bp = BPSysAve, diastolic_bp = BPDiaAve,
         direct_hdl = DirectChol, total_cholesterol = TotChol,
         diabetes = Diabetes, general_health = HealthGen,
         phys_health_bad_days = DaysPhysHlthBad,
         ment_health_bad_days = DaysMentHlthBad,
         sleep_hours = SleepHrsNight, sleep_trouble = SleepTrouble,
         phys_active = PhysActive, phys_active_days = PhysActiveDays,
         alcohol_days_year = AlcoholYear, smoked_100 = Smoke100,
         smoke_now = SmokeNow) |>
  mutate(diabetes = diabetes == "Yes",
         sleep_trouble = sleep_trouble == "Yes",
         phys_active = phys_active == "Yes",
         smoked_100 = smoked_100 == "Yes") |>
  select(subject_id, survey_year, sex, age_yrs, age_decade, race,
         education_level, marital_status, hh_income_midpoint, poverty_ratio,
         home_ownership, work_status, weight_kg, height_cm, bmi, pulse,
         systolic_bp, diastolic_bp, direct_hdl, total_cholesterol,
         diabetes, general_health, phys_health_bad_days, ment_health_bad_days,
         sleep_hours, sleep_trouble, phys_active, phys_active_days,
         alcohol_days_year, smoked_100, smoke_now)
write_clean(nhanes_df, "23_nhanes_adult")
make_thumb(filter(nhanes_df, !is.na(age_yrs), !is.na(systolic_bp)),
           "age_yrs", "systolic_bp", "23_nhanes_adult",
           "Systolic BP vs age (NHANES adults)")
add_summary(23, "23_nhanes_adult", nhanes_df,
            outcome = "systolic_bp", main_predictor = "age_yrs",
            license = "Underlying NHANES is US-government public domain; oibiostat sample has no explicit licence",
            load_code = 'data(nhanes.samp.adult, package = "oibiostat")',
            source_name = "US NHANES 2009-12 sample via oibiostat")

# --------------------------------------------------------------------------- #
# Row 24 — US county-level (usdata::county). n=3142 counties.                 #
# Ecological social-determinants-of-health data.                              #
# --------------------------------------------------------------------------- #
message("[24] county")
data("county", package = "usdata")
county_df <- as_tibble(county) |>
  rename(county_name = name,
         population_2000 = pop2000,
         population_2010 = pop2010,
         population_2017 = pop2017,
         population_change_pct = pop_change,
         poverty_rate = poverty,
         homeownership_rate = homeownership,
         multi_unit_housing_pct = multi_unit)
write_clean(county_df, "24_county")
make_thumb(filter(county_df, !is.na(poverty_rate), !is.na(unemployment_rate)),
           "poverty_rate", "unemployment_rate", "24_county",
           "Unemployment rate vs poverty rate (US counties)")
add_summary(24, "24_county", county_df,
            outcome = "unemployment_rate", main_predictor = "poverty_rate",
            license = "GPL-3 (usdata pkg); underlying US Census/ACS is public domain",
            load_code = 'data(county, package = "usdata")',
            source_name = "US Census/ACS via usdata")

# --------------------------------------------------------------------------- #
# Row 25 — Boston housing (ISLR2::Boston). n=506 census tracts.               #
# Environmental-health framing: nitric-oxide concentration as outcome.        #
# --------------------------------------------------------------------------- #
message("[25] boston")
data("Boston", package = "ISLR2")
boston_df <- as_tibble(Boston) |>
  rename(per_capita_crime_rate = crim,
         large_lot_residential_pct = zn,
         non_retail_business_pct = indus,
         nox_ppm_x10m = nox,
         avg_rooms_per_dwelling = rm,
         pct_built_before_1940 = age,
         distance_to_employment = dis,
         highway_access_idx = rad,
         property_tax_per_10k = tax,
         pupil_teacher_ratio = ptratio,
         lower_status_pct = lstat,
         median_home_value_usd_1k = medv) |>
  mutate(near_charles_river = as.logical(chas)) |>
  select(per_capita_crime_rate, large_lot_residential_pct,
         non_retail_business_pct, near_charles_river, nox_ppm_x10m,
         avg_rooms_per_dwelling, pct_built_before_1940,
         distance_to_employment, highway_access_idx, property_tax_per_10k,
         pupil_teacher_ratio, lower_status_pct, median_home_value_usd_1k)
write_clean(boston_df, "25_boston")
make_thumb(boston_df, "distance_to_employment", "nox_ppm_x10m",
           "25_boston",
           "NOx air pollution vs distance to employment centres")
add_summary(25, "25_boston", boston_df,
            outcome = "nox_ppm_x10m", main_predictor = "distance_to_employment",
            license = "GPL-2 (redistributed via ISLR2 pkg)",
            load_code = 'data(Boston, package = "ISLR2")',
            source_name = "Boston housing (Harrison & Rubinfeld 1978) via ISLR2")

summary_df <- bind_rows(summary_rows) |> arrange(row_id)
write_csv(summary_df, file.path(cleaned, "_summary.csv"))
message("\nWrote ", file.path(cleaned, "_summary.csv"))
print(summary_df)
