#!/usr/bin/env Rscript
# Build a small KLoSA Wave 1 teaching extract for logistic regression.
#
# Source: datasets_staging/raw/klosa_wave1/Lt01_e.dta
#   (KEIS KLoSA Wave 1 longitudinal tracking file; see raw/klosa_wave1/README.md)
#
# Filter matches slide_decks/intro_to_logistic_regression/build_figures.R:
#   men, age >= 45, measured grip > 0
# Binary outcome: weak_grip = grip <= 27 kg (EWGSOP2 cut for men)
#
# Extra covariates kept small for GitHub / classroom use.

suppressPackageStartupMessages({
  library(haven)
  library(dplyr)
  library(readr)
  library(ggplot2)
  library(tibble)
})

here <- normalizePath(dirname({
  m <- regmatches(commandArgs(trailingOnly = FALSE),
                  regexpr("--file=.*", commandArgs(trailingOnly = FALSE)))
  if (length(m)) sub("^--file=", "", m) else "datasets_staging/scripts/prepare_klosa.R"
}))
root <- normalizePath(file.path(here, ".."))
raw_path <- file.path(root, "raw", "klosa_wave1", "Lt01_e.dta")
cleaned <- file.path(root, "cleaned")
thumbs <- file.path(root, "thumbnails")
dir.create(cleaned, showWarnings = FALSE, recursive = TRUE)
dir.create(thumbs, showWarnings = FALSE, recursive = TRUE)

if (!file.exists(raw_path)) {
  stop("Missing ", raw_path, "\nSee datasets_staging/raw/klosa_wave1/README.md")
}

raw <- read_dta(raw_path)

edu_lab <- c(
  "1" = "elementary_or_less",
  "2" = "middle_school",
  "3" = "high_school",
  "4" = "college_or_more"
)
smoke_lab <- c("0" = "never", "1" = "former", "2" = "current")

klosa <- raw |>
  transmute(
    age = as.numeric(w01A002_age),
    grip = as.numeric(w01mgrip),
    sex = as.numeric(w01gender1),
    education_code = as.numeric(w01edu),
    smoking_code = as.numeric(w01smoke),
    bmi = as.numeric(w01bmi),
    hypertension_raw = as.numeric(w01chronic_a),
    diabetes_raw = as.numeric(w01chronic_b),
    marital_raw = as.numeric(w01marital),
    region2 = as.numeric(w01region2)
  ) |>
  filter(
    sex == 1,
    !is.na(age), age >= 45,
    !is.na(grip), grip > 0
  ) |>
  mutate(
    weak_grip = as.integer(grip <= 27),
    education = factor(
      edu_lab[as.character(education_code)],
      levels = c("elementary_or_less", "middle_school", "high_school", "college_or_more")
    ),
    smoking = factor(
      smoke_lab[as.character(smoking_code)],
      levels = c("never", "former", "current")
    ),
    # KLoSA chronic flags: 1 = yes, 5 = no
    hypertension = case_when(
      hypertension_raw == 1 ~ 1L,
      hypertension_raw == 5 ~ 0L,
      TRUE ~ NA_integer_
    ),
    diabetes = case_when(
      diabetes_raw == 1 ~ 1L,
      diabetes_raw == 5 ~ 0L,
      TRUE ~ NA_integer_
    ),
    married = case_when(
      marital_raw == 1 ~ 1L,
      marital_raw %in% c(2, 3, 4, 5) ~ 0L,
      TRUE ~ NA_integer_
    ),
    # region2: 1 = urban/metro-leaning, 2 = town/rural-leaning (KEIS coding)
    urban = case_when(
      region2 == 1 ~ 1L,
      region2 == 2 ~ 0L,
      TRUE ~ NA_integer_
    ),
    bmi = if_else(bmi > 0 & bmi < 80, round(bmi, 1), NA_real_)
  ) |>
  select(
    age, grip, weak_grip,
    education, smoking, bmi,
    hypertension, diabetes, married, urban
  )

stopifnot(nrow(klosa) == 4184L)

out_csv <- file.path(cleaned, "28_klosa_men_grip.csv")
write_csv(klosa, out_csv)

# Keep the course data/ folder in sync (CSV + zip for easy download)
data_dir <- normalizePath(file.path(root, "..", "data"), mustWork = FALSE)
if (dir.exists(data_dir)) {
  data_csv <- file.path(data_dir, "28_klosa_men_grip.csv")
  data_zip <- file.path(data_dir, "28_klosa_men_grip.zip")
  file.copy(out_csv, data_csv, overwrite = TRUE)
  owd <- setwd(data_dir)
  on.exit(setwd(owd), add = TRUE)
  if (file.exists(data_zip)) file.remove(data_zip)
  utils::zip("28_klosa_men_grip.zip", "28_klosa_men_grip.csv")
}

# Thumbnail: observed weak-grip rate by age band (logistic story)
bands <- klosa |>
  mutate(age_band = cut(age, breaks = c(seq(45, 85, 5), Inf),
                        right = FALSE, include.lowest = TRUE)) |>
  group_by(age_band) |>
  summarise(p_weak = mean(weak_grip), n = n(), .groups = "drop")

p <- ggplot(bands, aes(age_band, p_weak)) +
  geom_col(fill = "#2c7fb8", width = 0.75) +
  scale_y_continuous(labels = function(x) paste0(round(100 * x), "%"),
                     limits = c(0, NA)) +
  labs(x = "Age band", y = "% weak grip (<=27 kg)",
       subtitle = "KLoSA Wave 1 men 45+ | n = 4,184") +
  theme_minimal(base_size = 9) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1),
        panel.grid.minor = element_blank(),
        plot.subtitle = element_text(size = 7, colour = "grey40"))
ggsave(file.path(thumbs, "28_klosa_men_grip.png"), p,
       width = 3.2, height = 2.2, dpi = 140, bg = "white")

message("Wrote ", out_csv)
message("n = ", nrow(klosa),
        " | % weak = ", round(100 * mean(klosa$weak_grip), 1),
        " | complete cases (all cols) = ",
        sum(complete.cases(klosa)))
str(klosa, give.attr = FALSE)
