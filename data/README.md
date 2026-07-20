# Course datasets

Small teaching extracts used across GRAPH Courses lessons and practicals. Prefer the `.zip` when downloading from GitHub in a browser.

| File | N × p | What it is | Good for |
|------|------:|------------|----------|
| [`yaounde_data.csv`](yaounde_data.csv) ([zip](yaounde_data.zip)) | 971 × 53 | COVID-19 serological survey, Yaoundé, Cameroon (2020) | Descriptive stats, heights/weights, binary IgG/IgM outcomes |
| [`LSTbook_Framingham.csv`](LSTbook_Framingham.csv) | 4,238 × 16 | Framingham Heart Study extract (via `LSTbook`) | Linear regression, interactions (e.g. BMI × BP meds → systolic BP), CHD risk |
| [`28_klosa_men_grip.csv`](28_klosa_men_grip.csv) ([zip](28_klosa_men_grip.zip)) | 4,184 × 10 | KLoSA Wave 1 (2006), Korean men aged 45+ with grip strength | Logistic regression (`weak_grip ~ age`), also continuous grip |

## Quick load (R)

```r
library(readr)

yaounde   <- read_csv("data/yaounde_data.csv")
fram      <- read_csv("data/LSTbook_Framingham.csv")
klosa     <- read_csv("data/28_klosa_men_grip.csv")
```

From a raw GitHub URL (e.g. in Quarto live practicals):

```r
fram <- read.csv(
  "https://raw.githubusercontent.com/the-graph-courses/rsr_resources/main/data/LSTbook_Framingham.csv"
)
```

## Notes

### Yaoundé COVID-19 (`yaounde_data`)

Household serological survey in Yaoundé. Includes demographics, anthropometry, symptoms, treatment flags, and IgG/IgM results. Used heavily in the descriptive-statistics lessons (`here("data/yaounde_data.csv")`).

### Framingham (`LSTbook_Framingham`)

Export of `LSTbook::Framingham`. Columns include age, smoking, BP meds, cholesterol, systolic/diastolic BP, BMI, glucose, sex, and 10-year CHD. Used in the interaction / Framingham slide deck and practical.

### KLoSA men grip (`28_klosa_men_grip`)

Korean Longitudinal Study of Aging, Wave 1. Men aged 45+ with measured grip (kg). `weak_grip` is 1 when grip ≤ 27 kg (EWGSOP2 cut for men). Extra covariates: education, smoking, BMI, hypertension, diabetes, married, urban. Built from `datasets_staging/scripts/prepare_klosa.R`. Same core rows as the slim `klosa_men_grip_45plus.csv` files in the logistic slide decks.
