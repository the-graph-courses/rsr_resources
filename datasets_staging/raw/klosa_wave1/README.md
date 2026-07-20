# KLoSA Wave 1 (2006) — local raw source

## What this folder is for

Local staging of the Korean Longitudinal Study of Aging (KLoSA) Wave 1 longitudinal-tracking file used to build the teaching extract `cleaned/28_klosa_men_grip.csv`.

## Source on this machine

Originally downloaded from the KEIS survey portal and stored at:

```
/Users/kendavidn/Dropbox/Mac (2)/Downloads/KLoSA 1-9th wave (STATA)/Lt01_e.dta
```

(also available there as Excel, plus full wave files `w01_e.*` and derived `str01_e.*`)

The slim teaching CSV in the logistic slide decks was built from this file by
`slide_decks/intro_to_logistic_regression/build_figures.R` (filter: men, age ≥ 45, non-missing grip > 0).

## How to re-obtain (if missing)

1. Register at the [KEIS KLoSA download page](https://survey.keis.or.kr/klosa/klosadown/List.jsp)
2. Download **Wave 1–9 Stata** (or Excel)
3. Place `Lt01_e.dta` in this folder
4. Run `datasets_staging/scripts/prepare_klosa.R`

## Redistribution warning

KLoSA microdata is **registration-gated** and KEIS copyright policy prohibits unauthorized redistribution. Do **not** commit `Lt01_e.dta` (or other full wave files) to a public GitHub repo without written permission from KEIS. The `.dta` here is gitignored.

A thin cleaned teaching extract may still be legally sensitive. Prefer: (a) KEIS educational redistribution permission, (b) a download + prep script for students, or (c) synthetic data that mimics the teaching plots.
