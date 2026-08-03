# Final project datasets: full catalogue

Fifteen open health datasets for a multiple linear regression or multivariable
logistic regression final project. Every one is linked to a research paper.

This is the wide-comparison-table presentation from
`final_project_datasets_gpt-5.6-sol-c8r4`, repopulated with the full dataset pool
that was assembled in `final_project_datasets_fable5_d3n8`.

## Missing-data policy

Nothing is dropped for being incomplete. Any variable more than 10% missing is
marked `(x% missing)` beside its name in the index, and exact completeness for
every variable is in that dataset's codebook. The page explains why this matters:
`lm()` and `glm()` silently drop rows missing *any* model variable, so one
incomplete predictor can cost a large share of the sample without warning.

Incomplete variables are retained and marked, including the HIV cohort's viral-load
and CD4 measures (29–34%) and Kenya's `monthly_income` (31%).

## Layout

```
index.html                      the catalogue table
datasets/<slug>.html            one page per dataset (codebook, preview, distributions)
data/<slug>.csv                 the dataset
codebooks/<slug>_codebook.csv   variable, description, type, values, % complete
figures/<slug>_inspect_*.png    inspect_cat() / inspect_num() plots
analyses/<slug>/<outcome>.Rmd   worked R Markdown analysis per offered outcome
build/                          the generator
scripts/                        figure regeneration
```

## Rebuilding

```
Rscript build/prepare_data.R            # rebuild CSVs from the source deposits
python3 build/build_index.py            # index.html + detail pages + detail.js
Rscript scripts/generate_inspect_figures.R
```

`build/build_index.py` recomputes row counts, column counts and missingness from
the shipped CSVs on every run, and asserts that every variable it names actually
exists, so the page cannot drift from the data. Editorial content (blurbs,
suggested outcomes and predictors) lives in `build/meta.json`.

## Differences from the source builds

- **Nepal, Peru and Bangladesh** come from the corrected `c8r4` catalogue, not from
  `fable5_d3n8`: Peru's haemoglobin column is named `haemoglobin_g_dl` and its
  low-birth-weight recipe derives from the existing `low` category rather than a
  non-existent kilogram threshold; Bangladesh's `sub_district` is a labelled factor
  rather than integers 1–6; Nepal no longer ships `waist_cm` / `waist_raised`.
- **Peru** excludes `min_dietary_diversity` because it was asked only of the
  complementary-feeding subsample and is 36% missing.
- **Serbia antibiotics, workplace protection and primary-care sciatica** were
  added after checking their source deposits, outcome balance and usable predictors.
  Serbia's adequate-knowledge flag is rebuilt from the published score threshold
  because the deposited category conflicts with the paper.
- **Abuja household malaria** was added from the paper's deposited survey. Its
  microscopy outcome is complete, with 421 positive and 181 negative results.
- **European ageing and quality of life** uses the 5,341 rows in the supplied
  supporting file. The paper reports a different analytic sample, but the deposited
  data remain suitable for the teaching analyses offered here.
- **ESSENS adolescents** converts the source file's `100` missing-value sentinel
  to blank values and keeps the screen-time measures as categorical variables.
- **Sarcopenia in older adults in India** is not included. Its problem was not
  missingness: at n = 240 with 42 in the rarer outcome class it cannot carry three
  predictors at ten events per parameter, and all three of its binary outcomes are
  dichotomised copies of continuous columns in the same file.
- **NHANES adults, US births 2014 and FAMuSS** were withdrawn after their worked
  analyses were fitted. FAMuSS ships 595 rows that are an upstream `na.omit()` of
  1,397, so every column reads as 0% missing while 57% of the study has already been
  deleted; it also has no control group, since everybody trained. NHANES and US
  births were the catalogue's only two datasets without a linked source paper.
- **`abi_max` is withdrawn as an outcome** for the Japanese health-check cohort,
  though the column still ships. It has no clinical variance: 0 of 876 participants
  fall below the 0.90 PAD threshold and 94.7% sit in the normal 1.00–1.30 band. The
  reason is recorded in `EXCLUDED_OUTCOMES` in `build/build_index.py`.
