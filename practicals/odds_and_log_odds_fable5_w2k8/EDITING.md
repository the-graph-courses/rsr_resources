# Editing this practical

**Odds, log-odds, and odds ratios: WebR practical.** A Quarto + quarto-webr conversion of the
`practicals/odds_and_log_odds` gated-reveal tutorial, repositioned to sit **after** both
`slide_decks/intro_to_logistic_regression` and
`slide_decks/logistic_regression_coefficient_interpretation`. The learner runs real R in the
browser: computes odds and log-odds, refits the KLoSA weak-grip model, exponentiates with
`broom::tidy()`, and reads paper-style presentations.

Source is `index.qmd`; render with `quarto render index.qmd` (the folder carries its own copy
of `_extensions/` because quarto resolves the webr filter relative to the input file).

## Structure

- Style/JS copied from `practicals/linear_regression_interactions.qmd` (masthead, `.panel`,
  `.webr-wrap`, `.practice` MCQs with `data-answer`, completion code). New styles here:
  `.paper-quote` (paper sentences) and `.or-table` (OR-to-words table, mock "Table 2").
- 6 webr cells: odds from counts · logs/`qlogis` · fit `glm` + `tidy(mod)` ·
  fill-in-the-blank `tidy(mod, exponentiate = ____, conf.int = TRUE)` · `exp(coef(mod))` ·
  categorical `age_group` model (`cut(age, breaks = c(44, 64, 74, Inf))`).
- 9 MCQs (`p0`–`p8`); completion code `EXPODDS` at 8 of 9 correct.
- Data: `read.csv` from
  `https://raw.githubusercontent.com/the-graph-courses/rsr_resources/main/data/28_klosa_men_grip.csv`
  (same rows as the decks' `klosa_men_grip_45plus.csv`; verified to give the identical model:
  intercept -8.824, age 0.1168, OR 1.124 (1.114–1.135), age-group ORs 5.22 / 14.4).
- webr packages: `dplyr`, `ggplot2`, `broom` (all confirmed present in repo.r-wasm.org).

## Numbers

Keep the quoted outputs in the `<details class="answer">` blocks and Part 6 sentences in sync
with the coefficient deck (`out_glm.txt`, panel 3). The exponentiated intercept conf.low is
`0.0000779` (real R output). The "Current smoking 1.35 (0.98–1.86)" row of the mock Table 2
is **invented for the CI-includes-1 question** and is labeled illustrative in the caption.

## Verifying

E2E test recipe (per repo convention): serve `practicals/` with
`python3 -m http.server 8642`, then run the Playwright script (session scratchpad
`test_webr_copy2.py`): wait for `.qwebr-button-run:not([disabled])`, fix the blank cell via
Monaco (click + Meta+A + type), run all cells in order (cell 4 depends on `mod` from cell 3),
assert the outputs above, answer the MCQs, assert the completion code appears.

Note: the folder suffix (`fable5_w2k8`) marks this model's attempt; other models have their
own suffixed folders. Do not look at or edit those.
