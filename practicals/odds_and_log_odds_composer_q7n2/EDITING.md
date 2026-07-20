# Editing this practical

**Odds, log-odds, and odds ratios** (`composer_q7n2`). Quarto WebR R-practical.

## Placement in the course

Sits **after** both:

- `slide_decks/intro_to_logistic_regression`
- `slide_decks/logistic_regression_coefficient_interpretation`

## Format

- Source: `index.qmd`
- WebR extension: `_extensions/coatless/webr/` (copied from `intro_to_logistic_regression`)
- Render: `quarto render index.qmd` (from this folder)

## Teaching arc

1. Four scales recap (probability, odds, log-odds, OR) with R conversion table.
2. Refit KLoSA `weak_grip ~ age`; read log-odds from `summary()`.
3. `broom::tidy(..., exponentiate = TRUE, conf.int = TRUE)` and hand check with `exp(coef())`.
4. Paper phrasing table + example sentence.
5. Categorical `age_group` model; reference-group OR interpretation.
6. Seven practice questions + completion code `EXPORAT`.

## vs HTML version (`composer_m4k9`)

This version is R-first: fewer interactive SVG widgets, more live WebR cells. The HTML copy keeps the full Brilliant-style probability/odds/log walkthrough.

## Packages (webr YAML)

`broom`, `dplyr`

## Authoring

Ignore other suffixed folders when editing.
