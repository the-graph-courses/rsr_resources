# Editing this practical

**Odds, log-odds, and odds ratios (Quarto + WebR).** Sits **after** both
`slide_decks/intro_to_logistic_regression` and
`slide_decks/logistic_regression_coefficient_interpretation`.

Folder suffix `grok45_q3n7` marks this attempt. Do not look at other models' suffixed folders.

## Stack

- Source: `index.qmd`
- Engine: knitr + `coatless/webr` filter (local `_extensions/`)
- Packages loaded in WebR: `broom`, `dplyr`
- Interactive MCQs with completion code `ODDSRATIO` (need 9 of 11)

## Teaching arc

1. Probability / odds / log-odds review with a live conversion cell
2. Why the model uses log-odds, and why papers report odds ratios
3. `exp(coefficient)` and OR plain-language table
4. Live `broom::tidy(mod, exponentiate = TRUE, conf.int = TRUE)` on KLoSA
5. Age-group ORs versus reference
6. Paper-style reporting sentences + interpretation practice

## Render

From this folder:

```bash
quarto render index.qmd
```

## Language

Short sentences, one idea per line, **no em dashes**.
