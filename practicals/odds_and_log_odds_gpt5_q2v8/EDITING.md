# Editing this practical

**Odds, log-odds, and odds ratios.** This is a Quarto WebR practical that follows both
`slide_decks/intro_to_logistic_regression` and
`slide_decks/logistic_regression_coefficient_interpretation`.

Edit `index.qmd`, then render it with Quarto. The local `_extensions/coatless/webr` directory keeps
the WebR filter available when this folder is rendered on its own. The rendered `index.html` is a
build output and should not be edited directly.

## Teaching arc (6 parts)

1. Probability and odds, with live conversion code.
2. Odds and log-odds, including `log()` and `exp()`.
3. Exponentiating coefficients and changing the comparison interval.
4. Fitting the KLOSA model and using `broom::tidy(..., exponentiate = TRUE, conf.int = TRUE)`.
5. Paper-style tables, prose, confidence intervals, and reference information.
6. Categorical predictors and reference groups.

There are 10 scored interpretation questions. Eight correct answers reveal the completion code.

## How it works

- WebR cells share one browser-side R session. The Part 4 setup cell creates `klosa` and
  `grip_model`, which later cells use.
- The static multiple-choice cards use the inline script at the end of `index.qmd`.
- Ten scored questions are tracked by CSS classes. Eight correct answers reveal the completion code.
- Fixed output blocks show representative output even before WebR has started.

## Language and verification

Use short sentences and do not use em dashes. Render with `quarto render index.qmd`. Check that the
WebR editors appear, the setup cell runs before dependent cells, question feedback works, and the
layout remains readable on a narrow screen.

The folder suffix identifies this version. Do not use another suffixed folder as an editing source.
