# Editing this practical

**Odds, log-odds, and odds ratios** (`composer_m4k9`). Interactive HTML tutorial (Brilliant.org style).

## Placement in the course

Sits **after** both:

- `slide_decks/intro_to_logistic_regression`
- `slide_decks/logistic_regression_coefficient_interpretation`

It is **not** a pre-requisite warm-up anymore. Learners arrive having fit `glm(..., family = binomial)` and seen log-odds coefficients.

## Format

Single self-contained `index.html` (inline CSS + JS, KaTeX + Google Fonts from CDN). Not WebR/Quarto.

## Teaching arc (11 parts)

1. Probability (dot grid).
2. Odds (stacks); below 1; odds = 1.
3. Probability ↔ odds slider; lopsided odds scale.
4. What is a log (base-10 ruler); sign of a log.
5. Log-odds; transformation ladder.
6. Reading log-odds from waffle charts.
7. Probability and log-odds slider ladder.
8. **Bridge**: KLoSA weak grip model on log-odds scale (post-regression framing).
9. **Odds ratios**: exp(log-odds coef); OR table; paper phrasing patterns.
10. **R / broom**: `tidy(..., exponentiate = TRUE, conf.int = TRUE)` with static code + output blocks; categorical reference groups.
11. **Paper interpretation** practice questions + recap checks + summary.

## New sections vs original

- Parts 8–11 replace the old “next lesson” bridge.
- Static `.codeblock` / `.outblock` / `.paper-table` styles for R and reporting examples (no live R in this version).

## Authoring

- `index.html?all=1` reveals all steps for review.
- Ignore other suffixed folders (`opus48_k7m2`, `composer_q7n2`, etc.).
