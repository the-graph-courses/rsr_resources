# Editing this practical

**Odds, log-odds, and odds ratios (HTML).** A Brilliant.org-style interactive tutorial
that sits **after** both `slide_decks/intro_to_logistic_regression` and
`slide_decks/logistic_regression_coefficient_interpretation`. It is *not* WebR/Quarto.
It is a single self-contained `index.html` (inline CSS + JS, KaTeX + Google Fonts from CDN).
Light mode only.

Folder suffix `grok45_h8w4` marks this attempt. Do not look at other models' suffixed folders.

Reuses the deck design system (teal `#035f6c` / gold `#e5b44f` / mist / coral palette,
Space Grotesk + IBM Plex Mono).

## Scope

Parts 1–7 stay mostly generic (probability / odds / logs / log-odds with 10-person
examples). Parts 8–11 use the KLoSA weak-grip × age example the learners already met,
plus static R/`broom::tidy(..., exponentiate = TRUE)` output and paper-style reporting
practice. Odds ratios are the main reporting target.

## How it works

- Content is a vertical list of `<section class="step">`. **One step is revealed at a time.**
- Every step ends with a `.gate` `.continue` button. Check steps start locked until solved.
- Top progress bar + `#partlabel` use `data-part` (1–11; `TOTAL_PARTS = 11`).

## Teaching arc (11 parts)

1. Probability = share of a group.
2. Odds = happens vs does-not.
3. Probability ↔ odds; odds scale is not balanced.
4. What a log is; sign of a log.
5. Log-odds defined; transformation ladder.
6. Reading log-odds from waffle charts.
7. Probability and log-odds together (slider).
8. Bridge: you already saw the logistic model; papers report odds ratios.
9. Odds ratios via `exp(coefficient)`; reading OR above/below 1.
10. Getting ORs in R with `broom::tidy(..., exponentiate = TRUE)` (continuous + age group).
11. Paper reporting sentences; practice interpreting OR / CI / reference group; summary.

## Language

Short sentences, one idea per line, **no em dashes**.
`grep "—" index.html` should be 0.

## Authoring / verifying

- `index.html?all=1` reveals every step at once.
