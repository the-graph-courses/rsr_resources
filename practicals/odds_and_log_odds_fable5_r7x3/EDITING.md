# Editing this practical

**Odds, log-odds, and odds ratios: a hands-on tutorial.** A Brilliant.org-style interactive
tutorial that now sits **after** both `slide_decks/intro_to_logistic_regression` and
`slide_decks/logistic_regression_coefficient_interpretation` (it used to sit between them;
this copy extends it with odds ratios accordingly). It is *not* WebR/Quarto. It is a single
self-contained `index.html` (inline CSS + JS, KaTeX + Google Fonts from CDN). Light mode only.

Reuses the deck design system (teal `#035f6c` / gold `#e5b44f` / mist / coral palette,
Space Grotesk + IBM Plex Mono).

## Scope decision (important)

Parts 1-7 are **mostly generic**: they teach the probability of "an event" using simple
10-person examples (dot grids, waffles, stacks), not one running dataset. Parts 8-10 now
deliberately use the **KLoSA weak-grip model** from the two decks (equation
`log-odds = -8.82 + 0.117 * age`, OR/yr 1.124, age-group ORs 5.22 / 14.4), because the
practical follows those decks and its job is to bridge to `broom::tidy(exponentiate = TRUE)`
output and paper-style reporting. Keep those numbers in sync with the coefficient deck's
`out_glm.txt` / panel 3 (they were verified against `data/28_klosa_men_grip.csv`; the
displayed conf.low of the exponentiated intercept, 7.79e-5, matches real R output).

## How it works

- Content is a vertical list of `<section class="step">`. **One step is revealed at a time.**
  The reveal engine (`showStep` / `advance`) adds `.show`, builds that step's widgets, scrolls to it.
- Every step ends with a `.gate` `.continue` button. On a **concept** step it is live at once. On a
  **check** step it starts `.locked` (hidden); solving the check calls `solve()` →
  `unlockContinue()`. Every check has a hint whose last line is a "Show the answer /
  Set it for me" fallback (`.reveal`).
- Top progress bar + `#partlabel` come from each step's `data-part` (1-10; `TOTAL_PARTS = 10`).

## Teaching arc (10 parts, 42 steps)

The early parts were **shortened by about half** on user request (2026-07-20): concepts merged
into single cards, redundant checks cut, the "Odds are not balanced" squish card and the
tap-the-group pick check removed entirely. Keep this compact shape; do not re-split cards.

1. Probability: one merged concept card (intro + dot-grid share) + one numeric check.
2. Odds: compare-the-two-sides card; merged "below 1 / equal 1" card; mcq 0.25; numeric 6/3.
3. Probability ↔ odds slider concept + slider-numeric check.
4. One merged log card (base-10 log ruler + sign-of-a-log key, plain wording; e introduced
   as "a fixed mathematical constant, approximately 2.718") + mcq log(0.3) negative.
5. One merged log-odds card (definition eq + transformation ladder) + 2 mcqs.
6. Waffle reading: concept + one mcq (3 of 10 → negative).
7. Linked slider concept + slider check (set log-odds negative).
8. Log-odds in logistic regression: grip equation **plus the fitted probability S-curve**
   (`#gripcurve`), mcq line-scale, drug-trial numeric + sign mcq.
9. **Odds ratios**: two-group `.duo` stacks, OR = odds ÷ odds, OR above/equal/below 1,
   OR → plain-words table (`table.orwords`), `e^0.117 = 1.124`, then **the constant-percentage
   property**: annotated fitted curve (`#orcurve`) with decade dots (p 0.05→0.84) and an odds
   row multiplying ×3.2 per decade even as the curve flattens; mcq + ×3.2 numeric;
   exponentiate mcq.
10. **R output and papers**: `glm` + `broom::tidy()` code (`pre.code`) and outputs
    (`pre.out`), `exponentiate = TRUE, conf.int = TRUE`, paper sentence anatomy (`.quote`),
    percent-higher/lower checks, categorical-predictor ORs vs the reference group, an
    OR-below-1 paper sentence, four-row summary.

## Checks (`class="card check" data-type="…"`)

- `mcq` — `.opts > .opt`; mark the right one `data-correct`. Keep to **2** options.
- `mcq` + `.opts.pick` — each `.opt` holds a mini SVG (`data-mini="1"` stacks).
- `numeric` — `data-answer` + `data-tol`; `input[type=number]` + `.numcheck`.
- `slider-numeric` — `data-target-p` + `data-p-tol` + `data-answer` + `data-tol`.
- `slider` — `data-target="logneg"` (log-odds < 0).

## Widgets (built in JS from placeholder divs / ids)

`.dotgrid` `buildDotGrid` · `.waffle` `buildWaffle` · `.stacks` `buildStacks`
(data-happen/data-not/data-mini; also works inside the new `.duo .grp` two-group layout) ·
`#logruler` `buildLogRuler` · `#transform` `buildTransform` ·
`#gripcurve` / `#orcurve` `buildGripCurve(host, annotate)` (the real fitted KLoSA curve from
β₀ = -8.824312, β₁ = 0.116834; annotate=true adds decade dots p 0.05/0.14/0.34/0.63/0.84 and
the odds row 0.05/0.16/0.52/1.69/5.42 with ×3.2 labels; e^(10β₁) = 3.2166) ·
`.lab[data-slider]` `initSlider`. `LADDER` (top of the script) holds the reference values.
(`buildSquish` was removed together with the "Odds are not balanced" card.)
New static elements (no JS): `pre.code`, `pre.out` (with `<b>` highlights), `table.orwords`,
`.quote`.

## Language

Written for English-language learners: plain, complete sentences (the user dislikes clipped
imperative fragments like "Start at 1." and informal asides like "a bit like pi"), one idea
per line, **no em dashes** (`grep "—" index.html` should be 0; en dashes in numeric ranges
like 45–64 are fine), no figurative or catchy phrasing.

## Authoring / verifying

- `index.html?all=1` reveals every step at once (unlocks all continues) for screenshots.
- Flow test: a Playwright script that walks all 42 steps, solving every check type (click
  `.opt[data-correct]` / fill `data-answer` / `_setP()` for sliders, then `.continue`), and
  asserts the finish card and "Part 10 of 10" label.

Note: the folder suffix (`fable5_r7x3`) marks this model's attempt; other models have their
own suffixed folders. Do not look at or edit those.
