# Editing this practical

**Odds, log-odds, and odds ratios, a hands-on consolidation.** A Brilliant.org-style interactive tutorial that
sits **after** `slide_decks/intro_to_logistic_regression` and
`slide_decks/logistic_regression_coefficient_interpretation`. It is *not* WebR/Quarto. It is a single
self-contained `index.html` (inline CSS + JS, KaTeX + Google Fonts from CDN). Light mode only.

Reuses the deck design system (teal `#035f6c` / gold `#e5b44f` / mist / coral palette,
Space Grotesk + IBM Plex Mono).

## Scope decision (important)

The first seven parts remain mostly generic and use simple 10-person examples. Parts 8 and 9 use
the KLOSA age coefficient already introduced in the prerequisite lessons. They add odds-ratio
interpretation, exponentiated `broom::tidy()` output, and reporting examples without turning the
practical into another model-fitting lesson.

## How it works

- Content is a vertical list of `<section class="step">`. **One step is revealed at a time.**
  The reveal engine (`showStep` / `advance`) adds `.show`, builds that step's widgets, scrolls to it.
- Every step ends with a `.gate` `.continue` button. On a **concept** step it is live at once. On a
  **check** step it starts `.locked` (hidden); solving the check calls `solve()` →
  `unlockContinue()`. So a learner cannot skip a check, but every check has a hint whose last line
  is a "Show the answer / Set it for me" fallback (`.reveal`).
- Top progress bar + `#partlabel` come from each step's `data-part` (1-9; `TOTAL_PARTS = 9`).

## Teaching arc (9 parts)

1. Probability = a share of a group (dot grid).
2. Odds = happens vs does-not (stacks); count *or* probability phrasing; odds below 1 and the flip; odds = 1 is 50-50.
3. Probability ↔ odds slider; then the odds scale is not balanced (stops at 0, no top limit).
4. What is a log: a **base-10 log ruler** (0.1, 1, 10, 100 ↔ log −1,0,1,2), with a note that
   log-odds uses base e; then the sign of a log.
5. Log-odds defined; the **transformation ladder** (probability line ↔ log-odds line, dotted
   links tagged with the odds, seven rungs 0.05…0.95 incl. 0.95).
6. **Reading log-odds from waffle charts** (guess sign / rough size from a 10-square waffle).
7. Probability and log-odds together (linked slider); log-odds checks.
8. Odds ratios as comparisons, exponentiating a coefficient, interpreting values above and below
   1, and changing the predictor interval.
9. Exponentiated `broom::tidy()` output, paper-style tables and prose, confidence intervals,
   categorical reference groups, and summary.

## Checks (`class="card check"` and `data-type`)

- `mcq`: `.opts > .opt`; mark the right one `data-correct`. Keep to **2** options.
- `mcq` + `.opts.pick`: each `.opt` holds a mini SVG (`data-mini="1"` stacks).
- `numeric`: `data-answer` + `data-tol`; `input[type=number]` + `.numcheck`.
- `slider-numeric`: `data-target-p` + `data-p-tol` + `data-answer` + `data-tol`; the learner sets
  the slider near a probability, then types the odds. Both must be right to pass.
- `slider`: `data-target="logneg"` (log-odds < 0); `.lab` slider + `.numcheck`.

Each check carries `.hint-wrap` (`.hintbtn` + `.hintbox` + `.reveal`).

## Widgets (built in JS from placeholder divs / ids)

`.dotgrid` `buildDotGrid` · `.waffle` `buildWaffle` (grid of squares, coloured = event) ·
`.stacks` `buildStacks` (data-happen/data-not/data-mini; columns labelled "happened"/"did not") ·
`#squishA` `buildSquish` (odds are lopsided) · `#logruler` `buildLogRuler` (base-10 ruler) ·
`#transform` `buildTransform` (transformation ladder) · `.lab[data-slider]` `initSlider`
(`data-slider="ladder"` also draws the linked `.ladder-svg`). `LADDER` (top of the script) holds
the round p → odds → log-odds reference values used by `buildTransform`.

## Language

Written for English-language learners: short sentences, one idea per line, **no em dashes**, no
figurative or catchy phrasing. Keep it that way by searching for the Unicode em-dash character.

## Authoring / verifying

- `index.html?all=1` reveals every step at once (unlocks all continues) for screenshots.
- Full-page screenshot: headless Chrome `--window-size=840,17000 --screenshot=… "…?all=1"`.
- When updating the original flow test, account for 47 steps and the new Part 8 and Part 9 checks.

The folder suffix identifies this version. Do not use another suffixed folder as an editing source.
