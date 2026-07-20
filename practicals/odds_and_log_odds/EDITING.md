# Editing this practical

**Odds and log-odds, a hands-on warm-up.** A Brilliant.org-style interactive tutorial that
sits **between** `slide_decks/intro_to_logistic_regression` and
`slide_decks/interpreting_age_coefficient_opus48_b3k9`. It is *not* WebR/Quarto. It is a single
self-contained `index.html` (inline CSS + JS, KaTeX + Google Fonts from CDN). Light mode only.

Reuses the deck design system (teal `#035f6c` / gold `#e5b44f` / mist / coral palette,
Space Grotesk + IBM Plex Mono).

## Scope decision (important)

The practical is **mostly generic**: it teaches the probability of "an event" using simple
10-person examples (dot grids and square stacks), not one running dataset. There are only **two
small real touches**, both deliberate: the **drug-trial** odds question near the end, and the
**final bridge equation** `log-odds of weak grip = -8.82 + 0.117 * age`. An earlier version was
built around the KLoSA grip scatterplot + age bands; that was removed (it came from misdirected
feedback meant for a different deck). Do not reintroduce a running real dataset without asking.

## How it works

- Content is a vertical list of `<section class="step">`. **One step is revealed at a time.**
  The reveal engine (`showStep` / `advance`) adds `.show`, builds that step's widgets, scrolls to it.
- Every step ends with a `.gate` `.continue` button. On a **concept** step it is live at once. On a
  **check** step it starts `.locked` (hidden); solving the check calls `solve()` →
  `unlockContinue()`. So a learner cannot skip a check, but every check has a hint whose last line
  is a "Show the answer / Set it for me" fallback (`.reveal`).
- Top progress bar + `#partlabel` come from each step's `data-part` (1-8; `TOTAL_PARTS = 8`).

## Teaching arc (8 parts)

1. Probability = a share of a group (dot grid).
2. Odds = happens vs does-not (stacks); count *or* probability phrasing; odds below 1 and the flip; odds = 1 is 50-50.
3. Probability ↔ odds slider; then the odds scale is not balanced (stops at 0, no top limit).
4. What is a log: a **base-10 log ruler** (0.1, 1, 10, 100 ↔ log −1,0,1,2), with a note that
   log-odds uses base e; then the sign of a log.
5. Log-odds defined; the **transformation ladder** (probability line ↔ log-odds line, dotted
   links tagged with the odds, seven rungs 0.05…0.95 incl. 0.95).
6. **Reading log-odds from waffle charts** (guess sign / rough size from a 10-square waffle).
7. Probability and log-odds together (linked slider); log-odds checks.
8. Bridge to logistic regression (grip equation, one real touch), drug-trial question (the other
   real touch), recap, summary.

## Checks (`class="card check" data-type="…"`)

- `mcq` — `.opts > .opt`; mark the right one `data-correct`. Keep to **2** options.
- `mcq` + `.opts.pick` — each `.opt` holds a mini SVG (`data-mini="1"` stacks).
- `numeric` — `data-answer` + `data-tol`; `input[type=number]` + `.numcheck`.
- `slider-numeric` — `data-target-p` + `data-p-tol` + `data-answer` + `data-tol`; the learner sets
  the slider near a probability, then types the odds. Both must be right to pass.
- `slider` — `data-target="logneg"` (log-odds < 0); `.lab` slider + `.numcheck`.

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
figurative or catchy phrasing. Keep it that way (`grep "—" index.html` should be 0).

## Authoring / verifying

- `index.html?all=1` reveals every step at once (unlocks all continues) for screenshots.
- Full-page screenshot: headless Chrome `--window-size=840,17000 --screenshot=… "…?all=1"`.
- Flow test: `scratchpad/test_flow.py` (Playwright) clicks through all 41 steps, exercises a
  wrong-answer path, and handles every check type (mcq / numeric / slider-numeric / slider).

Note: this file has been edited by more than one model. The folder suffix (`opus48_k7m2`) marks the
attempt; ignore other models' suffixed folders.
