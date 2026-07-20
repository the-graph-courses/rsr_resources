# Editing this deck

The reporting deck, after `interpreting_age_coefficient_opus48_b3k9`. That deck now derives
the odds ratio and shows `broom::tidy` in a compressed panel 3; **this deck is about using
and reporting it**: get the OR out of R, read it, give a concrete predicted probability, and
write the paper sentence. Same engine and design system as the other decks.

Edit each panel in place; `data-step` controls when a fragment appears (an element with
`data-step="N"` shows once `step ≥ N-1`).

## The four panels (2×2, read TL → BL → TR → BR)

1. **Get the odds ratio out of R** (top-left) — `broom::tidy(exponentiate = TRUE,
   conf.int = TRUE)` gives OR + 95% CI in one step (`exp(cbind(coef, confint))` by hand). SVG:
   a `0.1168 → exp() → 1.124` converter that recaps the previous deck.
2. **Reading an odds ratio** (bottom-left) — OR > 1 higher, < 1 lower, = 1 no effect;
   `(OR−1)×100` up or `(1−OR)×100` down; anchors OR 2 = twice, OR 0.5 = half; and the caution
   that an OR is **not** a risk ratio ("higher odds", not "more likely"). The visual is a plain
   HTML **reference card** (`.read-card`), not a chart — an earlier number-line diagram was
   removed as not useful.
3. **Predicted probabilities** (top-right) — the concrete, reader-friendly number. R how-to:
   `predict(model, newdata, type = "response")` (note `type = "response"` gives a probability,
   not a log-odds; `marginaleffects::predictions()` for CIs). SVG: the logistic curve with gold
   dots at ages 50/60/70/80 (5% / 14% / 34% / 63%).
4. **Write it in a paper** (bottom-right) — a model sentence (OR + CI + a predicted
   probability) and a short reporting checklist. No "what's next" teaser.

Removed from an earlier draft (per author): the OR number-line diagram, the whole "scale to a
decade" slide, and the teaser. Categorical predictors are intentionally left for a later deck.

## Numbers (KLoSA, `glm(weak_grip ~ age, binomial)`)

`β₁ = 0.1168` → OR/yr `1.124` (95% CI 1.114–1.135), 12% higher odds per year. Predicted
probabilities from `sig(age)`: 50 → 4.8%, 60 → 14.0%, 70 → 34.4%, 80 → 62.8%. All from
`GLM`/`sig` at the top of the script; keep in sync with `out_glm.txt`.

## Visuals / step engine

`STEP_CHANGES` controls only the SVG groups shown and the active panel. Groups: `g1a/g1b/g1c`
(converter stages), `g3curve/g3marks` (predicted-probability curve + dots); `c3` is the 0→1
curve draw. Panel 2's reference card and panel 4's cards are plain HTML revealed by
`data-step`. `index.html?still=1&step=NN` renders one fully-built step for screenshots.
