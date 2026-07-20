# Editing this deck

Odds-ratio focused deck: what the age coefficient means, how it becomes an odds
ratio, and how to read/report it. Predicted probabilities are left out on purpose.

Everything is one self-contained `index.html`. Edit each panel in place; `data-step`
controls when a fragment appears (an element with `data-step="N"` shows once
`step ≥ N-1`). 33 steps total.

## The four panels (2×2)

1. **Weak grip by age: summary output** (top-left) — data import, `glm` + `summary`,
   coefficient table, then the figure builds in sequence (bars → dots → gold curve).
   Ends on β₁ = 0.1168 as a log-odds change. SVG `svg1`.
2. **The same model on two scales** (bottom-left) — derivation line by line from the
   probability form through η to the log-odds form, then the morph (probability
   S-curve → log-odds line). The coefficient is the slope. SVG `svg2` (morph `t`
   0→1 with the fitted equation).
3. **Odds ratios are comparisons** (top-right) — continuous age (`mod2`, OR 1.124)
   beside a teaching-only polytomous age-group model (`mod3`; vs 45–64: OR 5.22 for
   65–74, OR 14.4 for 75+). Each side has paper-ready reporting wording. Under both
   columns: a horizontal OR → plain-words strip, then a published forest-plot example
   from Mansuy et al., Nature Communications (2021), Fig. 2
   (`images/nature_or_forest_plot.png`).
4. **Blank** (bottom-right) — intentionally empty.

## Numbers (KLoSA, `glm(weak_grip ~ age, binomial)`)

β₁ = 0.1168 → OR/yr 1.124 (95% CI 1.114–1.135), 12% higher odds/yr.
For the teaching-only age-group model, the 45–64 reference yields OR 5.22
(95% CI 4.35–6.28) for 65–74 and OR 14.44 (95% CI 11.44–18.22) for 75+.
Keep the continuous results in sync with `out_glm.txt` / `GLM`.

## Visuals / step engine

`STEP_CHANGES` controls SVG groups and the active panel; each entry inherits the
previous. Groups: `g1bars/g1dots/g1curve` (P1), `g2plot/g2lab` (P2 morph).
Animation vars: `c1` (P1 curve draw), `t` (P2 morph). Panels 3 and 4 are plain HTML
revealed by `data-step`.
`index.html?still=1&step=NN` renders one fully-built step for screenshots.
