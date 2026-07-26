# Presenter script (draft) — Model fit 2: the LRT, anova(), and AIC

Narration draft keyed to deck steps (arrow through `index.html` while
reading). Panels light up in a 2×2 grid; practice overlays appear between
panels. Not a recording transcript.

**Panel 1 — Recap (steps 1–4)**

Last lesson we fitted two models to the KLoSA data. The null model ignores
age and predicts 0.206 for every man — the flat dashed line. The model
with age is the gold curve. The table computes both deviances exactly as in
part 1: 4259.0 for the null model, 3455.1 with age. The drop is 803.8.

But adding any predictor reduces deviance a little, even a useless one.
So the question is: is 803.8 more than chance alone could give? We need a
test.

**Practice overlay 1 (step 5)**

Four questions on the plot and the table.

**Panel 2 — The likelihood-ratio test (steps 6–9)**

Suppose age had no real effect. The drop would still not be exactly zero.
Statistics tells us its distribution: chi-squared, with degrees of freedom
equal to the number of added parameters. The curve shows the drops a
useless predictor produces. The 5% cutoff is 3.84. Our drop is 803.8 — far
beyond it.

In R this is one line: anova(m0, m1, test = "LRT"). Read the columns:
each model's residual deviance, the parameters added, the drop, and the
probability of such a drop by chance — here below 2.2 times 10 to the
minus 16. For a paper: "Age significantly improved model fit
(likelihood-ratio chi-squared(1) = 803.8, p < 0.001)."

**Practice overlay 2 (step 10)**

Four questions on reading the test.

**Panel 3 — Comparing models with anova() (steps 11–14)**

Now a second predictor. First an important rule: models can only be
compared on the same rows, so we remove men with missing BMI or smoking —
4,134 men remain, and the numbers change slightly. m1 uses age; m2 uses age
and BMI. anova(m1, m2, test = "LRT") gives a drop of 78.6 for one
parameter, p < 0.001. BMI improves the fit. Remember the two rules: the
models must be nested, and they must use the same rows.


With several predictors you do not need one anova call per pair.
m3 adds smoking — three categories, so two parameters. anova(m3,
test = "LRT") prints one row per term, added in order: age drops 770.0,
BMI drops another 78.6, smoking drops only 2.56 with p = 0.28. Smoking
gives no clear improvement. One caution: order matters. Each term is tested
after the terms above it.

**Practice overlay 3 (step 15)**

Four questions on the sequential table.

**Panel 4 — AIC (steps 16–18)**

Deviance always goes down when we add terms, so deviance alone cannot
choose between models. AIC adds a penalty: residual deviance plus 2 per
parameter. AIC(m0, m1, m2, m3) gives 4153.7, 3385.7, 3309.0, 3310.5.
Smoking removed 2.56 deviance but added 4 penalty points, so its AIC went
up. The lowest AIC is m2, age plus BMI — the same conclusion as the LRT.
Differences under about 2 are ties; then choose the simpler model. And
unlike the LRT, AIC can also compare non-nested models.

**Practice overlay 4 (step 19)**

Four final questions. After this, every part of the summary() bottom block
— null deviance, residual deviance, AIC — has been explained and used.
