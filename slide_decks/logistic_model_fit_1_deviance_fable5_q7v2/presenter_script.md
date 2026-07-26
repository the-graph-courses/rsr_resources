# Presenter script (draft) — Model fit 1: null deviance, residual deviance, AIC

Narration draft keyed to deck steps (arrow through `index.html` while
reading). Panels light up in a 2×2 grid; practice overlays appear between
panels. Not a recording transcript.

**Panel 1 — Scoring the model's predictions (steps 1–8)**

Every time you run summary() on a logistic model, the last three lines show
null deviance, residual deviance, and AIC. This lesson explains all three.

We start with a small example: six students, one exam. Each dot is one
student — 1 means they passed, 0 means they failed. Notice student D:
four hours of study, but D failed.

We fit the model with glm, and the gold curve shows the predicted
probability of passing at each number of hours. The green area below the
curve is the probability of passing. The red area above the curve is the
probability of failing. At every point, the two heights add up to 1.

Now we score each prediction. If the student passed, the score is the
predicted probability of passing — the green height at their position. If
the student failed, the score is the probability of failing — the red
height. Look at the coloured segments: student A failed, and the model gave
failing a probability of 0.954, so A's score is high. Student D failed, but
the model gave failing only 0.353. That prediction was poor.

Multiply the six scores: 0.084. This product is called the likelihood. It
is the probability the model gave to the results that actually happened.
A higher likelihood means a better fit. Two quick practice questions before
we move on.

**Practice overlay 1 (step 9)**

Work through the four questions using the plot. The key idea to check:
passed students are scored with the green height, failed students with the
red height.

**Panel 2 — From scores to deviance (steps 10–15)**

One problem: with thousands of people, the product of scores becomes far
too small for a computer to store. So we take logarithms — the product
becomes a sum. The table adds ln(score) across the six students: −2.478.
That is exactly what logLik(mod) returns.

Deviance is defined as −2 times the log-likelihood: here, 4.956. Lower
deviance means a better fit, and a perfect model would have deviance 0.
The −2 is there so that differences in deviance follow a chi-squared
distribution — that becomes important next lesson.

Now fit a model with no predictors: glm(pass ~ 1). Three of six students
passed, so it predicts 0.5 for everyone. Every score is 0.5, and the
deviance works out to 8.318.

Look again at the summary output: null deviance 8.3178, residual deviance
4.9560. The null deviance is the deviance of the no-predictor model. The
residual deviance is the deviance of your model. The degrees of freedom are
the number of people minus the number of parameters.

**Practice overlay 2 (step 16)**

Four questions on the table and the scale. Check the −2 rule and which
model each deviance belongs to.

**Panels 3 and 4 — The real data: KLoSA (steps 17–22)**

The same picture with real data: 4,184 men, weak grip predicted from age.
The red area below the curve is the probability of weak grip; the green
area above is the probability of no weak grip. The table is the same as
before, only longer: the sum of ln(score) is −1727.6, so the residual
deviance is 3455.1. The null model predicts 0.206 for everyone, and its
deviance is 4259.0.

The AIC line is simple arithmetic: residual deviance plus 2 per parameter.
Here 3455.1 + 4 = 3459.1. We use AIC to compare models — next lesson.

Adding age reduced the deviance by 803.8. Is that more than chance could
give? That question needs a test: the likelihood-ratio test, next lesson.

**Practice overlay 3 (step 23)**

Four questions on the real output. The last one sets up part 2: a drop
alone is not enough — we need a test.
