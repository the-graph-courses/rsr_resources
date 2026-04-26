# ──────────────────────────────────────────────────────────────────────────────
# Peeling back the "scale-location" plot
#
# The check_model() scale-location plot uses sqrt(|standardized residuals|)
# vs fitted values. This script walks through each transformation separately,
# so you can SEE why each one is done.
#
# The takeaway for teaching: you don't need any of this. The plain
# residuals-vs-fitted plot already tells you everything about homogeneity
# of variance. The three transforms just make the LOESS trend line behave.
# ──────────────────────────────────────────────────────────────────────────────

if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, patchwork)

# ── 1. Simulate data with deliberate heteroscedasticity ──────────────────────
# Variance of y grows with x (classic "fanning" pattern)
set.seed(42)
n <- 200
x <- runif(n, 0, 10)
y <- 2 + 1.5 * x + rnorm(n, mean = 0, sd = 0.4 + 0.5 * x)  # sd grows with x
dat <- tibble(x = x, y = y)

fit <- lm(y ~ x, data = dat)

# Pull everything we need in one place
diag <- tibble(
  fitted        = fitted(fit),
  resid_raw     = resid(fit),                 # y_i - ŷ_i
  leverage      = hatvalues(fit),             # h_ii
  sigma_hat     = sigma(fit),                 # √(RSS / (n-p))
  resid_std     = rstandard(fit),             # r_i / (σ̂ · √(1-h_i))
  resid_abs     = abs(resid_std),             # |standardized residual|
  resid_sqrt    = sqrt(resid_abs)             # √|standardized residual|
)

# ── 2. Show each step side-by-side ───────────────────────────────────────────
theme_clean <- theme_minimal(base_size = 12) +
  theme(plot.title     = element_text(face = "bold", size = 11),
        plot.subtitle  = element_text(size  = 9,  color = "grey40"),
        panel.grid.minor = element_blank())

p1 <- ggplot(diag, aes(fitted, resid_raw)) +
  geom_point(alpha = 0.55, color = "#006D77") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_smooth(se = FALSE, color = "#FFD166", linewidth = 1) +
  labs(title    = "Step 0 – Raw residuals",
       subtitle = "y-axis: y − ŷ    •    You can ALREADY see the fan.",
       x = "fitted values", y = "residual") +
  theme_clean

p2 <- ggplot(diag, aes(fitted, resid_std)) +
  geom_point(alpha = 0.55, color = "#006D77") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_smooth(se = FALSE, color = "#FFD166", linewidth = 1) +
  labs(title    = "Step 1 – Standardize: r / (σ̂·√(1−h))",
       subtitle = "Leverage-adjusted, unitless. Pattern unchanged, scale common.",
       x = "fitted values", y = "standardized residual") +
  theme_clean

p3 <- ggplot(diag, aes(fitted, resid_abs)) +
  geom_point(alpha = 0.55, color = "#006D77") +
  geom_smooth(se = FALSE, color = "#FFD166", linewidth = 1) +
  labs(title    = "Step 2 – Take absolute value",
       subtitle = "We only care about size of scatter, not sign.",
       x = "fitted values", y = "|standardized residual|") +
  theme_clean

p4 <- ggplot(diag, aes(fitted, resid_sqrt)) +
  geom_point(alpha = 0.55, color = "#006D77") +
  geom_smooth(se = FALSE, color = "#FFD166", linewidth = 1) +
  labs(title    = "Step 3 – Take square root",
       subtitle = "Compresses big values; LOESS trend becomes meaningful.",
       x = "fitted values", y = "√|standardized residual|") +
  theme_clean

(p1 | p2) / (p3 | p4) +
  plot_annotation(
    title    = "From raw residuals to the scale-location plot",
    subtitle = "All four plots tell the same story: variance grows with fitted value.",
    theme = theme(plot.title    = element_text(face = "bold", size = 15),
                  plot.subtitle = element_text(size = 11, color = "grey40")))

# ── 3. Same story with HOMOSCEDASTIC data (for contrast) ─────────────────────
# When variance is actually constant, the LOESS in plot 4 should be ~flat
set.seed(1)
dat2 <- tibble(x = runif(n, 0, 10),
               y = 2 + 1.5 * x + rnorm(n, 0, 1))   # constant sd = 1
fit2 <- lm(y ~ x, data = dat2)

diag2 <- tibble(
  fitted     = fitted(fit2),
  resid_raw  = resid(fit2),
  resid_std  = rstandard(fit2),
  resid_sqrt = sqrt(abs(rstandard(fit2)))
)

q1 <- ggplot(diag2, aes(fitted, resid_raw)) +
  geom_point(alpha = 0.55, color = "#14a3a8") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_smooth(se = FALSE, color = "#FFD166") +
  labs(title = "Homoscedastic – raw residuals", x = "fitted", y = "residual") +
  theme_clean

q2 <- ggplot(diag2, aes(fitted, resid_sqrt)) +
  geom_point(alpha = 0.55, color = "#14a3a8") +
  geom_smooth(se = FALSE, color = "#FFD166") +
  labs(title = "Homoscedastic – scale-location", x = "fitted",
       y = "√|std resid|") +
  theme_clean

q1 | q2

# ── 4. What does performance::check_model() actually do? ─────────────────────
# If you have `performance` installed, compare:
# performance::check_model(fit,  check = "homogeneity")
# performance::check_model(fit2, check = "homogeneity")
#
# You'll see the exact same plots as Step 3 above, plus a reference line.
