# SHIN assumption checks: simple linear regression with categorical predictors
# Binary (2 groups) and polytomous (3+ groups) examples
# See code_alongs/regression_diagnostics_shin.Rmd for reading check_model() panels

if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, performance, MASS)

# ---------------------------------------------------------------------------
# Binary predictor: birth weight ~ smoking status (birthwt)
# ---------------------------------------------------------------------------
birthwt <- birthwt %>%
  mutate(smoke = factor(smoke, levels = c(0, 1), labels = c("nonsmoker", "smoker")))

mod_binary <- lm(bwt ~ smoke, data = birthwt)
summary(mod_binary)

ggplot(birthwt, aes(smoke, bwt)) +
  geom_boxplot(outlier.shape = NA, fill = "grey90") +
  geom_jitter(width = 0.12, alpha = 0.35, size = 1.5) +
  labs(title = "Binary predictor: compare group means and spread", y = "Birth weight (g)") +
  theme_minimal(base_size = 13)

check_model(mod_binary, show_ci =FALSE  )  # H: equal spread across groups; I & N: as usual

plot(mod_binary, which = 3)
performance::check_heteroskedasticity(mod_binary)


diag_plots <- function(mod) {
  is_cat <- any(sapply(mod$model[-1], function(x) is.factor(x) || is.character(x)))
  if (is_cat) {
    aug <- broom::augment(mod)
    p1 <- ggplot(aug, aes(.fitted, sqrt(abs(.std.resid)))) +
      geom_jitter(width = 8, alpha = 0.4, color = "steelblue") +
      stat_summary(fun = mean, geom = "point", shape = 95, size = 20, color = "darkgreen") +
      labs(title = "Homogeneity of variance", x = "Fitted", y = "sqrt(|Std. resid|)") +
      theme_minimal(base_size = 13)
    p2 <- ggplot(aug, aes(sample = .std.resid)) +
      stat_qq(alpha = 0.5, color = "steelblue") + stat_qq_line(color = "darkgreen") +
      labs(title = "Normality of residuals") + theme_minimal(base_size = 13)
    patchwork::wrap_plots(p1, p2)
  } else {
    check_model(mod)
  }
}

diag_plots(mod_binary)

# ---------------------------------------------------------------------------
# Polytomous predictor: tooth length ~ vitamin C dose (ToothGrowth)
# Reference level = 0.5 mg (first level in factor())
# ---------------------------------------------------------------------------
tooth <- ToothGrowth %>%
  mutate(dose_group = factor(dose, levels = c(0.5, 1, 2)))

mod_poly <- lm(len ~ dose_group, data = tooth)
summary(mod_poly)

ggplot(tooth, aes(dose_group, len)) +
  geom_boxplot(outlier.shape = NA, fill = "grey90") +
  geom_jitter(width = 0.12, alpha = 0.35, size = 1.5) +
  labs(title = "Polytomous predictor: one mean shift per non-reference level", y = "Tooth length") +
  theme_minimal(base_size = 13)

check_model(mod_poly)
