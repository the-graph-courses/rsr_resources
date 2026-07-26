library(tidyverse)
library(forcats)

# Outcome: ever married (vs never married); drop non-answers
gss <- gss_cat |>
  filter(marital != "No answer", !is.na(age)) |>
  mutate(
    ever_married = if_else(marital == "Never married", 0L, 1L),
    race = fct_drop(race)
  )

# ---- Univariable: age only ----
m_uni <- glm(ever_married ~ age, family = binomial, data = gss)
summary(m_uni)

# Bin age; % ever married in each bin
gss_binned <- gss |>
  mutate(age_bin = cut_width(age, width = 5, boundary = 15, closed = "left")) |>
  group_by(age_bin) |>
  summarise(
    n = n(),
    pct_married = mean(ever_married) * 100,
    age_mid = mean(age),
    .groups = "drop"
  )

# Smooth logistic curve across the observed age range
curve_df <- tibble(age = seq(min(gss$age), max(gss$age), length.out = 200))
curve_df$pct_married <- predict(m_uni, newdata = curve_df, type = "response") * 100

ggplot() +
  geom_col(
    data = gss_binned,
    aes(x = age_mid, y = pct_married, fill = "Observed %"),
    width = 4,
    alpha = 0.7
  ) +
  geom_line(
    data = curve_df,
    aes(x = age, y = pct_married, colour = "Logistic fit"),
    linewidth = 1.2
  ) +
  scale_fill_manual(values = c("Observed %" = "steelblue"), name = NULL) +
  scale_colour_manual(values = c("Logistic fit" = "darkred"), name = NULL) +
  labs(
    title = "Ever married by age (GSS)",
    x = "Age (years)",
    y = "% ever married"
  ) +
  theme_minimal()

# ---- Multivariable: age + race ----
m_multi <- glm(ever_married ~ age + race, family = binomial, data = gss)
summary(m_multi)
anova(m_multi, test = "Chisq")

# Compare nested models
anova(m_uni, m_multi, test = "Chisq")
