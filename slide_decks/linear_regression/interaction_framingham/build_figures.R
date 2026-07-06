# Interaction terms — figures + real outputs for the slide deck
# Data: Framingham Heart Study via LSTbook
# Outcome: systolic BP; predictors: BMI, BP meds (+ covariates in adjusted model)
# Palette: gold #d4af33, mist #c4d8db, teal #035f6c, alarm #b5451f
if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, ragg, LSTbook)

gold <- "#d4af33"; mist <- "#c4d8db"; teal <- "#035f6c"; alarm <- "#b5451f"
violet <- "#8c7ab8"; coral <- "#e58b73"; ink <- "#073b42"; brown <- "#7a6212"
med_cols <- c("Not on meds" = "#3a8f9c", "On meds" = "#b5451f")

here_dir <- "figures"
deck_dir <- "."
dir.create(here_dir, showWarnings = FALSE)

base <- theme_minimal(base_size = 26) +
  theme(text = element_text(colour = teal, face = "bold"),
        axis.text = element_text(colour = teal),
        plot.subtitle = element_text(colour = "#3a555a", face = "bold", size = 18),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = "grey90"),
        plot.background = element_rect(fill = "white", colour = NA),
        legend.position = "bottom",
        legend.title = element_blank())

save_fig <- function(p, name, w = 7.4, h = 5.2)
  ggsave(file.path(here_dir, name), p, width = w, height = h, dpi = 200,
         device = ragg::agg_png, bg = "white")

write_out <- function(x, path)
  writeLines(sub("[[:space:]]+$", "", capture.output(print(x))), path)

# =============================================================================
# DATA + MODELS
# =============================================================================
dat <- LSTbook::Framingham |>
  select(sysBP, BMI, BPMeds, age, totChol, heartRate, cigsPerDay) |>
  mutate(BPMeds = factor(BPMeds, labels = c("Not on meds", "On meds")))

model_dat <- dat |>
  filter(!is.na(sysBP), !is.na(BMI), !is.na(BPMeds))

adjusted_dat <- dat |>
  filter(if_all(c(sysBP, BMI, BPMeds, age, totChol, heartRate, cigsPerDay), ~ !is.na(.x)))

m1 <- lm(sysBP ~ BMI, data = model_dat)
m2 <- lm(sysBP ~ BMI + BPMeds, data = model_dat)
m3 <- lm(sysBP ~ BMI * BPMeds, data = model_dat)
m4 <- lm(sysBP ~ BMI * BPMeds + age + totChol + heartRate + cigsPerDay, data = adjusted_dat)

write_out(summary(m1), file.path(deck_dir, "out_m1.txt"))
write_out(summary(m2), file.path(deck_dir, "out_m2.txt"))
write_out(summary(m3), file.path(deck_dir, "out_m3.txt"))
write_out(summary(m4), file.path(deck_dir, "out_m4.txt"))
write_out(anova(m2, m3), file.path(deck_dir, "out_anova_m2_m3.txt"))

gs <- model_dat |>
  group_by(BPMeds) |>
  summarise(n = n(), mean_BMI = mean(BMI), mean_sysBP = mean(sysBP), .groups = "drop")
print(as.data.frame(gs))

# fitted lines for plotting
bmi_seq <- seq(min(model_dat$BMI), max(model_dat$BMI), length.out = 100)
pred1 <- tibble(BMI = bmi_seq, fit = predict(m1, newdata = tibble(BMI = bmi_seq)))
pred_grid <- expand.grid(BMI = bmi_seq, BPMeds = levels(model_dat$BPMeds)) |>
  as_tibble() |>
  mutate(BPMeds = factor(BPMeds, levels = levels(model_dat$BPMeds)))
pred_grid <- pred_grid |>
  mutate(fit2 = predict(m2, newdata = pick(BMI, BPMeds)),
         fit3 = predict(m3, newdata = pick(BMI, BPMeds)))

set.seed(11)
ds <- model_dat |> slice_sample(n = 1200)

# =============================================================================
# FIG 1: raw scatter coloured by BP meds
# =============================================================================
f_scatter <- ggplot(ds, aes(BMI, sysBP, colour = BPMeds)) +
  geom_point(alpha = .45, size = 1.8) +
  scale_colour_manual(values = med_cols) +
  labs(x = "BMI (kg/m²)", y = "Systolic BP (mmHg)") + base
save_fig(f_scatter, "fig_scatter.png", w = 7.2, h = 5.0)

# =============================================================================
# FIG 2–4: model fit overlays
# =============================================================================
line_base <- function(title) {
  ggplot(ds, aes(BMI, sysBP, colour = BPMeds)) +
    geom_point(alpha = .45, size = 1.4) +
    scale_colour_manual(values = med_cols) +
    labs(title = title, x = "BMI (kg/m²)", y = "Systolic BP (mmHg)") + base +
    theme(plot.title = element_text(size = 22, hjust = .5))
}

f_m1 <- line_base("Model 1: BMI only") +
  geom_line(data = pred1, aes(x = BMI, y = fit), colour = brown, linewidth = 1.4,
            inherit.aes = FALSE)
save_fig(f_m1, "fig_one_line.png", w = 7.2, h = 5.0)

f_m2 <- line_base("Model 2: parallel lines (+ BPMeds)") +
  geom_line(data = pred_grid, aes(x = BMI, y = fit2, colour = BPMeds), linewidth = 1.5,
            inherit.aes = FALSE)

save_fig(f_m2, "fig_parallel.png", w = 7.2, h = 5.0)

f_m3 <- line_base("Model 3: BMI x BP meds") +
  geom_line(data = pred_grid, aes(x = BMI, y = fit3, colour = BPMeds), linewidth = 1.5,
            inherit.aes = FALSE) +
  scale_x_continuous(limits = c(0, NA))
save_fig(f_m3, "fig_interaction.png", w = 7.2, h = 5.0)

# =============================================================================
# FIG 5: three-panel evolution
# =============================================================================
pacman::p_load(patchwork)
p1 <- f_m1 + theme(legend.position = "none", plot.title = element_text(size = 18))
p2 <- f_m2 + theme(legend.position = "none", plot.title = element_text(size = 18))
p3 <- f_m3 + theme(legend.position = "none", plot.title = element_text(size = 18))
f_evolution <- p1 + p2 + p3 + plot_layout(nrow = 1)
save_fig(f_evolution, "fig_evolution.png", w = 14.5, h = 4.6)

cat("\nDONE. Figures in", here_dir, "\n")
