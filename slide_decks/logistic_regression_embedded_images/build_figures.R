# Intro to logistic regression - figures + real outputs for the HTML slide deck
# Data: KLoSA Wave 1 (2006), men aged 45+ with measured grip strength (w01mgrip)
#   continuous outcome : grip strength in kg
#   binary outcome     : weak grip = grip <= 27 kg (EWGSOP2 cut for men)
#   predictor          : age (exact age, w01A002_age; cohort is 45+ by design)
# Palette: gold #d4af33, mist #c4d8db, teal #035f6c, alarm #b5451f, brown #7a6212
if (!require(pacman)) install.packages("pacman")
pacman::p_load(haven, tidyverse, patchwork, ragg)

gold <- "#d4af33"; mist <- "#c4d8db"; teal <- "#035f6c"; alarm <- "#b5451f"
brown <- "#7a6212"; ink <- "#073b42"; violet <- "#8c7ab8"; coral <- "#e58b73"

fig_dir <- "figures"
dir.create(fig_dir, showWarnings = FALSE)

save_fig <- function(p, name, w = 7.4, h = 5.2)
  ggsave(file.path(fig_dir, name), p, width = w, height = h, dpi = 200,
         device = ragg::agg_png, bg = "white")

write_out <- function(x, path)
  writeLines(sub("[[:space:]]+$", "", capture.output(print(x))), path)

base <- theme_minimal(base_size = 26) +
  theme(text = element_text(colour = teal, face = "bold"),
        axis.text = element_text(colour = teal),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = "grey90"),
        plot.title = element_text(size = 24, hjust = 0.5),
        plot.background = element_rect(fill = "white", colour = NA))

# =============================================================================
# DATA
# =============================================================================
RAW <- "/Users/kendavidn/Dropbox/Mac (2)/Downloads/KLoSA 1-9th wave (STATA)/Lt01_e.dta"
raw <- read_dta(RAW)

klosa <- tibble(
  age  = as.numeric(raw$w01A002_age),
  grip = as.numeric(raw$w01mgrip),
  sex  = as.numeric(raw$w01gender1)
) |>
  filter(sex == 1, !is.na(age), !is.na(grip), grip > 0, age >= 45) |>
  mutate(weak_grip = as.integer(grip <= 27)) |>
  select(age, grip, weak_grip)

cat("n =", nrow(klosa), "| ages", min(klosa$age), "-", max(klosa$age),
    "| % weak =", round(100 * mean(klosa$weak_grip), 1), "\n")

# ship the teaching extract alongside the deck
write_csv(klosa, "klosa_men_grip_45plus.csv")

# =============================================================================
# MODELS  (fit on the individual rows, never on the bands)
# =============================================================================
fit_lin <- lm(grip ~ age, data = klosa)
fit_log <- glm(weak_grip ~ age, family = binomial, data = klosa)
fit_gau <- glm(grip ~ age, family = gaussian, data = klosa)   # == lm, to make the GLM point

write_out(summary(fit_lin), "out_lm.txt")
write_out(summary(fit_log), "out_glm.txt")
write_out(coef(fit_gau), "out_glm_gaussian.txt")

b0 <- coef(fit_log)[1]; b1 <- coef(fit_log)[2]
a0 <- coef(fit_lin)[1]; a1 <- coef(fit_lin)[2]

cat(sprintf("\nlm : grip = %.2f + (%.3f) * age\n", a0, a1))
cat(sprintf("glm: log-odds(weak) = %.3f + %.4f * age | OR/yr = %.3f | OR/10yr = %.2f\n",
            b0, b1, exp(b1), exp(10 * b1)))
cat(sprintf("50%% crossing at age %.1f\n", -b0 / b1))

# probability at representative ages, and the per-year jump in probability
pr <- function(a) plogis(b0 + b1 * a)
tab <- tibble(age = c(50, 60, 70, 75, 80, 85)) |>
  mutate(p = pr(age), logodds = b0 + b1 * age, odds = exp(logodds),
         d_prob_next_year = pr(age + 1) - pr(age))
print(as.data.frame(tab), digits = 3)
write_out(as.data.frame(tab), "out_pred_table.txt")

# =============================================================================
# AGE BANDS (display only): 5-year bands from 45, top band 85+
# =============================================================================
bands <- klosa |>
  mutate(band = cut(age, breaks = c(seq(45, 85, 5), Inf), right = FALSE,
                    labels = c("45-49", "50-54", "55-59", "60-64", "65-69",
                               "70-74", "75-79", "80-84", "85+"))) |>
  group_by(band) |>
  summarise(n = n(), mid = mean(age), n_weak = sum(weak_grip),
            p = mean(weak_grip), .groups = "drop") |>
  mutate(odds = p / (1 - p), logodds = log(odds),
         lab = paste0(round(100 * p), "%"))
print(as.data.frame(bands), digits = 3)
write_out(as.data.frame(bands), "out_bands.txt")

# band midpoints used as the x position of each bar
bands <- bands |> mutate(x = c(47, 52, 57, 62, 67, 72, 77, 82, 87.5))

curve_df <- tibble(age = seq(44, 92, .25)) |>
  mutate(p = pr(age), logodds = b0 + b1 * age)

# =============================================================================
# FIG 1: the linear-regression reminder - scatter + OLS line
# =============================================================================
set.seed(9)
f_scatter <- ggplot(klosa, aes(age, grip)) +
  geom_point(colour = teal, alpha = .16, size = 1.5) +
  geom_hline(yintercept = 27, linetype = "dashed", colour = alarm, linewidth = 1.1) +
  annotate("label", x = 90.5, y = 30.2, label = "weak-grip cut: 27 kg", colour = alarm,
           fontface = "bold", size = 6.2, hjust = 1, label.size = 0, fill = "white",
           alpha = .8) +
  geom_abline(intercept = a0, slope = a1, colour = gold, linewidth = 2.2) +
  annotate("label", x = 90.5, y = 66, hjust = 1, colour = brown, fontface = "bold", size = 6.6,
           label.size = 0, fill = "white", alpha = .8,
           label = sprintf("grip = %.1f - %.3f x age", a0, abs(a1))) +
  scale_x_continuous(breaks = seq(45, 90, 10)) +
  coord_cartesian(xlim = c(45, 91), ylim = c(5, 70)) +
  labs(x = "age (years)", y = "grip strength (kg)") + base
save_fig(f_scatter, "fig_scatter_lm.png", w = 8.0, h = 5.0)

# =============================================================================
# FIG 2: the problem - a straight line through the probabilities leaves [0, 1]
# =============================================================================
lpm <- lm(p ~ x, data = bands, weights = n)   # straight line through the bar tops
l0 <- coef(lpm)[1]; l1 <- coef(lpm)[2]

f_problem <- ggplot(bands, aes(x, p)) +
  annotate("rect", xmin = 36, xmax = 112, ymin = 1, ymax = 1.3,
           fill = alarm, alpha = .08) +
  annotate("rect", xmin = 36, xmax = 112, ymin = -.3, ymax = 0,
           fill = alarm, alpha = .08) +
  geom_col(fill = coral, colour = "white", width = 4.6) +
  geom_hline(yintercept = c(0, 1), colour = ink, linewidth = .9) +
  geom_abline(intercept = l0, slope = l1, colour = alarm, linewidth = 2.2,
              linetype = "22") +
  annotate("text", x = 74, y = 1.17, hjust = .5, colour = alarm, fontface = "bold",
           size = 6.4, label = "probability > 100%  (impossible)") +
  annotate("text", x = 74, y = -.18, hjust = .5, colour = alarm, fontface = "bold",
           size = 6.4, label = "probability < 0%  (impossible)") +
  scale_y_continuous(labels = scales::percent, breaks = seq(0, 1, .5)) +
  scale_x_continuous(breaks = seq(40, 110, 10)) +
  coord_cartesian(xlim = c(36, 112), ylim = c(-.3, 1.3)) +
  labs(x = "age (years)", y = "P(weak grip)",
       title = "A straight line runs off both ends") + base
save_fig(f_problem, "fig_binary_problem.png", w = 8.0, h = 5.0)

# =============================================================================
# FIG 3: the fix - bars + logistic S-curve (probability scale)
# =============================================================================
f_curve <- ggplot(bands, aes(x, p)) +
  geom_col(fill = coral, colour = "white", width = 4.6) +
  geom_text(aes(label = lab), vjust = -.5, colour = brown, fontface = "bold", size = 5.4) +
  geom_line(data = curve_df, aes(age, p), colour = alarm, linewidth = 2.4) +
  geom_hline(yintercept = c(0, 1), colour = mist, linewidth = .8) +
  scale_y_continuous(labels = scales::percent, breaks = seq(0, 1, .25)) +
  scale_x_continuous(breaks = seq(45, 90, 5)) +
  coord_cartesian(xlim = c(44, 91), ylim = c(0, 1)) +
  labs(x = "age (years)", y = "P(weak grip)",
       title = "the same bars, with a fitted S-curve") + base
save_fig(f_curve, "fig_prob_curve.png", w = 8.4, h = 5.0)

# =============================================================================
# FIG 4: THE PAYOFF - same model, two scales, x-axes aligned by patchwork
#   top    = probability scale -> an S-curve
#   bottom = log-odds scale    -> a straight line
# =============================================================================
p_top <- ggplot(bands, aes(x, p)) +
  geom_col(fill = coral, colour = "white", width = 4.6) +
  geom_line(data = curve_df, aes(age, p), colour = alarm, linewidth = 2.2) +
  scale_y_continuous(labels = scales::percent, breaks = seq(0, 1, .5),
                     limits = c(0, 1.02)) +
  scale_x_continuous(breaks = seq(45, 90, 5), limits = c(43.5, 91.5)) +
  labs(x = NULL, y = "P(weak)", title = "PROBABILITY scale: a curve") +
  base + theme(plot.title = element_text(size = 23, colour = alarm, hjust = .5),
               axis.text.x = element_blank())

p_bot <- ggplot(bands, aes(x, logodds)) +
  geom_line(data = curve_df, aes(age, logodds), colour = teal, linewidth = 2.2) +
  geom_point(colour = brown, size = 5) +
  scale_x_continuous(breaks = seq(45, 90, 5), limits = c(43.5, 91.5)) +
  scale_y_continuous(breaks = seq(-4, 2, 2)) +
  labs(x = "age (years)", y = "log-odds",
       title = "LOG-ODDS scale: a straight line") +
  base + theme(plot.title = element_text(size = 23, colour = teal, hjust = .5))

f_two <- p_top / p_bot + plot_layout(heights = c(1, 1))
save_fig(f_two, "fig_prob_vs_logodds.png", w = 8.2, h = 7.2)

# =============================================================================
# FIG 5: why the log-odds scale earns its keep
#   probability: the per-year step keeps changing
#   log-odds   : the per-year step is the same everywhere (+0.117)
# =============================================================================
mark <- tibble(age = c(50, 62, 75, 85)) |>
  mutate(p = pr(age), logodds = b0 + b1 * age,
         dp = pr(age + 1) - pr(age),
         dlab = sprintf("+%.1f pp", 100 * dp))

g_top <- ggplot(curve_df, aes(age, p)) +
  geom_line(colour = alarm, linewidth = 2.2) +
  geom_segment(data = mark, aes(x = age - 3.6, xend = age + 3.6,
                                y = p - 3.6 * (pr(age + .5) - pr(age - .5)),
                                yend = p + 3.6 * (pr(age + .5) - pr(age - .5))),
               colour = ink, linewidth = 1.5) +
  geom_point(data = mark, aes(age, p), colour = brown, size = 4.6) +
  geom_label(data = mark, aes(age, p, label = dlab), vjust = -.9, hjust = .5,
             colour = brown, fontface = "bold", size = 5.4, label.size = 0,
             fill = "white", alpha = .85) +
  scale_y_continuous(labels = scales::percent, breaks = seq(0, 1, .5), limits = c(0, 1.12)) +
  scale_x_continuous(breaks = seq(45, 90, 5), limits = c(43.5, 91.5)) +
  labs(x = NULL, y = "P(weak)", title = "per extra year, the jump keeps changing") +
  base + theme(plot.title = element_text(size = 22, colour = alarm, hjust = .5),
               axis.text.x = element_blank())

g_bot <- ggplot(curve_df, aes(age, logodds)) +
  geom_line(colour = teal, linewidth = 2.2) +
  geom_point(data = mark, aes(age, logodds), colour = brown, size = 4.6) +
  geom_label(data = mark, aes(age, logodds, label = "+0.117"), vjust = -.85,
             colour = brown, fontface = "bold", size = 5.4, label.size = 0,
             fill = "white", alpha = .85) +
  scale_x_continuous(breaks = seq(45, 90, 5), limits = c(43.5, 91.5)) +
  scale_y_continuous(breaks = seq(-4, 2, 2), limits = c(-4.2, 2.2)) +
  labs(x = "age (years)", y = "log-odds",
       title = "per extra year, always the same step") +
  base + theme(plot.title = element_text(size = 22, colour = teal, hjust = .5))

f_slope <- g_top / g_bot + plot_layout(heights = c(1, 1))
save_fig(f_slope, "fig_slope_constant.png", w = 8.2, h = 7.2)

# =============================================================================
# FIG 6: the ladder - probability -> odds -> log-odds, on a number line
# =============================================================================
lad <- tibble(p = c(.05, .1, .25, .5, .75, .9, .95)) |>
  mutate(odds = p / (1 - p), logodds = log(odds))
print(as.data.frame(lad), digits = 3)
write_out(as.data.frame(lad), "out_ladder.txt")

f_ladder <- ggplot(lad, aes(logodds, 0)) +
  annotate("segment", x = -3.5, xend = 3.5, y = 0, yend = 0,
           colour = teal, linewidth = 1.4,
           arrow = arrow(ends = "both", length = unit(.25, "cm"), type = "closed")) +
  geom_point(colour = alarm, size = 5) +
  geom_text(aes(label = scales::percent(p, accuracy = 1)), vjust = -1.5,
            colour = alarm, fontface = "bold", size = 6) +
  geom_text(aes(label = sprintf("%.2f", logodds)), vjust = 2.6,
            colour = teal, fontface = "bold", size = 6) +
  annotate("text", x = 0, y = .62, label = "probability  (0 to 1, boxed in)", colour = alarm,
           fontface = "bold", size = 7) +
  annotate("text", x = 0, y = -.72, label = "log-odds  (runs forever, both ways)", colour = teal,
           fontface = "bold", size = 7) +
  annotate("text", x = -4.35, y = 0, label = "-∞", colour = teal,
           fontface = "bold", size = 8, hjust = .5) +
  annotate("text", x = 4.35, y = 0, label = "+∞", colour = teal,
           fontface = "bold", size = 8, hjust = .5) +
  coord_cartesian(xlim = c(-4.7, 4.7), ylim = c(-.95, .85)) +
  theme_void() + theme(plot.background = element_rect(fill = "white", colour = NA),
                       plot.margin = margin(6, 10, 6, 10))
save_fig(f_ladder, "fig_ladder.png", w = 9.4, h = 2.7)

# =============================================================================
# FIG 7: bars only - the observed probabilities, before any model is fitted
# =============================================================================
f_bars <- ggplot(bands, aes(x, p)) +
  geom_col(fill = coral, colour = "white", width = 4.6) +
  geom_text(aes(label = lab), vjust = -.5, colour = brown, fontface = "bold", size = 5.6) +
  scale_y_continuous(labels = scales::percent, breaks = seq(0, 1, .25)) +
  scale_x_continuous(breaks = seq(45, 90, 5)) +
  coord_cartesian(xlim = c(44, 91), ylim = c(0, .92)) +
  labs(x = "age band (5-year)", y = "P(weak grip)",
       title = "% of men with weak grip, by age band") + base
save_fig(f_bars, "fig_prob_bars.png", w = 8.4, h = 5.0)

# =============================================================================
# FIGS 8-10: THE TRIPTYCH - three panels, identical geometry, identical anchors
#   Same four ages (50/60/70/80) marked in gold on every panel, so the eye can
#   carry one row across all three scales.
#     panel A  grip in kg        -> straight line   (teal)
#     panel B  probability       -> a CURVE         (alarm)
#     panel C  log-odds          -> straight line   (teal)
# =============================================================================
anchor_ages <- c(50, 60, 70, 80)
anch <- tibble(age = anchor_ages) |>
  mutate(grip = a0 + a1 * age, logodds = b0 + b1 * age, p = plogis(logodds))

# one shared look, so the three panels read as a single system
panel <- base + theme(
  plot.title = element_text(size = 25, hjust = .5, face = "bold"),
  axis.title = element_text(size = 22),
  axis.text  = element_text(size = 20, colour = teal),
  plot.margin = margin(6, 10, 4, 6))

XS <- scale_x_continuous(breaks = seq(50, 90, 10), limits = c(44, 92))

# ---- panel A: the linear model we already know ------------------------------
pA <- ggplot(klosa, aes(age, grip)) +
  geom_point(colour = teal, alpha = .13, size = 1.3) +
  geom_hline(yintercept = 27, linetype = "dashed", colour = alarm, linewidth = .9) +
  geom_abline(intercept = a0, slope = a1, colour = teal, linewidth = 2.2) +
  geom_point(data = anch, aes(age, grip), shape = 21, size = 5.4,
             fill = gold, colour = "#6b5312", stroke = 1.2) +
  XS + coord_cartesian(ylim = c(8, 66)) +
  labs(x = "age (years)", y = "grip (kg)", title = "grip: a straight line") + panel
save_fig(pA, "fig_panel_linear.png", w = 6.6, h = 5.0)

# ---- panel B: the same predictor, a binary outcome -> a curve ----------------
pB <- ggplot(bands, aes(x, p)) +
  geom_col(fill = coral, colour = "white", width = 4.6) +
  geom_line(data = curve_df, aes(age, p), colour = alarm, linewidth = 2.2) +
  geom_point(data = anch, aes(age, p), shape = 21, size = 5.4,
             fill = gold, colour = "#6b5312", stroke = 1.2) +
  scale_y_continuous(labels = scales::percent, breaks = seq(0, 1, .25)) +
  XS + coord_cartesian(ylim = c(0, 1)) +
  labs(x = "age (years)", y = "P(weak grip)",
       title = "probability: a CURVE") +
  panel + theme(plot.title = element_text(size = 25, hjust = .5, face = "bold",
                                          colour = alarm))
save_fig(pB, "fig_panel_prob.png", w = 6.6, h = 5.0)

# ---- panel C: same model, log-odds scale -> straight again -------------------
pC <- ggplot(bands, aes(x, logodds)) +
  geom_line(data = curve_df, aes(age, logodds), colour = teal, linewidth = 2.2) +
  geom_point(colour = coral, size = 4) +
  geom_point(data = anch, aes(age, logodds), shape = 21, size = 5.4,
             fill = gold, colour = "#6b5312", stroke = 1.2) +
  scale_y_continuous(breaks = seq(-4, 2, 1)) +
  XS + coord_cartesian(ylim = c(-4, 2)) +
  labs(x = "age (years)", y = "log-odds of weak grip",
       title = "log-odds: a straight line") + panel
save_fig(pC, "fig_panel_logodds.png", w = 6.6, h = 5.0)

cat("\nANCHORS (the row that runs across all three panels)\n")
print(as.data.frame(anch |> mutate(odds = exp(logodds))), digits = 3)

cat("\nDONE. Figures in", fig_dir, "\n")
