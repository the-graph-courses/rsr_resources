# Two-sample (independent) t-test - figures + real outputs for the SVG slide deck
# Hero data: MASS::birthwt  (birth weight, grams, by maternal smoking) - real health data
# Practice data: datasets::ToothGrowth (tooth length by vitamin-C delivery, OJ vs VC)
# Palette: gold #d4af33, mist #c4d8db, teal #035f6c, alarm #b5451f
if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, ragg, MASS)

gold <- "#d4af33"; mist <- "#c4d8db"; teal <- "#035f6c"; alarm <- "#b5451f"
violet <- "#8c7ab8"; coral <- "#e58b73"; ink <- "#073b42"; bluegrey <- "#3a6f78"

here_dir <- "figures"
deck_dir <- "."
dir.create(here_dir, showWarnings = FALSE)

base <- theme_minimal(base_size = 26) +
  theme(text = element_text(colour = teal, face = "bold"),
        axis.text = element_text(colour = teal),
        plot.subtitle = element_text(colour = "#3a555a", face = "bold", size = 18),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = "grey90"),
        plot.background = element_rect(fill = "white", colour = NA))

base_t <- theme_minimal(base_size = 34) +
  theme(text = element_text(colour = teal, face = "bold"),
        axis.text = element_text(colour = teal, size = 28),
        axis.title = element_text(size = 30),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = "grey90"),
        plot.background = element_rect(fill = "white", colour = NA))

save_fig <- function(p, name, w = 7.4, h = 5.2)
  ggsave(file.path(here_dir, name), p, width = w, height = h, dpi = 200,
         device = ragg::agg_png, bg = "white")

write_out <- function(x, path)
  writeLines(sub("[[:space:]]+$", "", capture.output(print(x))), path)

# =============================================================================
# HERO: birth weight (g) by maternal smoking - real two-sample t-test
# =============================================================================
data(birthwt, package = "MASS")
bw <- birthwt |>
  mutate(smoke = factor(smoke, levels = c(0, 1),
                        labels = c("non-smoker", "smoker")))

ts_bw <- t.test(bwt ~ smoke, data = bw)               # Welch, two-sided (default)
write_out(ts_bw, file.path(deck_dir, "out_birthwt.txt"))

# group summaries used verbatim on the slides
bw_summ <- bw |>
  group_by(smoke) |>
  summarise(n = n(), mean = mean(bwt), sd = sd(bwt))
print(bw_summ)

# ---- FIG: two independent clouds, each with its mean ------------------------
f_bw <- ggplot(bw, aes(smoke, bwt, colour = smoke)) +
  geom_jitter(width = .14, height = 0, size = 2.2, alpha = .5) +
  stat_summary(fun = mean, geom = "crossbar", width = .55, linewidth = 1, colour = teal) +
  scale_colour_manual(values = c("non-smoker" = bluegrey, "smoker" = coral), guide = "none") +
  labs(x = NULL, y = "birth weight (g)") +
  coord_cartesian(ylim = c(700, 5000)) + base
save_fig(f_bw, "fig_birthwt_groups.png", w = 5.6, h = 4.6)

# ---- FIG: t-distribution (df ~ 170) + our result, TWO-SIDED shading ---------
tv <- unname(ts_bw$statistic); dfv <- unname(ts_bw$parameter)
xs <- seq(-4, 4, .01); td <- tibble(x = xs, d = dt(xs, dfv))
f_td <- ggplot(td, aes(x, d)) +
  geom_area(fill = mist) +
  geom_area(data = filter(td, x >=  abs(tv)), fill = gold) +
  geom_area(data = filter(td, x <= -abs(tv)), fill = gold) +
  geom_vline(xintercept = tv, colour = alarm, linewidth = 1.6) +
  annotate("text", x = tv, y = .30, label = sprintf("t = %.2f", tv),
           colour = alarm, fontface = "bold", hjust = -0.06, size = 10) +
  scale_y_continuous(NULL, breaks = NULL) +
  coord_cartesian(ylim = c(0, .55)) +
  labs(x = "t  (standard errors apart)") + base_t
save_fig(f_td, "fig_tdist.png", w = 9.2, h = 4.6)

# ---- FIG: 95% CI for the difference in means (non-smoker - smoker) ----------
ci <- tibble(lab = "non-smoker - smoker",
             m = diff(rev(ts_bw$estimate)) * -1,   # estimate[1]-estimate[2]
             lo = ts_bw$conf.int[1], hi = ts_bw$conf.int[2])
ci$m <- unname(ts_bw$estimate[1] - ts_bw$estimate[2])
f_ci <- ggplot(ci, aes(m, lab)) +
  geom_vline(xintercept = 0, colour = gold, linewidth = 2) +
  geom_errorbar(aes(xmin = lo, xmax = hi), width = .22, orientation = "y",
                colour = teal, linewidth = 1.6) +
  geom_point(colour = alarm, size = 6) +
  annotate("text", x = 0, y = 1.34, label = "no difference", colour = "#7a6212",
           fontface = "bold", size = 6) +
  labs(x = "difference in mean birth weight (g), 95% CI", y = NULL) +
  coord_cartesian(xlim = c(-60, 560)) + base +
  theme(axis.text.y = element_blank())
save_fig(f_ci, "fig_ci_diff.png", w = 7.6, h = 2.6)

# =============================================================================
# PRACTICE: ToothGrowth - tooth length by vitamin-C delivery (OJ vs VC)
# =============================================================================
ts_tg <- t.test(len ~ supp, data = ToothGrowth)       # Welch, two-sided
write_out(ts_tg, file.path(deck_dir, "out_tooth.txt"))
tg_summ <- ToothGrowth |>
  group_by(supp) |>
  summarise(n = n(), mean = mean(len), sd = sd(len))
print(tg_summ)

f_tg <- ggplot(ToothGrowth, aes(supp, len, colour = supp)) +
  geom_jitter(width = .13, height = 0, size = 3.4, alpha = .8) +
  stat_summary(fun = mean, geom = "crossbar", width = .5, linewidth = 1, colour = teal) +
  scale_colour_manual(values = c("OJ" = "#e5a92f", "VC" = bluegrey), guide = "none") +
  labs(x = NULL, y = "tooth length") +
  coord_cartesian(ylim = c(3, 36)) + base
save_fig(f_tg, "fig_tooth_groups.png", w = 5.6, h = 4.6)

cat("\nDONE. Figures in", here_dir, "\n")
