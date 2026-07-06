# Donut sugar audit — figures + math for the SVG slide deck
# Palette: gold #d4af33, mist #c4d8db, teal #035f6c, alarm #b5451f
if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, ragg)

gold <- "#d4af33"; mist <- "#c4d8db"; teal <- "#035f6c"; alarm <- "#b5451f"
donut_dough <- "#c98245"; donut_edge <- "#7a4a22"

here_dir <- "figures"
deck_dir <- "."
mu <- 10

# ---- DATA (grams sugar) -----------------------------------------------------
# n = 10 audit: roughly normal, symmetric, no outliers -> a t-test is defensible
# even at n < 15. One-sided test (H1: mu > 10) is the primary test, and at n = 10
# it does NOT quite reject (p = 0.062). A larger n = 30 follow-up then does.
sugar10 <- c(7.1, 8.5, 9.5, 10.4, 11.2, 11.8, 12.5, 13.3, 14.2, 15.4)
sugar30 <- c(sugar10,
             10.9, 9.6, 11.7, 12.3, 10.2, 11.0, 13.1, 9.9, 12.6, 10.7,
             11.5, 10.4, 12.0, 9.3, 11.9, 13.6, 10.8, 11.3, 12.8, 8.9)

# ---- MATH -------------------------------------------------------------------
report <- function(x, mu = 10) {
  n <- length(x); m <- mean(x); s <- sd(x); se <- s / sqrt(n)
  t <- (m - mu) / se
  tibble(
    n = n, mean = round(m, 2), sd = round(s, 2), se = round(se, 2),
    t = round(t, 3), df = n - 1,
    p_greater = round(pt(t, n - 1, lower.tail = FALSE), 4),
    p_two   = round(2 * pt(-abs(t), n - 1), 4)
  )
}
stats <- bind_rows(report(sugar10), report(sugar30))
print(stats)
write_ttest <- function(test_result, path) {
  writeLines(sub("[[:space:]]+$", "", capture.output(print(test_result))), path)
}
write_ttest(t.test(sugar10, mu = mu, alternative = "greater"), file.path(deck_dir, "ttest_n10.txt"))
write_ttest(t.test(sugar30, mu = mu, alternative = "greater"), file.path(deck_dir, "ttest_n30.txt"))
write_csv(stats, file.path(deck_dir, "stats.csv"))

base <- theme_minimal(base_size = 26) +
  theme(text = element_text(colour = teal, face = "bold"),
        axis.text = element_text(colour = teal),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = "grey90"),
        plot.background = element_rect(fill = "white", colour = NA))

save_fig <- function(p, name, w = 7.4, h = 5.2)
  ggsave(file.path(here_dir, name), p, width = w, height = h, dpi = 200,
         device = ragg::agg_png, bg = "white")

pop_sd <- round(sd(sugar10), 1)

# ---- FIG 1: the 10 donuts ---------------------------------------------------
f1 <- ggplot(tibble(sugar = sugar10), aes(sugar)) +
  geom_dotplot(binwidth = .5, fill = donut_dough, colour = donut_edge) +
  geom_vline(xintercept = mu, colour = gold, linewidth = 2) +
  geom_vline(xintercept = mean(sugar10), colour = alarm, linewidth = 2, linetype = "dashed") +
  scale_y_continuous(NULL, breaks = NULL) +
  coord_cartesian(xlim = c(7, 16)) +
  labs(x = "grams sugar") + base
save_fig(f1, "fig1_dots.png", w = 9, h = 3.6)

# ---- FIG 1B: the 30-donut follow-up ----------------------------------------
f1b <- ggplot(tibble(sugar = sugar30), aes(sugar)) +
  geom_dotplot(binwidth = .35, fill = donut_dough, colour = donut_edge) +
  geom_vline(xintercept = mu, colour = gold, linewidth = 2) +
  geom_vline(xintercept = mean(sugar30), colour = alarm, linewidth = 2, linetype = "dashed") +
  scale_y_continuous(NULL, breaks = NULL) +
  coord_cartesian(xlim = c(7, 16)) +
  labs(x = "grams sugar") + base
save_fig(f1b, "fig1b_dots30.png", w = 9, h = 3.2)

# ---- FIG 2: the null world (centered on the claim) --------------------------
f2 <- ggplot(tibble(x = seq(4, 16, .05)), aes(x)) +
  stat_function(fun = dnorm, args = list(mu, pop_sd), geom = "area", fill = mist) +
  geom_vline(xintercept = mu, colour = teal, linewidth = 2) +
  annotate("text", x = mu, y = 0.38, label = "center = 10g", colour = teal, fontface = "bold") +
  scale_y_continuous(NULL, breaks = NULL) +
  labs(x = "grams sugar (if the claim is true)") + base
save_fig(f2, "fig2_null.png")

# ---- FIG 3: individuals vary a lot, the average barely ----------------------
f3 <- ggplot(tibble(x = seq(4, 16, .05)), aes(x)) +
  stat_function(fun = dnorm, args = list(mu, pop_sd), geom = "area", fill = mist, alpha = .8) +
  stat_function(fun = dnorm, args = list(mu, pop_sd / sqrt(10)), geom = "area", fill = gold, alpha = .85) +
  annotate("text", x = 13.2, y = .22, label = "one donut", colour = teal, fontface = "bold") +
  annotate("text", x = mu, y = .55, label = "average of 10", colour = "#7a6212", fontface = "bold") +
  scale_y_continuous(NULL, breaks = NULL) +
  labs(x = "grams sugar") + base
save_fig(f3, "fig3_se.png")

# ---- FIG 4: the t-distribution + our result (one-sided upper tail) ----------
tv <- report(sugar10)$t
df10 <- length(sugar10) - 1
xs <- seq(-4, 4, .01); td <- tibble(x = xs, d = dt(xs, df10))
base_t <- theme_minimal(base_size = 34) +
  theme(text = element_text(colour = teal, face = "bold"),
        axis.text = element_text(colour = teal, size = 30),
        axis.title = element_text(size = 32),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = "grey90"),
        plot.background = element_rect(fill = "white", colour = NA))
f4 <- ggplot(td, aes(x, d)) +
  geom_area(fill = mist) +
  geom_area(data = filter(td, x >=  abs(tv)), fill = gold) +
  geom_vline(xintercept = tv, colour = alarm, linewidth = 1.6) +
  annotate("text", x = tv, y = .30, label = sprintf("t = %.2f", tv),
           colour = alarm, fontface = "bold", hjust = -0.05, size = 11) +
  scale_y_continuous(NULL, breaks = NULL) +
  labs(x = "t  (standard errors from the claim)") + base_t
save_fig(f4, "fig4_tdist.png", w = 9.2, h = 4.8)

# ---- FIG 4B: the n = 30 t-distribution + follow-up result ------------------
tv30 <- report(sugar30)$t
df30 <- length(sugar30) - 1
td30 <- tibble(x = xs, d = dt(xs, df30))
f4b <- ggplot(td30, aes(x, d)) +
  geom_area(fill = mist) +
  geom_area(data = filter(td30, x >= abs(tv30)), fill = gold) +
  geom_vline(xintercept = tv30, colour = alarm, linewidth = 1.6) +
  annotate("text", x = 2.2, y = .30, label = sprintf("t = %.2f", tv30),
           colour = alarm, fontface = "bold", hjust = 0, size = 10) +
  scale_y_continuous(NULL, breaks = NULL) +
  labs(x = "t  (standard errors from the claim)") + base_t
save_fig(f4b, "fig4b_tdist30.png", w = 7.8, h = 3.2)

# ---- FIG 5: n = 10 vs n = 30 — same mean, tighter net ---------------------
ci_row <- function(x, lab) {
  n <- length(x); se <- sd(x)/sqrt(n); h <- qt(.975, n - 1) * se
  tibble(lab = lab, m = mean(x), lo = mean(x) - h, hi = mean(x) + h)
}
ci <- bind_rows(ci_row(sugar10, "n = 10"), ci_row(sugar30, "n = 30"))
f5 <- ggplot(ci, aes(m, lab)) +
  geom_vline(xintercept = mu, colour = gold, linewidth = 2) +
  geom_errorbar(aes(xmin = lo, xmax = hi), width = .18, orientation = "y", colour = teal, linewidth = 1.6) +
  geom_point(colour = alarm, size = 6) +
  labs(x = "estimated mean sugar (95% CI)", y = NULL) + base
save_fig(f5, "fig5_intervals.png")

cat("\nDONE. Figures written to", here_dir, "\n")
