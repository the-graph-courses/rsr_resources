# Types of t-tests + hypotheses — figures + real outputs for the SVG slide deck
# Palette: gold #d4af33, mist #c4d8db, teal #035f6c, alarm #b5451f
if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, ragg, patchwork)

gold <- "#d4af33"; mist <- "#c4d8db"; teal <- "#035f6c"; alarm <- "#b5451f"
violet <- "#8c7ab8"; coral <- "#e58b73"; ink <- "#073b42"

here_dir <- "figures"
deck_dir <- "."
dir.create(here_dir, showWarnings = FALSE)
set.seed(7)

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
# EXAMPLE DATA (health) + real t.test outputs
# =============================================================================

# ---- ONE-SAMPLE: mg of active drug per tablet, target = 500 mg --------------
pill <- c(498.2, 501.5, 499.7, 502.3, 500.8, 497.6,
          503.1, 499.0, 501.9, 500.4, 498.8, 502.7)
os <- t.test(pill, mu = 500)
write_out(os, file.path(deck_dir, "out_one_sample.txt"))

# ---- TWO-SAMPLE (independent): pain relief score (0-10), drug A vs drug B ----
relief_A <- c(4.1, 5.2, 3.8, 4.9, 4.4, 5.0, 3.5, 4.7, 4.2, 5.3)
relief_B <- c(6.3, 5.8, 6.9, 6.1, 7.2, 5.5, 6.6, 7.0, 6.2, 5.9)
ts <- t.test(relief_B, relief_A)
write_out(ts, file.path(deck_dir, "out_two_sample.txt"))

# ---- PAIRED: systolic BP (mmHg) before & after a 6-week drug, same patients --
# one patient barely changes (148 -> 148) and one gets worse (155 -> 161)
before <- c(150, 148, 162, 142, 155, 149, 163, 158)
after  <- c(141, 148, 151, 137, 161, 140, 152, 150)
paired1 <- t.test(after, before, paired = TRUE)   # H0: mean diff = 0
paired2 <- t.test(after - before, mu = 0)         # identical: one-sample on diffs
write_out(paired1, file.path(deck_dir, "out_paired.txt"))
write_out(paired2, file.path(deck_dir, "out_paired_as_one.txt"))

# =============================================================================
# FIGURES — one per test type, drawn in the same visual language
# =============================================================================

# ---- ONE-SAMPLE: a single cloud of dots vs one fixed line (the target) ------
ps <- tibble(x = "this batch", mg = pill)
f_one <- ggplot(ps, aes(x, mg)) +
  geom_hline(yintercept = 500, colour = gold, linewidth = 2.2) +
  geom_jitter(width = .09, height = 0, size = 4.2, alpha = .9, colour = "#3a6f78") +
  stat_summary(fun = mean, geom = "crossbar", width = .42, linewidth = 1, colour = alarm) +
  annotate("text", x = 1.5, y = 500, label = "target 500", colour = "#7a6212",
           fontface = "bold", hjust = 1, vjust = 1.5, size = 6.6) +
  labs(x = NULL, y = "mg active drug") +
  coord_cartesian(ylim = c(496.5, 503.8), clip = "off") + base
save_fig(f_one, "fig_one_group.png", w = 5.4, h = 4.6)

# ---- TWO-SAMPLE: two independent clouds, each with its mean -----------------
grp <- tibble(
  drug   = factor(rep(c("Drug A", "Drug B"), each = 10)),
  relief = c(relief_A, relief_B)
)
f_two <- ggplot(grp, aes(drug, relief, colour = drug)) +
  geom_jitter(width = .12, height = 0, size = 4, alpha = .85) +
  stat_summary(fun = mean, geom = "crossbar", width = .5, linewidth = 1, colour = teal) +
  scale_colour_manual(values = c("Drug A" = violet, "Drug B" = coral), guide = "none") +
  labs(x = NULL, y = "pain relief (0-10)") +
  coord_cartesian(ylim = c(3, 8)) + base
save_fig(f_two, "fig_two_group.png", w = 5.4, h = 4.6)

# ---- PAIRED: connected pairs  ->  collapse to one cloud of differences vs 0 --
pdat <- tibble(patient = factor(1:8), before = before, after = after) |>
  pivot_longer(c(before, after), names_to = "time", values_to = "bp") |>
  mutate(time = factor(time, levels = c("before", "after")))
pL <- ggplot(pdat, aes(time, bp, group = patient)) +
  geom_line(colour = violet, linewidth = 1.2, alpha = .9) +
  geom_point(colour = ink, size = 3) +
  labs(x = NULL, y = "systolic BP (mmHg)", subtitle = "each patient: before vs after") +
  coord_cartesian(ylim = c(135, 166)) + base

ddat <- tibble(g = "after - before", diff = after - before)
pR <- ggplot(ddat, aes(g, diff)) +
  geom_hline(yintercept = 0, colour = gold, linewidth = 2.2) +
  geom_jitter(width = .05, height = 0, colour = violet, size = 3.6) +
  stat_summary(fun = mean, geom = "crossbar", width = .42, linewidth = .9, colour = alarm) +
  labs(x = NULL, y = "difference (mmHg)", subtitle = "the differences, vs 0") +
  coord_cartesian(ylim = c(-14, 8)) + base

f_paired <- pL + pR + plot_layout(widths = c(1.15, 1))
save_fig(f_paired, "fig_paired.png", w = 8.6, h = 4.6)

cat("\nDONE. Figures in", here_dir, "\n")
