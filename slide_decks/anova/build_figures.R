# One-way ANOVA - figures + real outputs for the SVG slide deck
# Data: rsr::prevend  (PREVEND cohort, n = 500) - RFFT cognitive score by education
# Factor: education (primary < lower_secondary < higher_secondary < university)
# Palette: gold #d4af33, mist #c4d8db, teal #035f6c, alarm #b5451f
if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, ragg)
pacman::p_load_gh("the-graph-courses/rsr")

gold <- "#d4af33"; mist <- "#c4d8db"; teal <- "#035f6c"; alarm <- "#b5451f"
violet <- "#8c7ab8"; coral <- "#e58b73"; ink <- "#073b42"; bluegrey <- "#3a6f78"
brown <- "#7a6212"

# four education colours (light -> dark, ordered)
edu_cols <- c("primary"          = "#c9a96a",
              "lower_secondary"  = "#e58b73",
              "higher_secondary" = "#3a8f9c",
              "university"       = "#035f6c")

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

# short labels for compact axes
short <- c("primary" = "primary", "lower_secondary" = "lower\nsec.",
           "higher_secondary" = "higher\nsec.", "university" = "univ.")
# very short labels for the 3-panel partition figure
vshort <- c("prim", "l.sec", "h.sec", "univ")

# =============================================================================
# DATA + MODELS
# =============================================================================
prevend <- rsr::prevend |>
  mutate(education = factor(education,
    levels = c("primary", "lower_secondary", "higher_secondary", "university")))

grand <- mean(prevend$rfft)

gs <- prevend |>
  group_by(education) |>
  summarise(n = n(), mean = mean(rfft), sd = sd(rfft), .groups = "drop")
print(as.data.frame(gs))

fit <- aov(rfft ~ education, data = prevend)
write_out(summary(fit), file.path(deck_dir, "out_anova.txt"))

tuk <- TukeyHSD(fit)
write_out(tuk, file.path(deck_dir, "out_tukey.txt"))

welch <- oneway.test(rfft ~ education, data = prevend)
write_out(welch, file.path(deck_dir, "out_welch.txt"))

sm <- summary(fit)[[1]]
Fobs <- unname(sm$`F value`[1]); df1 <- sm$Df[1]; df2 <- sm$Df[2]
cat(sprintf("F=%.2f df=(%d,%d) eta2=%.3f\n", Fobs, df1, df2,
            sm$`Sum Sq`[1] / sum(sm$`Sum Sq`)))

# =============================================================================
# FIG 1: grouped jitter + group means (the raw picture)
# =============================================================================
set.seed(7)
f_groups <- ggplot(prevend, aes(education, rfft, colour = education)) +
  geom_hline(yintercept = grand, linetype = "dashed", colour = brown, linewidth = 1) +
  geom_jitter(width = .16, height = 0, size = 2.1, alpha = .45) +
  stat_summary(fun = mean, geom = "crossbar", width = .58, linewidth = 1.1, colour = ink) +
  annotate("text", x = 0.62, y = grand + 5, label = "grand mean", colour = brown,
           fontface = "bold", size = 6, hjust = 0) +
  scale_colour_manual(values = edu_cols, guide = "none") +
  scale_x_discrete(labels = short) +
  labs(x = NULL, y = "RFFT cognitive score") +
  coord_cartesian(ylim = c(0, 140)) + base
save_fig(f_groups, "fig_groups.png", w = 7.6, h = 5.0)

# =============================================================================
# FIG 2: variance partition - same data, three lenses
#   TOTAL (point -> grand mean) = BETWEEN (group mean -> grand) + WITHIN (point -> group mean)
# =============================================================================
set.seed(7)
d <- prevend |>
  group_by(education) |>
  mutate(xj = as.numeric(education) + runif(n(), -.18, .18),
         gmean = mean(rfft)) |>
  ungroup()
# sample a manageable number of segments so arrows read clearly
set.seed(11)
ds <- d |> group_by(education) |> slice_sample(n = 22) |> ungroup()

gm_df <- gs |>
  mutate(x = as.numeric(education),
         mean_lab = sprintf("%.1f", mean),
         mean_vjust = c(1.8, 1.8, -.9, -.9))

panel_theme <- base + theme(
  axis.text.x = element_text(size = 19),
  plot.title = element_text(size = 22, colour = teal, hjust = .5),
  legend.position = "none")

p_total <- ggplot(ds, aes(xj, rfft)) +
  geom_hline(yintercept = grand, linetype = "dashed", colour = brown, linewidth = 1) +
  geom_segment(aes(xend = xj, yend = grand), colour = brown, alpha = .5, linewidth = .5) +
  geom_point(aes(colour = education), size = 1.7, alpha = .8) +
  scale_colour_manual(values = edu_cols) +
  scale_x_continuous(breaks = 1:4, labels = vshort) +
  coord_cartesian(ylim = c(0, 140)) +
  labs(title = "TOTAL: SST", x = NULL, y = "RFFT") + panel_theme

p_between <- ggplot(gm_df, aes(x, mean)) +
  geom_hline(yintercept = grand, linetype = "dashed", colour = brown, linewidth = 1) +
  annotate("text", x = 4.45, y = grand - 6.2, label = sprintf("%.1f", grand),
           colour = brown, fontface = "bold", size = 3.4, hjust = 1) +
  geom_segment(aes(xend = x, yend = grand), colour = "#1f7a4d", linewidth = 2.6) +
  geom_point(aes(colour = education), size = 5) +
  geom_text(aes(label = mean_lab, vjust = mean_vjust), colour = ink,
            fontface = "bold", size = 4.6, show.legend = FALSE) +
  scale_colour_manual(values = edu_cols) +
  scale_x_continuous(breaks = 1:4, labels = vshort, limits = c(.5, 4.5)) +
  coord_cartesian(ylim = c(0, 140)) +
  labs(title = "BETWEEN: SSB (signal)", x = NULL, y = NULL) + panel_theme

p_within <- ggplot(ds, aes(xj, rfft)) +
  geom_segment(aes(xend = xj, yend = gmean), colour = violet, alpha = .55, linewidth = .65) +
  geom_point(aes(colour = education), size = 1.7, alpha = .8) +
  geom_segment(data = gm_df, aes(x = x - .25, xend = x + .25, y = mean, yend = mean),
               colour = ink, linewidth = 1.1) +
  scale_colour_manual(values = edu_cols) +
  scale_x_continuous(breaks = 1:4, labels = vshort) +
  coord_cartesian(ylim = c(0, 140)) +
  labs(title = "WITHIN: SSW (noise)", x = NULL, y = NULL) + panel_theme

pacman::p_load(patchwork)
f_part <- p_total + p_between + p_within + plot_layout(nrow = 1)
save_fig(f_part, "fig_partition.png", w = 13.5, h = 4.8)

# =============================================================================
# FIG 3: F-distribution df=(3,496); critical region + observed F far in tail
# =============================================================================
fcrit <- qf(.95, df1, df2)
xs <- seq(0, 6, .01)
fd <- tibble(x = xs, d = df(xs, df1, df2))
f_fdist <- ggplot(fd, aes(x, d)) +
  geom_area(fill = mist) +
  geom_area(data = filter(fd, x >= fcrit), fill = gold) +
  geom_vline(xintercept = fcrit, colour = brown, linewidth = 1.2, linetype = "dashed") +
  annotate("text", x = fcrit + .15, y = .55, hjust = 0, colour = brown,
           fontface = "bold", size = 8,
           label = sprintf("F* = %.2f\n(5%% tail)", fcrit)) +
  annotate("segment", x = 4.75, xend = 5.7, y = .34, yend = .07,
           colour = alarm, linewidth = 1.4,
           arrow = arrow(length = unit(.28, "cm"), type = "closed")) +
  annotate("text", x = 4.05, y = .31, hjust = 0.5, colour = alarm,
           fontface = "bold", size = 9, label = "observed\nF = 73.3") +
  annotate("text", x = 5.85, y = .02, hjust = 1, colour = alarm,
           fontface = "bold", size = 7, label = "(far off the chart)") +
  scale_y_continuous(NULL, breaks = NULL) +
  coord_cartesian(ylim = c(0, .85), xlim = c(0, 6)) +
  labs(x = sprintf("F  (df = %d, %d)", df1, df2)) + base_t
save_fig(f_fdist, "fig_fdist.png", w = 9.2, h = 4.8)

# =============================================================================
# FIG 4: Tukey HSD - all 6 pairwise differences with 95% family-wise CIs
# =============================================================================
tk <- as.data.frame(tuk$education)
tk$pair <- rownames(tk)
tk$pair <- gsub("_", " ", tk$pair)
tk <- tk |> arrange(diff) |> mutate(pair = factor(pair, levels = pair))
f_tukey <- ggplot(tk, aes(diff, pair)) +
  geom_vline(xintercept = 0, colour = brown, linewidth = 1.4) +
  geom_errorbarh(aes(xmin = lwr, xmax = upr), height = .32, colour = teal, linewidth = 1.3) +
  geom_point(colour = alarm, size = 4) +
  annotate("text", x = 1.5, y = 6.75, label = "no difference (0)", colour = brown,
           fontface = "bold", size = 5, hjust = 0) +
  labs(x = "difference in mean RFFT (95% family-wise CI)", y = NULL) +
  coord_cartesian(xlim = c(-2, 58), ylim = c(.5, 7.1)) +
  base + theme(axis.text.y = element_text(size = 20))
save_fig(f_tukey, "fig_tukey.png", w = 8.4, h = 5.0)

cat("\nDONE. Figures in", here_dir, "\n")
