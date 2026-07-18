# Extra figure for the one-slide intro: the raw 0/1 scatter, before binning.
# Uses the shipped teaching extract so it reproduces without the STATA file.
if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, ragg)

gold <- "#d4af33"; mist <- "#c4d8db"; teal <- "#035f6c"; alarm <- "#b5451f"
brown <- "#7a6212"; ink <- "#073b42"; coral <- "#e58b73"

klosa <- read_csv("klosa_men_grip_45plus.csv", show_col_types = FALSE)

base <- theme_minimal(base_size = 26) +
  theme(text = element_text(colour = teal, face = "bold"),
        axis.text = element_text(colour = teal),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = "grey90"),
        plot.title = element_text(size = 24, hjust = 0.5),
        plot.background = element_rect(fill = "white", colour = NA))

set.seed(9)
f_binary <- ggplot(klosa, aes(age, weak_grip)) +
  geom_jitter(height = .06, width = .35, colour = teal, alpha = .12, size = 1.4) +
  scale_y_continuous(breaks = c(0, 1),
                     labels = c("0\nnot weak", "1\nweak"),
                     limits = c(-.18, 1.18)) +
  scale_x_continuous(breaks = seq(45, 90, 10)) +
  coord_cartesian(xlim = c(45, 91)) +
  labs(x = "age (years)", y = "weak grip?",
       title = "every man is a 0 or a 1") + base
ggsave("figures/fig_binary_scatter.png", f_binary, width = 8.0, height = 3.9,
       dpi = 200, device = ragg::agg_png, bg = "white")
cat("done\n")
