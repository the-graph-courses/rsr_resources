library(ggplot2)
library(inspectdf)

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- sub("^--file=", "", script_arg)
project_dir <- normalizePath(file.path(dirname(script_path), ".."))
data_dir <- file.path(project_dir, "data")
figure_dir <- file.path(project_dir, "figures")

datasets <- sub("\\.csv$", "", list.files(data_dir, pattern = "\\.csv$"))

for (dataset in datasets) {
  data <- read.csv(file.path(data_dir, paste0(dataset, ".csv")), check.names = FALSE)

  categorical <- inspect_cat(data)
  categorical_plot <- show_plot(categorical) +
    labs(title = NULL) +
    theme_minimal(base_family = "Helvetica") +
    theme(
      panel.grid.minor = element_blank(),
      plot.background = element_rect(fill = "white", colour = NA)
    )
  ggsave(
    file.path(figure_dir, paste0(dataset, "_inspect_cat.png")),
    categorical_plot,
    width = 12,
    height = max(5, nrow(categorical) * 0.55),
    dpi = 160,
    bg = "white"
  )

  numeric <- inspect_num(data)
  numeric_plot <- show_plot(numeric) +
    labs(title = NULL) +
    theme_minimal(base_family = "Helvetica") +
    theme(
      panel.grid.minor = element_blank(),
      plot.background = element_rect(fill = "white", colour = NA)
    )
  ggsave(
    file.path(figure_dir, paste0(dataset, "_inspect_num.png")),
    numeric_plot,
    width = 12,
    height = max(5, nrow(numeric) * 0.42),
    dpi = 160,
    bg = "white"
  )
}
