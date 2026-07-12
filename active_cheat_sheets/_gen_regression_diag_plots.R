# Generate real performance::check_model() panels for the regression
# diagnostics active cheat sheet. Titles/subtitles stripped (students must
# identify each plot); fonts enlarged so labels survive small embedding.
suppressMessages({
  library(performance)
  library(see)
  library(ggplot2)
})

out_dir <- "active_cheat_sheets/regression_diagnostics_files"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

big_theme <- theme(
  text = element_text(size = 15),
  plot.title = element_blank(),
  plot.subtitle = element_blank(),
  axis.title = element_text(size = 15, face = "bold"),
  axis.text = element_text(size = 12.5),
  plot.margin = margin(6, 8, 4, 4)
)

grab_panel <- function(model, which, file, xlab = NULL, ylab = NULL) {
  cm <- check_model(model, panel = FALSE)
  pl <- plot(cm)
  p <- pl[[which]] + labs(title = NULL, subtitle = NULL) + big_theme
  if (!is.null(xlab)) p <- p + xlab(xlab)
  if (!is.null(ylab)) p <- p + ylab(ylab)
  ggsave(file.path(out_dir, file), p, width = 3.5, height = 2.65, dpi = 200, bg = "white")
}

# 0. MULTICOLLINEARITY: height ~ age + grade (near-duplicate predictors)
set.seed(21)
n <- 80
age <- runif(n, 6, 16)
grade <- age - 5 + rnorm(n, 0, 0.35)
height <- 90 + 6.2 * age + rnorm(n, 0, 4)
m_vif <- lm(height ~ age + grade)
grab_panel(m_vif, "VIF", "vif_bad.png")

# 1. LINEARITY violated: study hours vs quiz score, saturating curve
set.seed(42)
n <- 70
hours <- runif(n, 0, 10)
quiz <- pmin(100, 100 * (1 - exp(-0.45 * hours)) + rnorm(n, 0, 4))
m_lin <- lm(quiz ~ hours)
grab_panel(m_lin, "NCV", "linearity_bad.png")

# 2. HOMOGENEITY violated: household income vs medical spending, fan shape
set.seed(7)
n <- 90
income <- runif(n, 20, 200) # thousands
spend <- 0.5 + 0.04 * income + rnorm(n, 0, 0.0004 * income^1.7)
m_hom <- lm(spend ~ income)
grab_panel(m_hom, "HOMOGENEITY", "homogeneity_bad.png")

# 3. INFLUENTIAL point: BMI vs total cholesterol + one extreme point
set.seed(11)
n <- 24
bmi <- runif(n, 20, 32)
chol <- 120 + 3.2 * bmi + rnorm(n, 0, 9)
bmi <- c(bmi, 48)
chol <- c(chol, 130)
m_inf <- lm(chol ~ bmi)
grab_panel(m_inf, "OUTLIERS", "influential_bad.png")

# 4. NORMALITY violated: ER visits vs poverty, heavy right tail
set.seed(19)
n <- 80
poverty <- runif(n, 5, 40)
er_visits <- rpois(n, lambda = exp(-0.5 + 0.08 * poverty))
# sprinkle a few huge positive residuals (data-entry / rare events)
er_visits[c(3, 11, 27)] <- er_visits[c(3, 11, 27)] + c(18, 24, 30)
m_nrm <- lm(er_visits ~ poverty)
grab_panel(m_nrm, "QQ", "normality_bad.png",
           xlab = "Normal quantiles", ylab = "Deviation from line")

cat("done\n")
for (f in list.files(out_dir, full.names = TRUE)) {
  info <- file.info(f)
  cat(basename(f), round(info$size / 1024), "KB\n")
}
