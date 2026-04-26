body_fat <- read.csv("/Users/kendavidn/Dropbox/tgc_github_projects/rsr_resources/datasets_staging/cleaned/01_bodyfat_johnson.csv")

head(body_fat)


# spanish adults
# /Users/kendavidn/Dropbox/tgc_github_projects/rsr_resources/datasets_staging/cleaned/04_bodyfat_spanish.csv
spanish_adults <- read.csv("/Users/kendavidn/Dropbox/tgc_github_projects/rsr_resources/datasets_staging/cleaned/04_bodyfat_spanish.csv")

head(spanish_adults)


# birthweight
birthweight <- read.csv("/Users/kendavidn/Dropbox/tgc_github_projects/rsr_resources/datasets_staging/cleaned/03_birthweight.csv")

head(birthweight)

datasets_staging/cleaned/01_bodyfat_johnson.csv
datasets_staging/cleaned/02_diabetes_efron.csv
datasets_staging/cleaned/03_fev_kahn.csv
datasets_staging/cleaned/04_bodyfat_spanish.csv
datasets_staging/cleaned/05_birthweight_secher.csv
datasets_staging/cleaned/06_peak_power_bone.csv
datasets_staging/cleaned/07_alcohol_metabolism.csv
datasets_staging/cleaned/08_cystic_fibrosis.csv
datasets_staging/cleaned/09_bodyfat_german.csv
datasets_staging/cleaned/10_birthweight_baystate.csv
datasets_staging/cleaned/11_hiv_6mwt.csv
datasets_staging/cleaned/12_ggt_atherosclerosis.csv
datasets_staging/cleaned/13_cimt_ra.csv
datasets_staging/cleaned/14_oc_prostate.csv
datasets_staging/cleaned/15_pam13.csv
datasets_staging/cleaned/16_med_student_qol.csv
datasets_staging/cleaned/17_depression_anxiety.csv
datasets_staging/cleaned/18_prevend.csv
datasets_staging/cleaned/19_famuss.csv
datasets_staging/cleaned/20_kidiq.csv
datasets_staging/cleaned/21_births14.csv
datasets_staging/cleaned/22_un_member_states_2024.csv
datasets_staging/cleaned/23_nhanes_adult.csv
datasets_staging/cleaned/24_county.csv
datasets_staging/cleaned/25_boston.csv


#
# birthweight_secher.csv
birthweight_secher <- read.csv("/Users/kendavidn/Dropbox/tgc_github_projects/rsr_resources/datasets_staging/cleaned/05_birthweight_secher.csv")

head(birthweight_secher)


# peak_power_bone.csv
peak_power_bone <- read.csv("/Users/kendavidn/Dropbox/tgc_github_projects/rsr_resources/datasets_staging/cleaned/06_peak_power_bone.csv")

head(peak_power_bone)

# alcohol_metabolism.csv
alcohol_metabolism <- read.csv("/Users/kendavidn/Dropbox/tgc_github_projects/rsr_resources/datasets_staging/cleaned/07_alcohol_metabolism.csv")

head(alcohol_metabolism)

# cystic_fibrosis.csv
cystic_fibrosis <- read.csv("/Users/kendavidn/Dropbox/tgc_github_projects/rsr_resources/datasets_staging/cleaned/08_cystic_fibrosis.csv")

head(cystic_fibrosis)


# ---------------------------------------------------------------
# Main regressions from the documented outcome ~ predictor pairs
# ---------------------------------------------------------------

library(performance)
library(see)

# 22 — UN member states 2024: life expectancy ~ HDI
un_member_states <- read.csv("/Users/kendavidn/Dropbox/tgc_github_projects/rsr_resources/datasets_staging/cleaned/22_un_member_states_2024.csv")

fit_un <- lm(life_expectancy_yrs ~ hdi, data = un_member_states)
summary(fit_un)

# scatterplot
library(ggplot2)
ggplot(un_member_states, aes(x = hdi, y = life_expectancy_yrs)) +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_minimal()

par(mfrow = c(2, 2))
plot(fit_un)
summary(fit_un)


dev.off()


check_model(fit_un)

check_heteroscedasticity(fit_un)
check_outliers(fit_un)
check_normality(fit_un)



# 13 — CV risk factors & carotid intima-media thickness in RA: cimt_total ~ age_yrs
cimt_ra <- read.csv("/Users/kendavidn/Dropbox/tgc_github_projects/rsr_resources/datasets_staging/cleaned/13_cimt_ra.csv")

fit_cimt <- lm(cimt_total ~ age_yrs, data = cimt_ra)
summary(fit_cimt)
check_model(fit_cimt)


# 3 — FEV lung function in youths (Kahn/Tager): fev_l ~ height_in
fev_kahn <- read.csv("/Users/kendavidn/Dropbox/tgc_github_projects/rsr_resources/datasets_staging/cleaned/03_fev_kahn.csv")

fit_fev <- lm(fev_l ~ height_in, data = fev_kahn)
summary(fit_fev)
check_model(fit_fev)




set.seed(1234)
x <- rnorm(200)
z <- rnorm(200)
# quadratic relationship
y <- 2 * x + x^2 + 4 * z + rnorm(200)

d <- data.frame(x, y, z)
m <- lm(y ~ x + z, data = d)
check_model(m)
