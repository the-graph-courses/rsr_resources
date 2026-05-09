library(tidyverse)
library(haven)
library(broom)

base <- "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"

demo <- read_xpt(paste0(base, "DEMO_J.xpt")) %>%
  select(SEQN, sex = RIAGENDR, age = RIDAGEYR)

diet <- read_xpt(paste0(base, "DR1TOT_J.xpt")) %>%
  select(SEQN, kcal = DR1TKCAL, recall_status = DR1DRSTZ, wtdrd1 = WTDRD1)

dxa <- read_xpt(paste0(base, "DXX_J.xpt")) %>%
  select(SEQN, body_fat_pct = DXDTOPF, dxa_status = DXAEXSTS)

dat <- demo %>%
  inner_join(diet, by = "SEQN") %>%
  inner_join(dxa, by = "SEQN") %>%
  mutate(
    sex = factor(sex, levels = c(1, 2), labels = c("Male", "Female"))
  ) %>%
  filter(
    age >= 20,
    age <= 59,
    recall_status == 1,
    dxa_status == 1,
    !is.na(kcal),
    !is.na(body_fat_pct)
  )

# Pooled bivariate relationship
m0 <- lm(body_fat_pct ~ kcal, data = dat)

# Add sex as confounder
m1 <- lm(body_fat_pct ~ kcal + sex, data = dat)

# Allow sex-specific slopes
m2 <- lm(body_fat_pct ~ kcal * sex, data = dat)

tidy(m0)
tidy(m1)
tidy(m2)

# Sex-specific bivariate slopes
dat %>%
  group_by(sex) %>%
  do(tidy(lm(body_fat_pct ~ kcal, data = .))) %>%
  filter(term == "kcal")