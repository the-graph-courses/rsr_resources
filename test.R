# Load packages
if(!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse, openintro)


# scatter plot of weight vs weeks

preemies <- openintro::births14  %>% 
#filter(premie == "premie")  %>% 
mutate(weight_kg = weight * 0.45359) 
#%>%
#slice_tail(n = 20)

model <- lm(weight ~ weeks + gained, data = preemies)
summary(model)

ggplot(preemies, aes(x = weeks, y = weight)) +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_minimal()

performance::check_model(model)

# iris length vs iris width and check model

iris_data <- iris %>%
  mutate(obs_id = row_number())

model <- lm(Sepal.Length ~ Sepal.Width, data = iris_data)
summary(model)
performance::check_model(model)

slice_head(n = 5)


slice_tail(n = 8)
ggplot(aes(x = weeks, y = weight)) +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_minimal()

# summary of the data
summary(openintro::births14)

# correlation between weight and weeks
cor(openintro::births14$weight, openintro::births14$weeks)

# regression of weight on weeks

?openintro::births14


set.seed(3)
data_for_regression <- babies |>
  filter(gestation < 259) |>
  slice_sample(n = 9)   %>% 
  select(gestation, bwt)

# plot the data
ggplot(data_for_regression, aes(x = gestation, y = bwt)) +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_minimal()

fit <- lm(bwt ~ gestation, data = data_for_regression)
summary(fit)
check_model(fit)


pacman::p_load(tidyverse, openintro)
# Prep data
preemie_data <- openintro::births14 %>%
  filter(premie == "premie") %>%
  mutate(weight_kg = weight * 0.453592) %>%
  slice_tail(n = 8)  %>% 
  arrange(weeks, weight_kg) %>%
  select(weeks, weight_kg)

  preemie_data

preemie_data
# Fit model
model <- lm(weight_kg ~ weeks, data = preemie_data)

# View results
summary(model)


norm_good <- data.frame(
  x = c(155, 157, 159, 160, 162, 163, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 177, 178, 180, 182, 184, 186, 188, 190, 158, 164, 169, 176, 181, 185),
  y = c(3.66, 3.49, 3.76, 3.52, 3.92, 3.8, 4.11, 3.83, 4.04, 3.89, 4.25, 4.1, 4.38, 4.15, 4.37, 4.16, 4.56, 4.41, 4.66, 4.49, 4.82, 4.7, 5.07, 4.85, 3.68, 3.73, 4.24, 4.3, 4.67, 4.71)
)



model <- lm(y ~ x, data = norm_good)
summary(model)
performance::check_model(model)

