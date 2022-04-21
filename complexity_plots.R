library(tidyverse)
library(patchwork)

setwd("L:/Mi unidad/MSc Bioinformatics/Machine Learning/Code/results")
CV_table = read_csv("2-level_CV_table_2022_04_20_15_16_13.csv")


raw_data = read_csv("2lCV_all_data_2022_04_20_16_08_19.csv")

all_data = raw_data %>% 
  mutate(model = case_when(`# model_index` == 0 ~ "Baseline",
                           `# model_index` == 1 ~ "Logistic regression",
                           `# model_index` == 2 ~ "ANN")) %>% 
  select(-`# model_index`) %>% 
  relocate(model)


LR_data = all_data %>% 
  filter(model == "Logistic regression")

LR_perf_plot = ggplot(LR_data) +
  geom_smooth(mapping = aes(x = complexity,
                            y = val_error_rate,
                            color = "Validation error"),
              se = FALSE) +
  geom_smooth(mapping = aes(x = complexity,
                            y = train_error_rate,
                            color = "Train error"),
              se = FALSE) +
  scale_color_manual(name = "",
                     breaks = c("Validation error", "Train error"),
                     values = c("Validation error" = "orange",
                                "Train error" = "skyblue")) +
  scale_x_log10() +
  xlab("Lambda") +
  ylab("Validation error\n(avg. across folds)") +
  labs(title = "Logistic regression")


ANN_data = all_data %>% 
  filter(model == "ANN")

ANN_perf_plot = ggplot(ANN_data) +
  geom_smooth(mapping = aes(x = complexity,
                            y = val_error_rate,
                            color = "Validation error"),
              se = FALSE) +
  geom_smooth(mapping = aes(x = complexity,
                            y = train_error_rate,
                            color = "Train error"),
              se = FALSE) +
  scale_color_manual(name = "",
                     breaks = c("Validation error", "Train error"),
                     values = c("Validation error" = "orange",
                                "Train error" = "skyblue")) +
  scale_x_log10() +
  xlab("Number of hidden units") +
  ylab("Validation error\n(avg. across folds)") +
  labs(title = "Artificial neural network")


LR_perf_plot + ANN_perf_plot +
  plot_annotation(title="Error rates across throughout different complexity of models") +
  plot_layout(guides = "collect") &
  theme_bw() &
  theme(legend.position = "bottom")













