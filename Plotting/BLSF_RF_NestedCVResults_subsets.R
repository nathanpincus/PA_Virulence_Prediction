library(tidyverse)

# Read in nested CV with all features as comparison
All <- readr::read_csv("/path/to/BLSF_AGEs_RF_GSCV_NestedCVResults.csv") %>%
  dplyr::mutate(subset = "all")


# For subset into 2
pth <- "/path/to/"
files <- dir(path = pth,
             pattern = glob2rx("*Results*of2.csv"))

subset_2 <- tibble::tibble(filename = files) %>%
  dplyr::mutate(file_contents = purrr::map(filename,
                                           ~ read_csv(file.path(pth, .)))) %>%
  tidyr::unnest() %>%
  dplyr::mutate(subset = stringr::str_remove(filename, "BLSF_RF_NestedCVResults_subset")) %>%
  dplyr::mutate(subset = stringr::str_remove(subset, ".csv")) %>%
  dplyr::bind_rows(All)
subset_2$subset <- factor(subset_2$subset,levels = c("all", "1of2", "2of2"))


# Dot Plot with Mean + 95% CI indicated
ggplot(data = subset_2,  mapping = aes(x=subset, y=test_accuracy)) + 
  ylim(0, 1) +
  geom_dotplot(binaxis='y', stackdir = 'center', dotsize = 0.7) + 
  stat_summary(fun.y = mean, geom = "errorbar", aes(ymax = ..y.., ymin = ..y..),
               width = 0.75, size = 1, linetype = "solid", colour = "red") +  
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar",
               width = (0.75/2), size = 0.5, linetype = "solid", colour = "red") +
  labs(y = "Accuracy", x = "Subset") +
  theme(text = element_text(size=12), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"), axis.text.x = element_text(angle = 30, hjust = 1))
ggsave("/path/to/BLSF_RF_NestedCVResults_subset2.pdf", width=5, height=4, units="in")
ggsave("/path/to/BLSF_RF_NestedCVResults_subset2.png", width=5, height=4, units="in")

# For subset into 4
files <- dir(path = pth,
             pattern = glob2rx("*Results*of4.csv"))

subset_4 <- tibble::tibble(filename = files) %>%
  dplyr::mutate(file_contents = purrr::map(filename,
                                           ~ read_csv(file.path(pth, .)))) %>%
  tidyr::unnest() %>%
  dplyr::mutate(subset = stringr::str_remove(filename, "BLSF_RF_NestedCVResults_subset")) %>%
  dplyr::mutate(subset = stringr::str_remove(subset, ".csv")) %>%
  dplyr::bind_rows(All)
subset_4$subset <- factor(subset_4$subset,levels = c("all", "1of4", "2of4", "3of4", "4of4"))

# Dot Plot with Mean + 95% CI indicated
ggplot(data = subset_4,  mapping = aes(x=subset, y=test_accuracy)) + 
  ylim(0, 1) +
  geom_dotplot(binaxis='y', stackdir = 'center', dotsize = 0.7) + 
  stat_summary(fun.y = mean, geom = "errorbar", aes(ymax = ..y.., ymin = ..y..),
               width = 0.75, size = 1, linetype = "solid", colour = "red") +  
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar",
               width = (0.75/2), size = 0.5, linetype = "solid", colour = "red") +
  labs(y = "Accuracy", x = "Subset") +
  theme(text = element_text(size=12), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"), axis.text.x = element_text(angle = 30, hjust = 1))
ggsave("/path/to/BLSF_RF_NestedCVResults_subset4.pdf", width=5, height=4, units="in")
ggsave("/path/to/BLSF_RF_NestedCVResults_subset4.png", width=5, height=4, units="in")

# For subset into 10
files <- dir(path = pth,
             pattern = glob2rx("*Results*of10.csv"))

subset_10 <- tibble::tibble(filename = files) %>%
  dplyr::mutate(file_contents = purrr::map(filename,
                                           ~ read_csv(file.path(pth, .)))) %>%
  tidyr::unnest() %>%
  dplyr::mutate(subset = stringr::str_remove(filename, "BLSF_RF_NestedCVResults_subset")) %>%
  dplyr::mutate(subset = stringr::str_remove(subset, ".csv")) %>%
  dplyr::bind_rows(All)
subset_10$subset <- factor(subset_10$subset,levels = c("all", "1of10", "2of10", "3of10", "4of10", "5of10", "6of10", "7of10", "8of10", "9of10", "10of10"))

# Dot Plot with Mean + 95% CI indicated
ggplot(data = subset_10,  mapping = aes(x=subset, y=test_accuracy)) + 
  ylim(0, 1) +
  geom_dotplot(binaxis='y', stackdir = 'center', dotsize = 0.7) + 
  stat_summary(fun.y = mean, geom = "errorbar", aes(ymax = ..y.., ymin = ..y..),
               width = 0.75, size = 1, linetype = "solid", colour = "red") +  
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar",
               width = (0.75/2), size = 0.5, linetype = "solid", colour = "red") +
  labs(y = "Accuracy", x = "Subset") +
  theme(text = element_text(size=12), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"), axis.text.x = element_text(angle = 30, hjust = 1))
ggsave("/path/to/BLSF_RF_NestedCVResults_subset10.pdf", width=5, height=4, units="in")
ggsave("/path/to/BLSF_RF_NestedCVResults_subset10.png", width=5, height=4, units="in")
