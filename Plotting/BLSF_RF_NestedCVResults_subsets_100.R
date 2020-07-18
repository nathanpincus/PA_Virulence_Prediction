library(tidyverse)

# For subset into 100
pth <- "/path/to/"
files <- dir(path = pth,
             pattern = glob2rx("*Results*of100.csv"))

subset_100 <- tibble::tibble(filename = files) %>%
  dplyr::mutate(file_contents = purrr::map(filename,
                                           ~ read_csv(file.path(pth, .)))) %>%
  tidyr::unnest() %>%
  dplyr::mutate(subset = stringr::str_remove(filename, "BLSF_RF_NestedCVResults_subset")) %>%
  dplyr::mutate(subset = stringr::str_remove(subset, ".csv"))

summary_100 <- subset_100 %>%
  dplyr::group_by(subset) %>%
  dplyr::summarise(mean(test_accuracy)) %>%
  dplyr::ungroup()

mean(summary_100$`mean(test_accuracy)`)
length(which(summary_100$`mean(test_accuracy)` < 0.6))

# Dot Plot with Mean indicated
ggplot(data = summary_100,  mapping = aes(y=`mean(test_accuracy)`, x="")) + 
  ylim(0, 1) +
  geom_dotplot(binaxis='y', stackdir = 'center', dotsize = 0.7, colour = "darkblue", fill = "darkblue") + 
  stat_summary(fun.y = mean, geom = "errorbar", aes(ymax = ..y.., ymin = ..y..),
               width = 0.75, size = .75, linetype = "solid", colour = "red") +  
  labs(y = "Mean Nested Cross-Validaiton Accuracy", x = "Subsets") +
  theme(text = element_text(size=12), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"), axis.text.x = element_text(angle = 30, hjust = 1))
ggsave("/path/to/BLSF_RF_NestedCVResults_subset100.pdf", width=5, height=4, units="in")
ggsave("/path/to/BLSF_RF_NestedCVResults_subset100.png", width=5, height=4, units="in")
