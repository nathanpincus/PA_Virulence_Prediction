library(tidyverse)

tree_counts <- readr::read_csv("/path/to/BLSF_AGEs_RF_tree_counts.csv") %>%
  tidyr::gather(key = X1, value = n_trees)

ggplot(data = tree_counts, aes(x = n_trees)) + 
  scale_color_grey() +
  geom_histogram(binwidth = 10, boundary = 0, alpha = 0.85, fill = "#0D2D6C") +
  theme_classic() + ylab("Count") + xlab("Number of Trees")
ggsave("/path/to/trees_per_AGE_hist.pdf", width=5, height=4, units="in")
ggsave("/path/to/trees_per_AGE_hist.png", width=5, height=4, units="in")