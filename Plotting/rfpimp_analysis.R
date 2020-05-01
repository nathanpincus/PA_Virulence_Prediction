library(tidyverse)

# Import importances from each rfpimp run
files <- dir(path = "/path/to/", pattern = "*.csv")

imp <- tibble::tibble(filename = files) %>%
  dplyr::mutate(file_contents = purrr::map(filename,
                                           ~ read_csv(file.path("/path/to/", .))))


imp2 <- tidyr::unnest(imp)

# Make table of average importance for each variable - sort by increasing importance
avg_imp <- imp2 %>%
  dplyr::group_by(Feature) %>%
  summarise(mean(Importance)) %>%
  dplyr::arrange(desc(`mean(Importance)`))

# For the top 10 variables - show dotplot of importances
top10 <- avg_imp$Feature[1:10]
imp_top10 <- imp2 %>%
  dplyr::filter(Feature %in% top10)

# Order variables
imp_top10$Feature <- factor(imp_top10$Feature,levels = top10)

# Dot Plot with Mean + 95% CI indicated
ggplot(data = imp_top10, mapping = aes(x=Feature, y=Importance)) + 
  geom_dotplot(binaxis='y', stackdir = 'center', dotsize = 0.05, stackratio = 1) + 
  stat_summary(fun.y = mean, geom = "errorbar", aes(ymax = ..y.., ymin = ..y..),
               width = 0.75, size = 1, linetype = "solid", colour = "red") +  
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar",
               width = (0.75/2), size = 0.5, linetype = "solid", colour = "red") +
  labs(y = "Permutation Importance", x = "AGE") +
  theme(text = element_text(size=12), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"), axis.text.x = element_text(angle = 30, hjust = 1))
ggsave("/path/to/OOB_Permutation_Imp_100x.pdf", width=5, height=4, units="in")
ggsave("/path/to/OOB_Permutation_Imp_100x.png", width=5, height=4, units="in")


