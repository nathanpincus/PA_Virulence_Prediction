library(tidyverse)

# Load in table of nested CV results
f <- "/path/to/BLSF_RF_GSCV_8mers_NestedCVResults.csv"
p <- "/path/to/BLSF_RF_GSCV_8mers_NestedCVResults.pdf"
g <- "/path/to/BLSF_RF_GSCV_8mers_NestedCVResults.png"

CV_raw <- readr::read_csv(f) %>%
  dplyr::select(Accuracy = test_accuracy, Sensitivity = test_sensitivity, Specificity = test_specificity, PPV = test_PPV, AUC = test_AUC, F1 = test_f1)

CV <- readr::read_csv(f) %>%
  dplyr::select(Accuracy = test_accuracy, Sensitivity = test_sensitivity, Specificity = test_specificity, PPV = test_PPV, AUC = test_AUC, F1 = test_f1) %>%
  tidyr::gather(key = Statistic, value = Score)

# Mean accuracy
mean(CV_raw$Accuracy)

# Order variables
CV$Statistic <- factor(CV$Statistic,levels = c("Accuracy", "Sensitivity", "Specificity", "PPV", "AUC", "F1"))

# Dot Plot with Mean + 95% CI indicated
ggplot(data= CV, mapping = aes(x=Statistic, y=Score)) + 
  ylim(0, 1) +
  geom_dotplot(binaxis='y', stackdir = 'center', dotsize = 0.7) + 
  stat_summary(fun.y = mean, geom = "errorbar", aes(ymax = ..y.., ymin = ..y..),
               width = 0.75, size = 1, linetype = "solid", colour = "red") +  
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar",
               width = (0.75/2), size = 0.5, linetype = "solid", colour = "red") +
  labs(y = "Score", x = "Statistic", title = "8-mers") +
  theme(text = element_text(size=12), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"),
        plot.title = element_text(hjust = 0.5))
ggsave(p, width=5, height=4, units="in")
ggsave(g, width=5, height=4, units="in")
