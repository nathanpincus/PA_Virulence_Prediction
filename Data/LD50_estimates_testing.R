#Load drc package
library(drc)

#Read in input data for all experiments (including dose, starting mice, surviving mice)
input <- read.csv("/path/to/PA_mouse_experiments_testing.csv", header = TRUE)

#make a list of all unique strains
strains <- unique(input$Strain)

#Create a data frame to hold a summary of all experiments for each strain
mort <- as.data.frame(matrix(nrow=length(strains),ncol=3))
rownames(mort) <- strains
colnames(mort) <- c("dose","total","dead")

#Initialize empty lists for dose, total mice, and mortalities.  This will be a list of vectors, where each vector is the corresponding data for all experiments with a given strain
d <- list()
tot <- list()
m <- list()

# Iterate through each unique strain
# Create a sub-table including just the experiments with that strain
# extract all dose, total mice, and mortality data from these strains to the above vectors
for(strain in strains){
  print(strain)
 # print(which(input$Strain == strain))
  t <- input[which(input$Strain == strain),]
  d[[strain]] <- t[,3]
  tot[[strain]] <- t[,4]
  m[[strain]] <- t[,5]
}

# Use the dose, total mice, and mortality lists to fill the summary data frame
mort$dose <- d
mort$total <- tot
mort$dead <- m

#Create a data frame to hold LD50 and SD for each strain
LD50s <- as.data.frame(matrix(nrow=length(strains),ncol=2))
rownames(LD50s) <- strains
colnames(LD50s) <- c("LD50","SD")

# For each strain extract the dose, total mice, and mortality information from the mort table
# Try to fit to bionomial distribution using drc (drm) - if possible put in LD50s table
for(strain in strains){
  doses <- unlist(mort[strain,"dose"])
  total_mice <- unlist(mort[strain,"total"])
  mortalities <- unlist(mort[strain,"dead"])
  print(strain)
  try({
    model <- drm(mortalities/total_mice ~ doses, weights = total_mice, fct = LL.2(), type = "binomial")
    LD50s[strain,"LD50"] <- ED(model,50)[1]
    LD50s[strain,"SD"] <- ED(model,50)[2]
  })
}

LD50s$rounded <- round(LD50s$LD50, digits = 1)
write.csv(LD50s, file="/path/to/LD50_estimates_testing.csv")
