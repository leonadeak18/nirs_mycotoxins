## Import dataset
library(readxl)
data <- as.data.frame(read_excel("Austrian/data/nirs_intact.xlsx"))
row.names(data) <- data$Sample_ID
#split into y and spectra
spectra <- list(
  "intact" = as.data.frame(data[,c(1, 13:ncol(data))]),
  "ground" = as.data.frame(read_excel("Austrian/data/nirs_milled.xlsx"))
)
# remove the wavelengths to become 1100 - 2500 nm
source("Austrian/utils/preparation.R")
spectra <- lapply(spectra, function(df) {
  prepare_df(df, r1 = 780, r2 = 2500)
})

y_raw <- cbind(data['Deoxynivalenol'], data['Total_DON'])
colnames(y_raw) <- c("DON", "Total_DON")

#Outlier detection
library(mdatools)
outlier_intact <- pca(spectra$intact, ncomp = 3)
plotCumVariance(outlier_intact, show.label = TRUE) ## 3 components explain > 99% of variance
plotDistances(outlier_intact, show.label = TRUE) ## outlier on 99631 and 99632

outlier_ground <- pca(spectra$ground, ncomp = 3)
plotCumVariance(outlier_ground, show.label = TRUE) ## 3 components explain > 99% of variance
plotDistances(outlier_ground, show.label = TRUE) ## outlier on 99022 and 99098

#remove the outliers
spectra$intact <- spectra$intact[!rownames(spectra$intact) %in% c("99631", "99632"), ]
spectra$ground <- spectra$ground[!rownames(spectra$ground) %in% c("99022", "99098"), ]
y <- list(
  intact = y_raw[!rownames(y_raw) %in% c("99631", "99632"), ],
  ground = y_raw[!rownames(y_raw) %in% c("99022", "99098"), ]
)

library(caret)

## Create the stratification factor
distribution <- as.factor(ifelse(y$intact$DON < 1000, "LC", "HC"))

## Creates all 10 splits at once
set.seed(270725)
split_indices <- createDataPartition(distribution, p = 0.8, times = 10)
## Initialize empty lists
train_spectra <- list()
test_spectra <- list()
train_y <- list()
test_y <- list()

## Loop through the 10 sets of indices
for (i in 1:10) {
  idx <- split_indices[[i]] 
  
  # Use idx for training, and -idx (everything else) for testing
  train_spectra[[i]] <- lapply(spectra, function(df) df[idx, ])
  test_spectra[[i]]  <- lapply(spectra, function(df) df[-idx, ])
  
  train_y[[i]] <- lapply(y, function(df) df[idx, ])
  test_y[[i]]  <- lapply(y, function(df) df[-idx, ])
}

## Export the splits as csv files
for (i in 1:10) {
  write.csv(train_spectra[[i]]$intact, paste0("Austrian/data/splits_2/X_train_intact_split_", i, ".csv"), row.names = TRUE)
  write.csv(test_spectra[[i]]$intact, paste0("Austrian/data/splits_2/X_test_intact_split_", i, ".csv"), row.names = TRUE)
  
  write.csv(train_spectra[[i]]$ground, paste0("Austrian/data/splits_2/X_train_ground_split_", i, ".csv"), row.names = TRUE)
  write.csv(test_spectra[[i]]$ground, paste0("Austrian/data/splits_2/X_test_ground_split_", i, ".csv"), row.names = TRUE)
  
  write.csv(train_y[[i]]$ground, paste0("Austrian/data/splits_2/y_train_ground_split_", i, ".csv"), row.names = TRUE)
  write.csv(test_y[[i]]$ground, paste0("Austrian/data/splits_2/y_test_ground_split_", i, ".csv"), row.names = TRUE)

  write.csv(train_y[[i]]$intact, paste0("Austrian/data/splits_2/y_train_intact_split_", i, ".csv"), row.names = TRUE)
  write.csv(test_y[[i]]$intact, paste0("Austrian/data/splits_2/y_test_intact_split_", i, ".csv"), row.names = TRUE)
}
