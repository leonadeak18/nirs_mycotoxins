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
  prepare_df(df, r1 = 1100, r2 = 2500)
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
distribution_intact <- as.factor(ifelse(y$intact$DON < 1000, "LC", "HC"))
distribution_ground <- as.factor(ifelse(y$ground$DON < 1000, "LC", "HC"))

## Creates all 10 splits at once
set.seed(270725)
split_indices_intact <- createDataPartition(distribution_intact, p = 0.8, times = 10)
split_indices_ground <- createDataPartition(distribution_ground, p = 0.8, times = 10)

## Initialize empty lists
train_spectra <- list(); test_spectra <- list()
train_y <- list(); test_y <- list()

## Loop through the 10 sets
for (i in 1:10) {
  # Extract specific indices for this iteration
  idx_i <- split_indices_intact[[i]]
  idx_g <- split_indices_ground[[i]]
  
  # --- Process Intact ---
  # Assuming spectra and y are lists containing an "intact" element
  train_spectra$intact[[i]] <- spectra$intact[idx_i, ]
  test_spectra$intact[[i]]  <- spectra$intact[-idx_i, ]
  
  train_y$intact[[i]]       <- y$intact[idx_i, ]
  test_y$intact[[i]]        <- y$intact[-idx_i, ]
  
  # --- Process Ground ---
  train_spectra$ground[[i]] <- spectra$ground[idx_g, ]
  test_spectra$ground[[i]]  <- spectra$ground[-idx_g, ]
  
  train_y$ground[[i]]       <- y$ground[idx_g, ]
  test_y$ground[[i]]        <- y$ground[-idx_g, ]
  
  # --- Export Files ---
  # Ensure the directory exists: dir.create("Austrian/data/splits/", recursive = TRUE, showWarnings = FALSE)
  
  # Export Intact
  write.csv(train_spectra$intact[[i]], paste0("Austrian/data/splits/X_train_intact_split_", i, ".csv"), row.names = TRUE)
  write.csv(test_spectra$intact[[i]],  paste0("Austrian/data/splits/X_test_intact_split_", i, ".csv"), row.names = TRUE)
  write.csv(train_y$intact[[i]],       paste0("Austrian/data/splits/y_train_intact_split_", i, ".csv"), row.names = TRUE)
  write.csv(test_y$intact[[i]],        paste0("Austrian/data/splits/y_test_intact_split_", i, ".csv"), row.names = TRUE)
  
  # Export Ground
  write.csv(train_spectra$ground[[i]], paste0("Austrian/data/splits/X_train_ground_split_", i, ".csv"), row.names = TRUE)
  write.csv(test_spectra$ground[[i]],  paste0("Austrian/data/splits/X_test_ground_split_", i, ".csv"), row.names = TRUE)
  write.csv(train_y$ground[[i]],       paste0("Austrian/data/splits/y_train_ground_split_", i, ".csv"), row.names = TRUE)
  write.csv(test_y$ground[[i]],        paste0("Austrian/data/splits/y_test_ground_split_", i, ".csv"), row.names = TRUE)
}
