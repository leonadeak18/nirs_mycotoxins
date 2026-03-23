# import all dataset
## import the spectra
spectra <- list(
  train = list(
    "intact" = read.csv("nirs/train_test_set/X_train_intact.csv"),
    "ground" = read.csv("nirs/train_test_set/X_train_ground.csv")
  ),
  test = list(
    "intact" = read.csv("nirs/train_test_set/X_test_intact.csv"),
    "ground" = read.csv("nirs/train_test_set/X_test_ground.csv")
   )
)
### format the spectra
source("./utils/preparation.R")
spectra <- lapply(spectra, function(material) {
  lapply(material, function(df) {
    prepare_df(df, 1100, 2500)
  })
})

## import the reference values
files <- list(
  train = list(
    intact = "nirs/train_test_set/y_train_intact.csv",
    ground = "nirs/train_test_set/y_train_ground.csv"
  ),
  test = list(
    intact = "nirs/train_test_set/y_test_intact.csv",
    ground = "nirs/train_test_set/y_test_ground.csv"
  )
)

ref <- lapply(files, function(material) {
  lapply(material, function(path) {
    df <- read.csv(path)
    list(
      DON = df[, c(1, 2)],
      Total_DON = df[, c(1, 3)]
    )
  })
})
### format the reference values
ref <- lapply(ref, function(df_type) {
  lapply(df_type, function(material) {
    lapply(material, function(df) {
      binary_transform(df, cutoff = 1000)
    })
  })
})
str(ref$train$intact$DON)

# Load necessary libraries
library(mixOmics)
library(spectacles)
library(prospectr)
library(EMSC)

# 1. Define preprocessing methods
prep_methods <- list(
  'Raw'     = function(x) savitzkyGolay(x, w = 15, p = 2, m = 0),
  'SNV'     = function(x) savitzkyGolay(standardNormalVariate(x), w = 15, p = 2, m = 0),
  '1st_Der' = function(x) savitzkyGolay(x, w = 15, p = 2, m = 1),
  '2nd_Der' = function(x) savitzkyGolay(x, w = 15, p = 2, m = 2),
  'EMSC'    = function(x) {
    df <- EMSC(x, degree = 2)
    temp <- df$corrected
    output <- savitzkyGolay(temp, w = 15, p = 2, m = 0)
    return(output)
  },
  "RNV"     = function(x) savitzkyGolay(rnv(x, r = 0.25), w = 15, p = 2, m = 0)
)

# Main function
pipeline_2_plsda <- function(X_list, y_list, max_comp = 20, k_folds = 10, n_iter = 10) {
  
  #define an empty list to store results
  results_list <- list() 
  distance_metrics <- c('max.dist', 'centroids.dist', 'mahalanobis.dist')
  data_types <- c('overall', 'BER')
  
  # Separate train and test datasets
  X_train <- X_list[["train"]]
  y_train <- y_list[["train"]]
  X_test <- X_list[["test"]]
  y_test <- y_list[["test"]]

  # Loop through each dataset
  for (material in names(X_train)) {
    X_material <- X_train[[material]]
    
    for (toxin in names(y_train[[material]])) {
      y_toxin_raw <- y_train[[material]][[toxin]]
      y_toxin <- as.factor(unlist(y_toxin_raw))
      dataset_name <- paste("Dataset", material, "with Ref", toxin)
      
      # Loop through each preprocessing method
      for (prep_name in names(prep_methods)) {
        X_prep <- prep_methods[[prep_name]](X_material)
        
        # Execute PLS-DA with cross-validation
        plsda_model <- plsda(X_prep, y_toxin, ncomp = max_comp, max.iter = 1000, near.zero.var = TRUE)
        perf_plsda <- perf(plsda_model, validation = 'Mfold', folds = k_folds, 
                           progressBar = TRUE, nrepeat = n_iter)
        
        # message() is generally preferred over print() for progress updates in R loops
        message(paste(dataset_name, "- Preprocessing:", prep_name, "is done"))
        
        # Store results and predict on test set
        for (metric in distance_metrics) {
          for (type in data_types) {
            
            # choice.ncomp matrix
            opt_comp <- perf_plsda$choice.ncomp[type, metric]
            
            #Safety check: mixOmics occasionally returns NA for optimal components
            if (is.na(opt_comp)) {
              err_rate <- NA
              err_rate_sd <- NA
            } else {
              err_rate <- perf_plsda$error.rate[[type]][opt_comp, metric]
              err_rate_sd <- perf_plsda$error.rate.sd[[type]][opt_comp, metric]

              ## Predict on test set
              X_test_prep <- prep_methods[[prep_name]](X_test[[material]])
              prediction <- predict(plsda_model, X_test_prep, dist = metric)
              test_results <- data.frame(prediction$class[[metric]][, opt_comp], 
                                         Truth = as.factor(unlist(y_test[[material]][[toxin]])))
              accuracy <- sum(test_results[,1] == test_results$Truth) / nrow(test_results)
              sensitivity <- sum(test_results[,1] == "HC" & test_results$Truth == "HC") / sum(test_results$Truth == "HC")
              specificity <- sum(test_results[,1] == "LC" & test_results$Truth == "LC") / sum(test_results$Truth == "LC")
            }
            
            # Append to list
            results_list[[length(results_list) + 1]] <- data.frame(  
              X = material,
              Y = toxin,
              Preprocessing = prep_name,
              distance_metrics = metric,
              data_types = type,
              min_ncomp = opt_comp,
              error_rate = err_rate,
              error_rate_sd = err_rate_sd,
              n_repeats = n_iter,
              test_accuracy = ifelse(is.na(opt_comp), NA, accuracy),
              test_sensitivity = ifelse(is.na(opt_comp), NA, sensitivity),
              test_specificity = ifelse(is.na(opt_comp), NA, specificity)
            )
          }
        }
      }
    }
  } 
  message("Pipeline execution completed!")
  
  # 5. Bind the list into a single data frame at the very end
  final_results <- do.call(rbind, results_list)
  return(final_results)
}

# Run the pipeline
don_plsda <- pipeline_2_plsda(X_list = spectra, y_list = ref, max_comp = 20, k_folds = 10, n_iter = 10)

# Save the outputs
write.csv(don_plsda, "./results/2_plsda_results.csv", row.names = FALSE)
save.image("./results/2_plsda_results.RData")
load("./results/2_plsda_results.RData")

# Execute PLS-DA with cross-validation
X <- spectra[["train"]][["intact"]]
y <- as.factor(unlist(ref[["train"]][["intact"]][["DON"]]))
library(mixOmics)
plsda_model <- plsda(X, y, ncomp = 20, max.iter = 1000, near.zero.var = TRUE)
perf_plsda <- perf(plsda_model, validation = 'Mfold', folds = 10, 
                           progressBar = TRUE, nrepeat = 3)
prediction_trial <- predict(plsda_model, spectra[["test"]][["intact"]], dist = "max.dist")        
trial <- data.frame(prediction_trial$class$max.dist[,12], Truth = as.factor(unlist(ref[["test"]][["intact"]][["DON"]])))
trial_auc <- auroc(prediction_trial)
