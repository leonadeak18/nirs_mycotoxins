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
      three_level_transform(df, cutoff = 1000, uncertainty_percent = 0.5)
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
pipeline_3_splsda <- function(X_list, y_list, max_comp = 20, k_folds = 10, n_iter = 10, list_keepX = list_keepX) {
  #define cores for parallel processing
  n_cores <- round(parallel::detectCores() * 0.9, 0)
  #auto set correct parallel assignment for mixOmics perf function
  if (.Platform$OS.type == "unix") {
  
  parallel_param <- MulticoreParam(workers = n_cores)
  message("Linux/Mac detected: Using MulticoreParam with ", n_cores, " cores.")
  
  } else if (.Platform$OS.type == "windows") {
  
  # Triggered on Windows
  parallel_param <- SnowParam(workers = n_cores)
  message("Windows detected: Using SnowParam with ", n_cores, " cores.")
  
  } else {
  
  # Fallback for unsupported systems to prevent crashes
  parallel_param <- SerialParam()
  message("Unrecognized OS: Defaulting to sequential processing (1 core).")
  
  }

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
        
        # Execute sPLS-DA with cross-validation
        tune_splsda <- tune.splsda(X, y, ncomp = 20, max.iter = 1000, near.zero.var = TRUE, test.keepX = list.keepX, validation = 'Mfold', folds = 10, progressBar = TRUE, nrepeat = 10, BPPARAM = parallel_param, seed = 123)
        # Optimal number of components based on t-tests on the error rate
        ncomp <- tune_splsda$choice.ncomp$ncomp
        select.keepX <- tune_splsda$choice.keepX[1:ncomp] 
        splsda_model <- splsda(X, y, ncomp = ncomp, keepX = select.keepX) 
        perf_splsda <- perf(splsda_model, validation = 'Mfold', folds = k_folds, 
                           progressBar = TRUE, nrepeat = n_iter, BPPARAM = parallel_param, seed = 123)
        
        # message() is generally preferred over print() for progress updates in R loops
        message(paste(dataset_name, "- Preprocessing:", prep_name, "is done"))
        
        # Store results and predict on test set
        for (metric in distance_metrics) {
          for (type in data_types) {
            
            # choice.ncomp matrix
            opt_comp <- perf_splsda$choice.ncomp[type, metric]
            
            #Safety check: mixOmics occasionally returns NA for optimal components
            if (is.na(opt_comp)) {
              err_rate <- NA
              err_rate_sd <- NA
            } else {
              err_rate <- perf_splsda$error.rate[[type]][opt_comp, metric]
              err_rate_sd <- perf_splsda$error.rate.sd[[type]][opt_comp, metric]

              ## Predict on test set
              X_test_prep <- prep_methods[[prep_name]](X_test[[material]])
              prediction <- predict(splsda_model, X_test_prep, dist = metric)
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
list.keepX <- c(10, 50, seq(100, 1000, by = 100))
don_splsda <- pipeline_3_splsda(X_list = spectra, y_list = ref, max_comp = 20, k_folds = 10, n_iter = 10, list_keepX = list.keepX)

# Save the outputs
write.csv(don_splsda, "./results/3_splsda_results.csv", row.names = FALSE)
save.image("./results/3_splsda_results.RData")
load("./results/3_splsda_results.RData")
