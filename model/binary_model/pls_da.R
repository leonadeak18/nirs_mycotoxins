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

# testing for PLS-DA
library(mixOmics)
library(spectacles)

# Main function
pipeline <- function(X_list, y_list, max_comp = 20, k_folds = 10, n_iter = 10) {
  
  final_results <- data.frame()
  distance_metrics <- c('max.dist', 'centroids.dist', 'mahalanobis.dist')
  data_types <- c('overall', 'BER')
  # loop through each dataset
  for (material in names(X_list)) {
     X_material <- X_list[[material]]
    for (toxin in names(y_list[[material]])){
      y_toxin <- y_list[[material]][[toxin]]
      dataset_name <- paste("Dataset", material , "with Ref", toxin)

      ## Loop through each preprocessing method
      prep_methods <- list(
          'Raw' = function(x) {
              savitzkyGolay(x, w = 15, p = 2, m = 0)},
          'SNV' = function(x){
              standardNormalVariate(savitzkyGolay(x, w = 15, p = 2, m = 0))
              },
          '1st_Der' = function(x){
              savitzkyGolay(x, w = 15, p = 2, m = 1)
              },
          '2nd_Der' = function(x){
              savitzkyGolay(x, w = 15, p = 2, m = 2)
          },
          'EMSC' = function(x){
              df <- EMSC(savitzkyGolay(x, w = 15, p = 2, m = 0), degree = 2)
              return(df$corrected)
          },
          "RNV" = function(x){
              rnv(x, r = 0.25)
          }
      )
      for (prep_name in names(prep_methods)){
        X_prep <- prep_methods[[prep_name]](X_material)
        
        # Execute PLS-DA with cross-validation
        plsda_model <- plsda(X_prep, y_toxin, ncomp = max_comp, max.iter = 1000, near.zero.var = TRUE)
        perf_plsda <- perf(plsda_model, validation = 'Mfold', folds = k_folds, 
                            progressBar = TRUE, nrepeat = n_iter)
        print(paste(dataset_name, "with Preprocessing:", prep_name, "is done"))
        # Store results
        for (metric in distance_metrics){
          for (type in data_types){
            final_results <- rbind(final_results, data.frame(  
              Dataset = dataset_name,
              Preprocessing = prep_name,
              distance_metrics = metric,
              data_types = type,
              min_ncomp = perf_plsda$choice.ncomp[, metric][type],
              error_rate = perf_plsda$error.rate[[type]][perf_plsda$choice.ncomp[, metric][type], metric],
              error_rate_sd = perf_plsda$error.rate.sd[[type]][perf_plsda$choice.ncomp[, metric][type], metric],
              n_repeats = n_iter
            ))
            }
          }

      }


    }
  }  
  print("Pipeline execution completed!")
  return(final_results)
}


don_plsda <- pipeline(X_list = spectra[["train"]], y_list = ref[["train"]], max_comp = 20, k_folds = 10, n_iter = 10)
write.csv(don_plsda, "plsda_results.csv", row.names = FALSE)
save.image("plsda_results.RData")
