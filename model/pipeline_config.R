load_nirs_data <- function(splits = 100, threshold = 1000, 
                           weighted = c("none", "prob", "triangle")[1],
                           toxins = c("DON", "Total_DON"),
                          material = c('intact', 'ground'), base_path = "./data/train_test_set_austrian/") {
  source("./utils/preparation.R")
  # --- Setup Metadata ---
  sources <- material
  targets <- toxins
  
  # Initialize nested lists to hold the data
  X_train <- list(); X_test <- list()
  y_tr_bin <- list(); y_te_bin <- list()
  train_country <- list(); test_country <- list()
  
  for (source in sources) {
    # Initialize sub-lists for each source
    X_train[[source]] <- list()
    X_test[[source]] <- list()
    y_tr_bin[[source]] <- list()
    y_te_bin[[source]] <- list()
    train_country[[source]] <- list()
    test_country[[source]] <- list()
    
    for (i in 1:splits) {
      split_num <- sprintf("%03d", i)
      # Load raw data
      X_train[[source]][[i]] <- read.csv(paste0(base_path, "X_", source, "_train_", split_num, ".csv"), row.names = 1)
      colnames(X_train[[source]][[i]]) <- sub("^X", "", colnames(X_train[[source]][[i]]))

      X_test[[source]][[i]]  <- read.csv(paste0(base_path, "X_", source, "_test_", split_num, ".csv"), row.names = 1)
      colnames(X_test[[source]][[i]]) <- sub("^X", "", colnames(X_test[[source]][[i]]))

      y_train_raw <- read.csv(paste0(base_path, "y_train_", split_num, ".csv"), row.names = 1)
      y_test_raw  <- read.csv(paste0(base_path, "y_test_", split_num, ".csv"), row.names = 1)
      
      # Transform to binary
      if (weighted == "prob") {
        y_tr_bin[[source]][[i]] <- binary_transform_with_prob(y_train_raw, threshold = threshold, targets = targets)
        y_te_bin[[source]][[i]] <- binary_transform_with_prob(y_test_raw, threshold = threshold, targets = targets)
      } else if (weighted == "triangle") {
        y_tr_bin[[source]][[i]] <- binary_transform_with_triangle(y_train_raw, threshold = threshold, targets = targets)
        y_te_bin[[source]][[i]] <- binary_transform_with_triangle(y_test_raw, threshold = threshold, targets = targets)
      } else {
        y_tr_bin[[source]][[i]] <- binary_transform(y_train_raw, threshold = threshold, targets = targets)
        y_te_bin[[source]][[i]] <- binary_transform(y_test_raw, threshold = threshold, targets = targets)
      }
    }
  }
  
  # Return everything as a named list
  return(list(
    X_train = X_train,
    X_test = X_test,
    y_train = y_tr_bin,
    y_test = y_te_bin
  ))
}

