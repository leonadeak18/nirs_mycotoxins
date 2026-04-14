# import all dataset
source("./model/hierarchical/pipeline_config.R")
nirs_data <- load_nirs_data(weighted = "none")

library(rchemo)
library(prospectr)
library(pROC)
library(PRROC)

country_plsda <- function(nirs_data, max_nlv = 25) {
  
  results_list <- list()
    
  materials <- c("intact", "ground")
  
  for (material in materials) {
    for (i in 1:10) {     
      # Data Cleaning and Binary transformation
      X_train_raw <- as.matrix(nirs_data$X_train[[material]][[i]])
      X_test_raw  <- as.matrix(nirs_data$X_test[[material]][[i]])
      
      ## take the country origin as the target variable
      y_train  <- unlist(nirs_data$train_country[[material]][[i]])
      names(y_train) <- rownames(X_train_raw)
      y_test   <- unlist(nirs_data$test_country[[material]][[i]])
      names(y_test) <- rownames(X_test_raw)
      
      # Apply Spectra Preprocessing
      X_train_prep <- savitzkyGolay(X_train_raw, w = 15, p = 2, m = 1)
      X_test_prep  <- savitzkyGolay(X_test_raw, w = 15, p = 2, m = 1)
      
      # Optimize the nLV through CV with k =10 for 10-times
      res_nlv <- plsr_plsda_allsteps(
                X = X_train_prep, Xname = NULL, 
                Xscaling = "none", 
                Y = y_train, Yscaling = "none", weights = NULL,
                newX = NULL, newXname = NULL,
                
                method = "plsrda",
                prior = "unif",
                step = "nlvtest",
                nlv = max_nlv, 
                modeloutput = c("scores","loadings","coef","vip"), 
                
                cvmethod = "kfolds", 
                nbrep = 10, 
                seed = 270725, 
                nfolds = 10, 
                criterion = "err", 
                selection = "localmin"
      )
      
      # extract the optimum nlv
      opt_idx <- which(res_nlv$optimum == 1)[1]
      error_cv    <- res_nlv$err_mean[[opt_idx]]
      error_cv_sd <- res_nlv$err_sd[[opt_idx]]
      
      # Final model & prediction on test set
      final_mod <- plsrda(X = X_train_prep, y = y_train, weights = NULL, nlv = opt_idx)
      preds     <- predict(final_mod, X_test_prep)
      
      # Binarized Matrix Extraction
      y_test_f <- factor(y_test, levels = c("Austria", "France"))
      pred_f   <- factor(preds$pred, levels = c("Austria", "France"))
      cm       <- table(Predicted = pred_f, Actual = y_test_f)
      
      # Safely extract 1s (Positives) and 0s (Negatives)
      tp <- if("France" %in% rownames(cm) && "France" %in% colnames(cm)) cm["France", "France"] else 0
      tn <- if("Austria" %in% rownames(cm) && "Austria" %in% colnames(cm)) cm["Austria", "Austria"] else 0
      fp <- if("France" %in% rownames(cm) && "Austria" %in% colnames(cm)) cm["France", "Austria"] else 0
      fn <- if("Austria" %in% rownames(cm) && "France" %in% colnames(cm)) cm["Austria", "France"] else 0
      
      # determine the accuracy, sensitivity, and specitifity
      acc  <- (tp + tn) / sum(cm)
      sens <- if((tp + fn) > 0) tp / (tp + fn) else 0
      spec <- if((tn + fp) > 0) tn / (tn + fp) else 0
      
      #determine the MCC
      mcc_num <- (as.numeric(tp) * tn) - (as.numeric(fp) * fn)
      mcc_den <- sqrt(as.numeric(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
      mcc_val <- if(is.na(mcc_den) || mcc_den == 0) 0 else mcc_num / mcc_den
      
      # AUC-ROC & Optimal Threshold 
      # Extract column 2 as our positive class scores.
      scores_pos <- preds$posterior[, 2]
      
      # Force the direction to "<" so lower will predicts 0 and higher to predict 1
      roc_obj <- roc(y_test_f, scores_pos, levels = c("Austria", "France"), direction = "<", quiet = TRUE)
      auc_val <- as.numeric(auc(roc_obj))
      ## deteermine the optimal threshold
      best_coords <- coords(roc_obj, x = "best", best.method = "youden", ret = "threshold")
      opt_thresh  <- as.numeric(unlist(best_coords))[1] 
      
      # Determine the PR-Curve
      scores_hc <- scores_pos[y_test == "France"]
      scores_lc <- scores_pos[y_test == "Austria"]
      
      if(length(scores_hc) > 0 && length(scores_lc) > 0) {
        ## PRROC package demand scores.class0 for positive
        pr_obj <- pr.curve(scores.class0 = scores_hc, scores.class1 = scores_lc, curve = FALSE)
        pr_auc_val <- pr_obj$auc.integral
      } else {
        pr_auc_val <- NA 
      }
      
      # Store results
      res_row <- data.frame(
        Sample_source = material,
        Split = i,
        LV = opt_idx,
        Err_CV = round(error_cv, 4),
        Err_CV_SD = round(error_cv_sd, 4),
        Accuracy = round(acc, 4),
        Sensitivity = round(sens, 4),
        Specitifity = round(spec, 4),
        MCC = round(mcc_val, 4),
        ROC_AUC = round(auc_val, 4),
        PR_AUC = round(pr_auc_val, 4),
        Opt_Threshold = round(opt_thresh, 4)
      )
      
      results_list[[length(results_list) + 1]] <- res_row
      cat(sprintf("Done: Dataset %d| Material %s\n", i, material))
    }
  }
  
  final_df <- do.call(rbind, results_list)
  message("Pipeline completed!")
  return(final_df)
}

# --- Execution ---
country_plsda_results <- country_plsda(nirs_data)
write.csv(country_plsda_results, "./results/hierarchical/country_local_plsda_results.csv", row.names = FALSE)
