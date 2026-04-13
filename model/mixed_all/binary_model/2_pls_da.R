# import all dataset
source("./model/hierarchical/pipeline_config.R")
nirs_data <- load_nirs_data(weighted = "none", threshold = 1000)

# Load necessary libraries
library(rchemo)
library(pROC)
library(prospectr)
library(spectacles)
library(PRROC)
library(plsVarSel)

# Define preprocessing methods
prep_methods <- list(
  'SNV'     = function(x) savitzkyGolay(standardNormalVariate(x), w = 15, p = 2, m = 0),
  'Detrend_SNV' = function(x) savitzkyGolay(detrend(x, wav = as.numeric(colnames(x))), w = 15, p = 2, m = 0),
  '1st_Der' = function(x) savitzkyGolay(x, w = 15, p = 2, m = 1),
  '2nd_Der' = function(x) savitzkyGolay(x, w = 15, p = 2, m = 2),
  "RNV"     = function(x) savitzkyGolay(rnv(x, r = 0.25), w = 15, p = 2, m = 0)
)

# Define variable selection methods
varsel_methods <- list(
  "None" = function(x, y, max_comp, k_folds){1:ncol(x)},
  "VIP" = function(x, y, max_comp, k_folds) {
      df <- filterPLSR(y, x, ncomp = max_comp, ncomp.opt = "minimum", validation = "CV", segments = k_folds, VIP.threshold = 1)
      return(df$VIP)
      },
  "SR" = function(x, y, max_comp, k_folds) {
      df <- filterPLSR(y, x, ncomp = max_comp, ncomp.opt = "minimum", validation = "CV", segments = k_folds, SR.threshold = 1)
      return(df$SR)
  }
)

pipeline_binary_plsda_sequential <- function(nirs_data, max_nlv = 25) {
  
  results_list <- list()
  preps <- names(prep_methods)
  varsel_names <- names(varsel_methods)  
  materials <- c("intact", "ground")
  toxins <- c("DON", "Total_DON")
  
  for (material in materials) {
    for (i in 1:10) { 
      for (toxin in toxins) {
        for (prep_name in preps) {
          
          # Data Cleaning and Binary transformation
          X_train_raw <- as.matrix(nirs_data$X_train[[material]][[i]])
          X_test_raw  <- as.matrix(nirs_data$X_test[[material]][[i]])
          
          ## Extract raw classes and strictly convert HC->1 and LC->0
          y_train_raw  <- unlist(nirs_data$y_train[[material]][[i]][[toxin]]$Class)
          y_train      <- ifelse(y_train_raw == "HC", "1", "0")
          names(y_train) <- rownames(X_train_raw)

          #w_train      <- as.numeric(unlist(nirs_data$y_train[[material]][[i]][[toxin]]$Weight))
          #names(w_train) <- rownames(X_train_raw)

          ## Convert test set to 1 and 0
          y_test_raw   <- unlist(nirs_data$y_test[[material]][[i]][[toxin]]$Class)
          y_test       <- ifelse(y_test_raw == "HC", "1", "0")
          
          # Apply Spectra Preprocessing
          X_train_prep <- prep_methods[[prep_name]](X_train_raw)
          X_test_prep  <- prep_methods[[prep_name]](X_test_raw)
          
          for (varsel_name in varsel-names){
            selected_vars <- varsel_methods[[varsel_name]](
              x = X_train_prep, 
              y = as.numeric(y_train), 
              max_comp = max_nlv, k_folds = 10)
          
            X_train_prep <- X_train_prep[, selected_vars, drop = FALSE]
            X_test_prep  <- X_test_prep[, selected_vars, drop = FALSE]
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
                      selection = "globalmin"
            )
            
            # extract the optimum nlv
            opt_idx <- which(res_nlv$optimum == 1)[1]
            error_cv    <- res_nlv$err_mean[[opt_idx]]
            error_cv_sd <- res_nlv$err_sd[[opt_idx]]
            
            # Final model & prediction on test set
            final_mod <- plsrda(X = X_train_prep, y = y_train, weights = NULL, nlv = opt_idx)
            preds     <- predict(final_mod, X_test_prep)
            
            # Binarized Matrix Extraction
            y_test_f <- factor(y_test, levels = c("0", "1"))
            pred_f   <- factor(preds$pred, levels = c("0", "1"))
            cm       <- table(Predicted = pred_f, Actual = y_test_f)
            
            # Safely extract 1s (Positives) and 0s (Negatives)
            tp <- if("1" %in% rownames(cm) && "1" %in% colnames(cm)) cm["1", "1"] else 0
            tn <- if("0" %in% rownames(cm) && "0" %in% colnames(cm)) cm["0", "0"] else 0
            fp <- if("1" %in% rownames(cm) && "0" %in% colnames(cm)) cm["1", "0"] else 0
            fn <- if("0" %in% rownames(cm) && "1" %in% colnames(cm)) cm["0", "1"] else 0
            
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
            roc_obj <- roc(y_test_f, scores_pos, levels = c("0", "1"), direction = "<", quiet = TRUE)
            auc_val <- as.numeric(auc(roc_obj))
            ## deteermine the optimal threshold
            best_coords <- coords(roc_obj, x = "best", best.method = "youden", ret = "threshold")
            opt_thresh  <- as.numeric(unlist(best_coords))[1] 
            
            # Determine the PR-Curve
            scores_hc <- scores_pos[y_test == "1"]
            scores_lc <- scores_pos[y_test == "0"]
            
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
              Target_toxin = toxin,
              Preprocessing = prep_name,
              Split = i,
              LV = opt_idx,
              Err_CV = round(error_cv, 4),
              Err_CV_SD = round(error_cv_sd, 4),
              TN = tn,
              FP = fp,
              FN = fn,
              TP = tp,
              Accuracy = round(acc, 4),
              Sensitivity = round(sens, 4),
              Specitifity = round(spec, 4),
              MCC = round(mcc_val, 4),
              ROC_AUC = round(auc_val, 4),
              PR_AUC = round(pr_auc_val, 4),
              Opt_Threshold = round(opt_thresh, 4)
            )
            
            results_list[[length(results_list) + 1]] <- res_row
            cat(sprintf("Done: Dataset %d| Material %s | Toxin %s | Prep %s | VarSel: %s\n", i, material, toxin, prep_name, varsel_name))
          }
        }
      }
    }
  }
  
  final_df <- do.call(rbind, results_list)
  message("Pipeline completed!")
  return(final_df)
}

# --- Execution ---
binary_plsda_results <- pipeline_binary_plsda_sequential(nirs_data)
write.csv(binary_plsda_results, "./results/mixed_all/2_plsda_results.csv", row.names = FALSE)
