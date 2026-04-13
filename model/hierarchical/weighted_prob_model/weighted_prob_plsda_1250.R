# import all dataset
source("./model/hierarchical/pipeline_config.R")
nirs_data <- load_nirs_data(weighted = "prob", threshold = 1250)

library(rchemo)
library(prospectr)
library(pROC)
library(PRROC)
library(plsVarSel)
library(spectacles)

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
pipeline_hierarchical_weighted_binary <- function(nirs_data, max_nlv = 25) {
  
  results_list <- list()
  preps <- names(prep_methods)
  varsel_names <- names(varsel_methods)
  materials <- c("intact", "ground")
  toxins <- "Total_DON"
  
  for (material in materials) {
    for (i in 1:10) {
      
      # Import the raw spectra data
      X_train_full_raw <- as.matrix(nirs_data$X_train[[material]][[i]])
      X_test_full_raw  <- as.matrix(nirs_data$X_test[[material]][[i]])
      
      # import the country origin
      y_country_train <- nirs_data$train_country[[material]][[i]]
      y_country_f <- as.factor(y_country_train)
      countries_available <- unique(y_country_train)
      
      # ==========================================
      # PHASE 1: LEVEL 1 MODEL (COUNTRY FILTER)
      # Fixed to 1st Derivative and nlv = 10
      # ==========================================
      # Apply 1st_Der explicitly for Level 1
      X_train_L1 <- prep_methods[["1st_Der"]](X_train_full_raw)
      X_test_L1  <- prep_methods[["1st_Der"]](X_test_full_raw)
      
      # Create a plsda model for country origin prediction
      mod_country <- plsrda(X = X_train_L1, y = y_country_f, nlv = 10)
      
      # Predict Test Set Countries
      pred_country_test <- predict(mod_country, X_test_L1)$pred
      
      # ==========================================
      # PHASE 2: LEVEL 2 MODELS (DON CLASSIFICATION)
      # ==========================================
      for (toxin in toxins) {
        # Import y_train and y_test for the current toxin and convert to binary classes
        y_train_toxin_raw <- unlist(nirs_data$y_train[[material]][[i]][[toxin]]$Class)
        y_train_toxin <- ifelse(y_train_toxin_raw == "HC", "1", "0")
        names(y_train_toxin) <- rownames(X_train_full_raw)
        
        y_test_toxin_raw <- unlist(nirs_data$y_test[[material]][[i]][[toxin]]$Class)
        y_test_toxin <- ifelse(y_test_toxin_raw == "HC", "1", "0")

        #weights for every samples
        w_train      <- as.numeric(unlist(nirs_data$y_train[[material]][[i]][[toxin]]$Weight))
        names(w_train) <- rownames(X_train_full_raw)
        
        for (prep_name in preps) {
          
          # Apply the spectral preprocessing for the train and test sets
          X_train_full <- prep_methods[[prep_name]](X_train_full_raw)
          X_test_full  <- prep_methods[[prep_name]](X_test_full_raw)
          
          for (varsel_name in varsel_names){
            # Initialize vectors to hold final predictions and scores
            final_test_classes <- rep(NA, nrow(X_test_full))
            final_test_scores  <- rep(NA, nrow(X_test_full))
            names(final_test_classes) <- rownames(X_test_full)
            
            sub_lvs <- c()
            sub_errs <- c()          
            for (country in countries_available) {
              
              # Train Country-Specific Toxin Model
              train_idx <- which(y_country_train == country)
              X_train_sub <- X_train_full[train_idx, , drop = FALSE]
              y_train_sub <- y_train_toxin[train_idx]
              w_train_sub <- w_train[train_idx]

              # Execute variable selection on training subset
              selected_vars <- varsel_methods[[varsel_name]](
                x = X_train_sub, 
                y = as.numeric(y_train_sub), 
                max_comp = max_nlv, 
                k_folds = 10)
              X_train_sub <- X_train_sub[, selected_vars, drop = FALSE]
            
              # Optimize Level 2 using your allsteps CV function
              res_toxin <- plsr_plsda_allsteps(
                X = X_train_sub, Y = y_train_sub, step = "nlvtest", nlv = max_nlv, 
                cvmethod = "kfolds", nbrep = 10, weights = w_train_sub,
                seed = 270725, nfolds = 10, criterion = "err", selection = "globalmin", prior = "unif",
                Xscaling = "none", Yscaling = "none", method = "plsrda"
              )
              
              opt_idx_toxin <- which(res_toxin$optimum == 1)[1]
              mod_toxin     <- plsrda(X = X_train_sub, y = y_train_sub, weights = w_train_sub, nlv = opt_idx_toxin)
              
              sub_lvs <- c(sub_lvs, opt_idx_toxin)
              sub_errs <- c(sub_errs, res_toxin$err_mean[[opt_idx_toxin]])
              
              # --- Hierarchical Routing ---
              # Use the predictions from our fixed Level 1 model!
              test_route_idx <- which(pred_country_test == country)
              
              if (length(test_route_idx) > 0) {
                X_test_routed <- X_test_full[test_route_idx, selected_vars, drop = FALSE]
                preds_routed  <- predict(mod_toxin, X_test_routed)
                
                final_test_classes[test_route_idx] <- as.character(preds_routed$pred)
                final_test_scores[test_route_idx]  <- preds_routed$posterior[, 2]
              }
            }
            # ==========================================
            # PHASE 3: EVALUATE ASSEMBLED PREDICTIONS
            # ==========================================
            y_test_f <- factor(y_test_toxin, levels = c("0", "1"))
            pred_f   <- factor(final_test_classes, levels = c("0", "1"))
            cm       <- table(Predicted = pred_f, Actual = y_test_f)
            
            tp <- if("1" %in% rownames(cm) && "1" %in% colnames(cm)) cm["1", "1"] else 0
            tn <- if("0" %in% rownames(cm) && "0" %in% colnames(cm)) cm["0", "0"] else 0
            fp <- if("1" %in% rownames(cm) && "0" %in% colnames(cm)) cm["1", "0"] else 0
            fn <- if("0" %in% rownames(cm) && "1" %in% colnames(cm)) cm["0", "1"] else 0
            
            acc  <- (tp + tn) / sum(cm)
            sens <- if((tp + fn) > 0) tp / (tp + fn) else 0
            spec <- if((tn + fp) > 0) tn / (tn + fp) else 0
            
            mcc_num <- (as.numeric(tp) * tn) - (as.numeric(fp) * fn)
            mcc_den <- sqrt(as.numeric(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc_val <- if(is.na(mcc_den) || mcc_den == 0) 0 else mcc_num / mcc_den
            
            scores_pos <- final_test_scores
            roc_obj <- pROC::roc(y_test_f, scores_pos, levels = c("0", "1"), direction = "<", quiet = TRUE)
            auc_val <- as.numeric(pROC::auc(roc_obj))
            
            best_coords <- pROC::coords(roc_obj, x = "best", best.method = "youden", ret = "threshold")
            opt_thresh  <- as.numeric(unlist(best_coords))[1] 
            
            scores_hc <- scores_pos[y_test_toxin == "1"]
            scores_lc <- scores_pos[y_test_toxin == "0"]
            
            if(length(scores_hc) > 0 && length(scores_lc) > 0) {
              pr_obj <- PRROC::pr.curve(scores.class0 = scores_hc, scores.class1 = scores_lc, curve = FALSE)
              pr_auc_val <- pr_obj$auc.integral
            } else {
              pr_auc_val <- NA 
            }
            
            res_row <- data.frame(
              Sample_source = material,
              Target_toxin = toxin,
              Preprocessing = prep_name,
              Variable_Selection = varsel_name,
              Split = i,
              Avg_SubModel_LV = round(mean(sub_lvs), 1),
              Avg_SubModel_Err_CV = round(mean(sub_errs), 4),
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
            cat(sprintf("Done: Fold %d | Mat: %s | Tox: %s | Prep: %s | VarSel: %s\n", i, material, toxin, prep_name, varsel_name))
          }
        }
      }
    }
  }
  
  final_df <- do.call(rbind, results_list)
  message("Hierarchical Pipeline completed!")
  return(final_df)
}

# --- Execution ---
weighted_plsda_results <- pipeline_hierarchical_weighted_binary(nirs_data)
write.csv(weighted_plsda_results, "./results/hierarchical/weighted_prob_plsda_1250_results.csv", row.names = FALSE)