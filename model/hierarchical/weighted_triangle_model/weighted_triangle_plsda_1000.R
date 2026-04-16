# import all dataset
source("./model/hierarchical/pipeline_config.R")
nirs_data <- load_nirs_data(weighted = "triangle", threshold = 1000)

library(rchemo)
library(prospectr)
library(pROC)
library(PRROC)
library(plsVarSel)
library(spectacles)

# Define preprocessing methods
prep_methods <- list(
  'SNV'         = function(x) savitzkyGolay(standardNormalVariate(x), w = 15, p = 2, m = 0),
  'Detrend_SNV' = function(x) savitzkyGolay(detrend(x, wav = as.numeric(colnames(x))), w = 15, p = 2, m = 0),
  '1st_Der'     = function(x) savitzkyGolay(x, w = 15, p = 2, m = 1),
  "RNV"         = function(x) savitzkyGolay(rnv(x, r = 0.25), w = 15, p = 2, m = 0)
)

# Define variable selection methods
varsel_methods <- list(
  "None" = function(x, y, max_comp, k_folds){1:ncol(x)},
  "VIP"  = function(x, y, max_comp, k_folds) {
      df <- filterPLSR(y, x, ncomp = max_comp, ncomp.opt = "minimum", validation = "CV", segments = k_folds, VIP.threshold = 1)
      return(df$VIP)
  },
  "SR"   = function(x, y, max_comp, k_folds) {
      df <- filterPLSR(y, x, ncomp = max_comp, ncomp.opt = "minimum", validation = "CV", segments = k_folds, SR.threshold = 1)
      return(df$SR)
  }
)

pipeline_hierarchical_weighted_binary <- function(nirs_data, max_nlv = 25) {
  
  results_list <- list()
  preps <- names(prep_methods)
  varsel_names <- names(varsel_methods)
  materials <- c("intact", "ground")
  toxins <- c("DON", "Total_DON")
  
  for (material in materials) {
    for (i in 1:10) {
      
      # Import the raw spectra data
      X_train_full_raw <- as.matrix(nirs_data$X_train[[material]][[i]])
      X_test_full_raw  <- as.matrix(nirs_data$X_test[[material]][[i]])
      
      # import the country origin for BOTH train and test sets
      y_country_train <- nirs_data$train_country[[material]][[i]]
      y_country_test  <- nirs_data$test_country[[material]][[i]]
      
      y_country_f <- as.factor(y_country_train)
      countries_available <- unique(y_country_train)
      
      # ==========================================
      # PHASE 1: LEVEL 1 MODEL (COUNTRY FILTER)
      # Fixed to 1st Derivative and nlv = 10
      # ==========================================
      X_train_L1 <- prep_methods[["1st_Der"]](X_train_full_raw)
      X_test_L1  <- prep_methods[["1st_Der"]](X_test_full_raw)
      
      # Create a plsda model for country origin prediction
      mod_country <- rchemo::plsrda(X = X_train_L1, y = y_country_f, nlv = 10)
      pred_country_test <- predict(mod_country, X = X_test_L1)$pred
      
      # ==========================================
      # PHASE 2: LEVEL 2 MODELS (DON CLASSIFICATION)
      # ==========================================
      for (toxin in toxins) {
        y_train_toxin_raw <- unlist(nirs_data$y_train[[material]][[i]][[toxin]]$Class)
        y_train_toxin <- ifelse(y_train_toxin_raw == "HC", "1", "0")
        names(y_train_toxin) <- rownames(X_train_full_raw)
        
        y_test_toxin_raw <- unlist(nirs_data$y_test[[material]][[i]][[toxin]]$Class)
        y_test_toxin <- ifelse(y_test_toxin_raw == "HC", "1", "0")

        # weights for every sample
        w_train      <- as.numeric(unlist(nirs_data$y_train[[material]][[i]][[toxin]]$Weight))
        names(w_train) <- rownames(X_train_full_raw)
        
        for (prep_name in preps) {
          
          # Apply the spectral preprocessing
          X_train_full <- prep_methods[[prep_name]](X_train_full_raw)
          X_test_full  <- prep_methods[[prep_name]](X_test_full_raw)
          
          for (varsel_name in varsel_names){
            
            final_test_classes <- rep(NA, nrow(X_test_full))
            final_test_scores  <- rep(NA, nrow(X_test_full))
            names(final_test_classes) <- rownames(X_test_full)
            
            # Map LVs and Errors specifically to the country
            sub_lvs <- list()
            sub_errs <- list()          
            
            for (country in countries_available) {
              
              # Train Country-Specific Toxin Model
              train_idx <- which(y_country_train == country)
              X_train_sub <- X_train_full[train_idx, , drop = FALSE]
              y_train_sub <- y_train_toxin[train_idx]
              w_train_sub <- w_train[train_idx]

              # Execute variable selection
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
              
              # Store metrics
              sub_lvs[[country]]  <- opt_idx_toxin
              sub_errs[[country]] <- res_toxin$err_mean[[opt_idx_toxin]]
              
              # --- Hierarchical Routing ---
              test_route_idx <- which(pred_country_test == country)
              
              if (length(test_route_idx) > 0) {
                X_test_routed <- X_test_full[test_route_idx, selected_vars, drop = FALSE]
                preds_routed  <- predict(mod_toxin, X = X_test_routed)
                
                final_test_classes[test_route_idx] <- as.character(preds_routed$pred)
                final_test_scores[test_route_idx]  <- preds_routed$posterior[, 2]
              }
            }
            
            # ==========================================
            # PHASE 3: DYNAMIC EVALUATION (OVERALL + PER COUNTRY)
            # ==========================================
            eval_groups <- c("Overall", countries_available)
            
            for (group in eval_groups) {
              
              if (group == "Overall") {
                idx <- 1:length(y_test_toxin)
                group_lv  <- round(mean(unlist(sub_lvs)), 1)
                group_err <- round(mean(unlist(sub_errs)), 4)
              } else {
                idx <- which(y_country_test == group)
                group_lv  <- sub_lvs[[group]]
                group_err <- round(sub_errs[[group]], 4)
              }
              
              # Skip if a specific subset is empty for this split
              if (length(idx) == 0) next
              
              y_test_f_sub <- factor(y_test_toxin[idx], levels = c("0", "1"))
              pred_f_sub   <- factor(final_test_classes[idx], levels = c("0", "1"))
              scores_pos_sub <- final_test_scores[idx]
              
              cm <- table(Predicted = pred_f_sub, Actual = y_test_f_sub)
              
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
              
              # Safe ROC execution
              if (length(unique(y_test_toxin[idx])) > 1) {
                roc_obj <- pROC::roc(y_test_f_sub, scores_pos_sub, levels = c("0", "1"), direction = "<", quiet = TRUE)
                auc_val <- as.numeric(pROC::auc(roc_obj))
                best_coords <- pROC::coords(roc_obj, x = "best", best.method = "youden", ret = "threshold")
                opt_thresh  <- as.numeric(unlist(best_coords))[1] 
              } else {
                auc_val <- NA
                opt_thresh <- NA
              }
              
              scores_hc <- scores_pos_sub[y_test_toxin[idx] == "1"]
              scores_lc <- scores_pos_sub[y_test_toxin[idx] == "0"]
              
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
                Evaluation_Group = group,
                Model_LV = group_lv,
                Model_Err_CV = group_err,
                TN = tn, 
                FP = fp, 
                FN = fn, 
                TP = tp,
                Accuracy = round(acc, 4),
                Sensitivity = round(sens, 4),
                Specitifity = round(spec, 4),
                F1_Score = round(2 * (sens * spec) / (sens + spec), 4),
                MCC = round(mcc_val, 4),
                ROC_AUC = round(auc_val, 4),
                PR_AUC = round(pr_auc_val, 4),
                Opt_Threshold = round(opt_thresh, 4)
              )
              
              results_list[[length(results_list) + 1]] <- res_row
            }
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
write.csv(weighted_plsda_results, "./results/hierarchical/weighted_triangle_plsda_1000_results.csv", row.names = FALSE)