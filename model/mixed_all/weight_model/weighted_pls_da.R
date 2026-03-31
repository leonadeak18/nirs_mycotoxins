# ==========================================
# PLS-DA MODEL OPTIMIZATION — pls package
# ==========================================

library(pls)
library(caret)       # for createFolds / confusionMatrix
library(pROC)        # for roc_curve and AUC

# ------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------

#' Compute balanced class weights (mirrors sklearn's class_weight='balanced')
compute_balanced_weights <- function(y) {
  classes     <- unique(y)
  n_total     <- length(y)
  n_classes   <- length(classes)
  class_counts <- table(y)
  
  # balanced formula: w_c = n_total / (n_classes * n_c)
  class_w <- n_total / (n_classes * class_counts)
  
  # assign per-sample weight
  sample_w <- as.numeric(class_w[as.character(y)])
  return(sample_w)
}

#' Apply sqrt(w) scaling to X and y  
#' Equivalent to weighted least squares inside PLS
apply_sqrt_weights <- function(X, y_numeric, sample_weights) {
  sqrt_w    <- sqrt(sample_weights)
  X_weighted <- X * sqrt_w                  # broadcast row-wise
  y_weighted <- y_numeric * sqrt_w
  return(list(X = X_weighted, y = y_weighted))
}

#' Youden's J threshold optimization
find_optimal_threshold <- function(y_true, scores) {
  roc_obj   <- roc(y_true, scores, quiet = TRUE)
  coords_df <- coords(roc_obj, "best", 
                       best.method = "youden", 
                       ret = c("threshold", "sensitivity", "specificity"))
  return(coords_df$threshold[1])
}

#' Evaluate binary predictions
evaluate_predictions <- function(y_true, y_pred, scores) {
  roc_obj     <- roc(y_true, scores, quiet = TRUE)
  auc_val     <- as.numeric(auc(roc_obj))
  cm          <- confusionMatrix(
                    factor(y_pred, levels = c(0,1)), 
                    factor(y_true, levels = c(0,1)),
                    positive = "1"
                  )
  accuracy    <- as.numeric(cm$overall["Accuracy"])
  sensitivity <- as.numeric(cm$byClass["Sensitivity"])
  specificity <- as.numeric(cm$byClass["Specificity"])
  
  return(list(
    AUC         = auc_val,
    Accuracy    = accuracy,
    Sensitivity = sensitivity,
    Specificity = specificity
  ))
}

# ------------------------------------------
# MAIN OPTIMIZATION LOOP
# ------------------------------------------

n_splits       <- 10
n_components_grid <- 2:20          # latent variables to search over
cv_folds       <- 10
set.seed(270725)

plsda_results <- list()
result_idx    <- 1

for (source in sources) {
  for (target in targets) {
    for (i in seq_len(n_splits)) {
      
      cat(sprintf("Evaluating: %s samples for %s (Split %d)\n", source, target, i))
      
      # --- Extract data for this split ---
      X_tr_spectra <- as.matrix(X_train[[source]][[i]])
      X_te_spectra <- as.matrix(X_test[[source]][[i]])
      
      y_tr <- ifelse(y_train_binary[[source]][[i]][[target]] == "HC", 1, 0)
      y_te <- ifelse(y_test_binary[[source]][[i]][[target]] == "HC", 1, 0)
      
      # --- Encode country ---
      country_tr <- as.numeric(as.factor(train_country[[source]][[i]])) - 1
      country_te <- as.numeric(as.factor(test_country[[source]][[i]])) - 1
      # Note: fit encoding on train, apply to test to avoid leakage
      country_levels <- levels(as.factor(train_country[[source]][[i]]))
      country_te     <- as.numeric(factor(test_country[[source]][[i]], 
                                           levels = country_levels)) - 1
      
      # --- Compute balanced class weights ---
      sample_weights <- compute_balanced_weights(y_tr)
      cat(sprintf(
        "  Class weights -> HC: %.3f | LC: %.3f\n",
        unique(sample_weights[y_tr == 1]),
        unique(sample_weights[y_tr == 0])
      ))
      
      for (pipe_name in names(preprocessing_pipelines)) {
        preprocessor <- preprocessing_pipelines[[pipe_name]]
        
        # Apply preprocessing to spectra only
        X_train_prep <- preprocessor$fit_transform(X_tr_spectra)
        X_test_prep  <- preprocessor$transform(X_te_spectra)
        
        for (fp_name in names(feature_pipelines)) {
          cat(sprintf("  Running -> Prep: %s | Feature: %s... ", pipe_name, fp_name))
          start_time <- proc.time()["elapsed"]
          
          result <- tryCatch({
            
            fp_method <- feature_pipelines[[fp_name]]
            
            # Apply feature extraction / selection
            if (!is.null(fp_method)) {
              X_tr_feat <- fp_method$fit_transform(X_train_prep, y_tr)
              X_te_feat <- fp_method$transform(X_test_prep)
            } else {
              X_tr_feat <- X_train_prep
              X_te_feat <- X_test_prep
            }
            
            # Skip if all features dropped
            if (ncol(X_tr_feat) == 0) {
              cat("Skipped (0 features selected).\n")
              return(NULL)
            }
            
            # Glue country column to spectra
            X_tr_final <- cbind(X_tr_feat, country = country_tr)
            X_te_final <- cbind(X_te_feat, country = country_te)
            
            # Cap n_components to data dimensions
            max_comp   <- min(max(n_components_grid),
                              nrow(X_tr_final) - 1,
                              ncol(X_tr_final))
            comp_grid  <- n_components_grid[n_components_grid <= max_comp]
            
            # ==========================================
            # STEP 1 — Find best n_components via CV
            # using WEIGHTED X and y
            # ==========================================
            folds      <- createFolds(y_tr, k = cv_folds, list = TRUE)
            cv_auc_per_ncomp <- numeric(length(comp_grid))
            
            for (comp_idx in seq_along(comp_grid)) {
              n_comp   <- comp_grid[comp_idx]
              fold_aucs <- numeric(cv_folds)
              
              for (fold_idx in seq_along(folds)) {
                val_idx  <- folds[[fold_idx]]
                tr_idx   <- setdiff(seq_len(nrow(X_tr_final)), val_idx)
                
                # Split fold
                X_fold_tr <- X_tr_final[tr_idx, , drop = FALSE]
                X_fold_val <- X_tr_final[val_idx, , drop = FALSE]
                y_fold_tr  <- y_tr[tr_idx]
                y_fold_val <- y_tr[val_idx]
                w_fold_tr  <- sample_weights[tr_idx]
                
                # Recompute weights inside fold (avoid leakage)
                w_fold_tr <- compute_balanced_weights(y_fold_tr)
                
                # Apply sqrt(w) weighting
                weighted  <- apply_sqrt_weights(X_fold_tr, 
                                                 y_fold_tr, 
                                                 w_fold_tr)
                
                # Fit PLS on weighted fold
                pls_fold  <- plsr(
                  y ~ X,
                  data       = list(y = weighted$y, X = weighted$X),
                  ncomp      = n_comp,
                  scale      = TRUE,
                  method     = "kernelpls"    # fast & numerically stable
                )
                
                # Predict on UNWEIGHTED validation fold
                val_scores <- predict(pls_fold, 
                                       newdata = list(X = X_fold_val), 
                                       ncomp   = n_comp)[,,1]
                
                # Guard against degenerate folds
                if (length(unique(y_fold_val)) < 2) {
                  fold_aucs[fold_idx] <- 0.5
                } else {
                  fold_aucs[fold_idx] <- as.numeric(
                    auc(roc(y_fold_val, as.numeric(val_scores), quiet = TRUE))
                  )
                }
              }
              cv_auc_per_ncomp[comp_idx] <- mean(fold_aucs)
            }
            
            # Select best n_components
            best_n_comp <- comp_grid[which.max(cv_auc_per_ncomp)]
            cat(sprintf("Best ncomp: %d | ", best_n_comp))
            
            # ==========================================
            # STEP 2 — Out-of-fold CV scores on full
            # training set for threshold optimisation
            # ==========================================
            oof_scores <- numeric(length(y_tr))
            
            for (fold_idx in seq_along(folds)) {
              val_idx   <- folds[[fold_idx]]
              tr_idx    <- setdiff(seq_len(nrow(X_tr_final)), val_idx)
              
              X_fold_tr  <- X_tr_final[tr_idx, , drop = FALSE]
              X_fold_val <- X_tr_final[val_idx, , drop = FALSE]
              y_fold_tr  <- y_tr[tr_idx]
              w_fold_tr  <- compute_balanced_weights(y_fold_tr)
              
              weighted   <- apply_sqrt_weights(X_fold_tr, y_fold_tr, w_fold_tr)
              
              pls_fold   <- plsr(
                y ~ X,
                data  = list(y = weighted$y, X = weighted$X),
                ncomp = best_n_comp,
                scale = TRUE,
                method = "kernelpls"
              )
              
              oof_scores[val_idx] <- predict(
                pls_fold,
                newdata = list(X = X_fold_val),
                ncomp   = best_n_comp
              )[,,1]
            }
            
            # ==========================================
            # STEP 3 — Threshold optimization (Youden's J)
            # ==========================================
            optimal_threshold <- find_optimal_threshold(y_tr, oof_scores)
            cv_preds <- as.integer(oof_scores >= optimal_threshold)
            
            # CV metrics
            cv_metrics <- evaluate_predictions(y_tr, cv_preds, oof_scores)
            
            # ==========================================
            # STEP 4 — Refit on FULL training set
            # and evaluate on test set
            # ==========================================
            w_full    <- compute_balanced_weights(y_tr)
            weighted_full <- apply_sqrt_weights(X_tr_final, 
                                                 y_tr, 
                                                 w_full)
            
            final_pls <- plsr(
              y ~ X,
              data   = list(y = weighted_full$y, X = weighted_full$X),
              ncomp  = best_n_comp,
              scale  = TRUE,
              method = "kernelpls"
            )
            
            # Predict on test set (unweighted — test set is never weighted)
            test_scores <- predict(
              final_pls,
              newdata = list(X = X_te_final),
              ncomp   = best_n_comp
            )[,,1]
            
            test_preds  <- as.integer(test_scores >= optimal_threshold)
            test_metrics <- evaluate_predictions(y_te, test_preds, test_scores)
            
            elapsed <- proc.time()["elapsed"] - start_time
            cat(sprintf("Done! (Test AUC: %.3f | CV Acc: %.3f)\n",
                        test_metrics$AUC, cv_metrics$Accuracy))
            
            # Return result row
            list(
              Sample_Source       = source,
              Target_Toxin        = target,
              Split               = i,
              Preprocessing       = pipe_name,
              Feature_Method      = fp_name,
              Best_n_components   = best_n_comp,
              Optimal_Threshold   = optimal_threshold,
              True_CV_AUC         = cv_metrics$AUC,
              True_CV_Accuracy    = cv_metrics$Accuracy,
              True_CV_Sensitivity = cv_metrics$Sensitivity,
              True_CV_Specificity = cv_metrics$Specificity,
              Test_AUC            = test_metrics$AUC,
              Test_Accuracy       = test_metrics$Accuracy,
              Test_Sensitivity    = test_metrics$Sensitivity,
              Test_Specificity    = test_metrics$Specificity,
              Time_Sec            = round(elapsed, 2)
            )
            
          }, error = function(e) {
            cat(sprintf("Failed. Error: %s\n", conditionMessage(e)))
            NULL
          })
          
          if (!is.null(result)) {
            plsda_results[[result_idx]] <- result
            result_idx <- result_idx + 1
          }
        }
      }
    }
  }
}

# ------------------------------------------
# COMBINE AND SAVE RESULTS
# ------------------------------------------
optimized_algorithm <- do.call(rbind, lapply(plsda_results, as.data.frame))
write.csv(optimized_algorithm,
          "./results/global_with_country/2_PLSDA_results.csv",
          row.names = FALSE)