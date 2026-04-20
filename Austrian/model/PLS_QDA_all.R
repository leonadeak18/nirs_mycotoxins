# Set a global seed for the entire script
set.seed(270725)

# Import pipeline config
source("./Austrian/model/pipeline_config.R")

# Load necessary libraries
library(rchemo)
library(pROC)
library(prospectr)
library(spectacles)
library(PRROC)
library(plsVarSel)

# Define preprocessing methods
prep_methods <- list(
  'SNV'         = function(x) savitzkyGolay(standardNormalVariate(x), w = 15, p = 2, m = 0),
  'Detrend_SNV' = function(x) savitzkyGolay(detrend(x, wav = as.numeric(colnames(x))), w = 15, p = 2, m = 0),
  '1st_Der'     = function(x) savitzkyGolay(x, w = 15, p = 2, m = 1),
  "RNV"         = function(x) savitzkyGolay(rnv(x, r = 0.25), w = 15, p = 2, m = 0)
)

# Define variable selection methods
varsel_methods <- list(
  "None" = function(x, y, max_comp, k_folds, cur_seed){ 1:ncol(x) },
  "VIP"  = function(x, y, max_comp, k_folds, cur_seed) {
      set.seed(cur_seed)
      df <- filterPLSR(y, x, ncomp = max_comp, ncomp.opt = "minimum", validation = "CV", segments = k_folds, VIP.threshold = 1)
      return(df$VIP)
  },
  "SR"   = function(x, y, max_comp, k_folds, cur_seed) {
      set.seed(cur_seed)
      df <- filterPLSR(y, x, ncomp = max_comp, ncomp.opt = "minimum", validation = "CV", segments = k_folds, SR.threshold = 1)
      return(df$SR)
  }
)

# Refactored pipeline function handling both weighted/unweighted and toxin targets
pipeline_binary_plsqda_sequential <- function(nirs_data, target_toxins, use_weights = FALSE, max_nlv = 25) {
  
  results_list <- list()
  preps <- names(prep_methods)
  varsel_names <- names(varsel_methods)  
  materials <- c("intact", "ground")
  
  for (material in materials) {
    for (i in 1:10) { 
      
      # Create a unique but perfectly reproducible seed for this specific split
      current_seed <- 270725 + i
      
      for (toxin in target_toxins) {
        for (prep_name in preps) {
          
          X_train_raw <- as.matrix(nirs_data$X_train[[material]][[i]])
          X_test_raw  <- as.matrix(nirs_data$X_test[[material]][[i]])
          
          y_train_raw  <- unlist(nirs_data$y_train[[material]][[i]][[toxin]]$Class)
          y_train      <- ifelse(y_train_raw == "HC", "1", "0")
          names(y_train) <- rownames(X_train_raw)

          # Conditionally assign weights
          if (use_weights) {
            w_train <- as.numeric(unlist(nirs_data$y_train[[material]][[i]][[toxin]]$Weight))
            names(w_train) <- rownames(X_train_raw)
          } else {
            w_train <- NULL
          }

          y_test_raw   <- unlist(nirs_data$y_test[[material]][[i]][[toxin]]$Class)
          y_test       <- ifelse(y_test_raw == "HC", "1", "0")
          
          X_train_prep <- prep_methods[[prep_name]](X_train_raw)
          X_test_prep  <- prep_methods[[prep_name]](X_test_raw)
          
          for (varsel_name in varsel_names){
            
            # Pass the dynamic seed to the variable selection
            selected_vars <- varsel_methods[[varsel_name]](
              x = X_train_prep, 
              y = as.numeric(y_train), 
              max_comp = max_nlv, k_folds = 10, cur_seed = current_seed)
          
            X_train_prep_sel <- X_train_prep[, selected_vars, drop = FALSE]
            X_test_prep_sel  <- X_test_prep[, selected_vars, drop = FALSE]

            # --- NEW: INITIAL CAP ---
            # Start with a reasonable ceiling based on overall class sizes
            min_cv_samples <- floor(min(table(y_train)) * 0.9)
            current_max_nlv <- min(max_nlv, min_cv_samples - 2, length(selected_vars))
            
            if (current_max_nlv < 1) {
              cat(sprintf("Skipped: VarSel %s left too few variables or samples for QDA.\n", varsel_name))
              next 
            }
            
            # --- NEW: THE TRY-CATCH BACKOFF LOOP ---
            success <- FALSE
            res_nlv <- NULL
            
            while (!success && current_max_nlv >= 1) {
              
              # Attempt to run the Cross-Validation
              res_nlv <- tryCatch({
                plsr_plsda_allsteps(
                  X = X_train_prep_sel, Xname = NULL, 
                  Xscaling = "none", 
                  Y = y_train, Yscaling = "none", weights = w_train,
                  newX = NULL, newXname = NULL,
                  method = "plsqda",
                  prior = "unif",
                  step = "nlvtest",
                  nlv = current_max_nlv, 
                  modeloutput = c("scores","loadings","coef","vip"), 
                  cvmethod = "kfolds", 
                  nbrep = 10, 
                  seed = current_seed, 
                  nfolds = 10, 
                  criterion = "err", 
                  selection = "globalmin"
                )
              }, error = function(e) {
                # If it crashes, return the error object instead of stopping the script
                return(e)
              })
              
              # Check if the result was an error (like our singular matrix crash)
              if (inherits(res_nlv, "error")) {
                # Silently step down the Latent Variables by 1 and try again
                current_max_nlv <- current_max_nlv - 1
              } else {
                # If no error, we break out of the loop successfully!
                success <- TRUE
              }
            }
            
            # If we stepped all the way down to 0 and it still failed, skip to the next method
            if (!success) {
              cat(sprintf("Skipped: VarSel %s failed to converge at any LV for QDA.\n", varsel_name))
              next
            }
            
            opt_idx <- which(res_nlv$optimum == 1)[1]
            error_cv    <- res_nlv$err_mean[[opt_idx]]
            error_cv_sd <- res_nlv$err_sd[[opt_idx]]
            
            final_mod <- plsqda(X = X_train_prep_sel, y = y_train, weights = w_train, nlv = opt_idx)
            preds     <- predict(final_mod, X_test_prep_sel)
            
            y_test_f <- factor(y_test, levels = c("0", "1"))
            pred_f   <- factor(preds$pred, levels = c("0", "1"))
            cm       <- table(Predicted = pred_f, Actual = y_test_f)
            
            tp <- if("1" %in% rownames(cm) && "1" %in% colnames(cm)) cm["1", "1"] else 0
            tn <- if("0" %in% rownames(cm) && "0" %in% colnames(cm)) cm["0", "0"] else 0
            fp <- if("1" %in% rownames(cm) && "0" %in% colnames(cm)) cm["1", "0"] else 0
            fn <- if("0" %in% rownames(cm) && "1" %in% colnames(cm)) cm["0", "1"] else 0
            
            acc  <- (tp + tn) / sum(cm)
            sens <- if((tp + fn) > 0) tp / (tp + fn) else 0
            spec <- if((tn + fp) > 0) tn / (tn + fp) else 0
            
            prec <- if((tp + fp) > 0) tp / (tp + fp) else 0
            f1_val <- if((prec + sens) > 0) 2 * ((prec * sens) / (prec + sens)) else 0
            
            mcc_num <- (as.numeric(tp) * tn) - (as.numeric(fp) * fn)
            mcc_den <- sqrt(as.numeric(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc_val <- if(is.na(mcc_den) || mcc_den == 0) 0 else mcc_num / mcc_den
            
            scores_pos <- preds$posterior[, 2]
            roc_obj <- roc(y_test_f, scores_pos, levels = c("0", "1"), direction = "<", quiet = TRUE)
            auc_val <- as.numeric(auc(roc_obj))
            
            best_coords <- coords(roc_obj, x = "best", best.method = "youden", ret = "threshold")
            opt_thresh  <- as.numeric(unlist(best_coords))[1] 
            
            scores_hc <- scores_pos[y_test == "1"]
            scores_lc <- scores_pos[y_test == "0"]
            
            if(length(scores_hc) > 0 && length(scores_lc) > 0) {
              pr_obj <- pr.curve(scores.class0 = scores_hc, scores.class1 = scores_lc, curve = FALSE)
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
              F1_score = round(f1_val, 4),
              MCC = round(mcc_val, 4),
              ROC_AUC = round(auc_val, 4),
              PR_AUC = round(pr_auc_val, 4),
              Opt_Threshold = round(opt_thresh, 4)
            )
            
            results_list[[length(results_list) + 1]] <- res_row
            cat(sprintf("Done: Dataset %d | Material %s | Toxin %s | Prep %s | VarSel: %s\n", i, material, toxin, prep_name, varsel_name))
          }
        }
      }
    }
  }
  
  final_df <- do.call(rbind, results_list)
  return(final_df)
}

# --- Execution ---

# Define the configurations for the 4 runs
run_configs <- list(
  list(threshold = 1000, weighted = "none",     toxin = "DON",       output = "plsqda_1000_results.csv"),
  list(threshold = 1250, weighted = "none",     toxin = "Total_DON", output = "plsqda_1250_results.csv"),
  list(threshold = 1000, weighted = "triangle", toxin = "DON",       output = "weighted_plsqda_1000_results.csv"),
  list(threshold = 1250, weighted = "triangle", toxin = "Total_DON", output = "weighted_plsqda_1250_results.csv")
)

# Loop through each configuration, load the data, run the pipeline, and save the output
for (cfg in run_configs) {
  
  message(sprintf("\n--- Starting Run: Threshold = %d, Weighted = %s, Toxin = %s ---", 
                  cfg$threshold, cfg$weighted, cfg$toxin))
  
  # Load the specific dataset for this run
  current_nirs_data <- load_nirs_data(weighted = cfg$weighted, threshold = cfg$threshold)
  
  # Determine if we should use weights in the modeling functions
  use_weights_flag <- (cfg$weighted == "triangle")
  
  # Run the pipeline
  binary_plsqda_results <- pipeline_binary_plsqda_sequential(
    nirs_data = current_nirs_data, 
    target_toxins = cfg$toxin, 
    use_weights = use_weights_flag
  )
  
  # Save the results
  output_path <- file.path("./Austrian/results", cfg$output)
  write.csv(binary_plsqda_results, output_path, row.names = FALSE)
  
  message(sprintf("Saved results to %s\n", output_path))
}

message("All pipelines completed successfully!")