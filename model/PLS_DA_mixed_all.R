# Set a global seed and parallel-safe RNG for the entire script
RNGkind("L'Ecuyer-CMRG")
set.seed(270725)

# Import pipeline config
source("./model/pipeline_config.R")

# Load necessary libraries
library(rchemo)
library(pROC)
library(prospectr)
library(PRROC)
library(plsVarSel)

# Define preprocessing methods
prep_methods <- list(
  'SNV'         = function(x) savitzkyGolay(standardNormalVariate(x), w = 15, p = 2, m = 0),
  '1st_Der'     = function(x) savitzkyGolay(x, w = 15, p = 2, m = 1)
)

# Define variable selection methods
varsel_methods <- list(
  "None" = function(x, y, max_comp, k_folds, cur_seed){ 1:ncol(x) },
  "VIP"  = function(x, y, max_comp, k_folds, cur_seed) {
      set.seed(cur_seed)
      df <- filterPLSR(y, x, ncomp = max_comp, ncomp.opt = "minimum", validation = "CV", segments = k_folds, VIP.threshold = 1)
      return(df$VIP)
  }
)

# Refactored pipeline function 
pipeline_binary_plsda_sequential <- function(nirs_data, target_toxins, use_weights = FALSE, apply_downsampling = TRUE, max_nlv = 25) {
  
  results_list <- list()
  preps <- names(prep_methods)
  varsel_names <- names(varsel_methods)  
  materials <- c("intact", "ground")
  
  for (material in materials) {
    for (i in 1:100) { 
      
      current_seed <- 270725 + i
      
      for (toxin in target_toxins) {
        
        # 1. Extract Raw Data for this split
        X_train_raw <- as.matrix(nirs_data$X_train[[material]][[i]])
        X_test_raw  <- as.matrix(nirs_data$X_test[[material]][[i]])
        
        y_train_raw  <- unlist(nirs_data$y_train[[material]][[i]][[toxin]]$Class)
        y_train      <- ifelse(y_train_raw == "HC", "1", "0")
        names(y_train) <- rownames(X_train_raw)

        # 2. Conditionally assign weights
        if (use_weights) {
          w_train <- as.numeric(unlist(nirs_data$y_train[[material]][[i]][[toxin]]$Weight))
          names(w_train) <- rownames(X_train_raw)
        } else {
          w_train <- NULL
        }

        # 3. Apply Downsampling (Single-Shot Optimized)
        if (apply_downsampling) {
          class_0_idx <- which(y_train == "0")
          class_1_idx <- which(y_train == "1") # Always minority
          
          target_size <- length(class_1_idx)
          
          smart_downsample <- function(class_idx, target_size, weights) {
            if (is.null(weights)) return(sample(class_idx, target_size))
            
            class_weights <- weights[class_idx]
            uncertain_idx <- class_idx[class_weights < 1]
            certain_idx   <- class_idx[class_weights == 1]
            
            prop_uncertain <- length(uncertain_idx) / length(class_idx)
            n_uncertain <- min(round(target_size * prop_uncertain), length(uncertain_idx))
            n_certain   <- target_size - n_uncertain
            
            sampled_uncertain <- sample(uncertain_idx, n_uncertain)
            sampled_certain   <- sample(certain_idx, n_certain)
            
            return(c(sampled_uncertain, sampled_certain))
          }
          
          combo_downsample_seed <- current_seed + match(material, materials) * 1000L + match(toxin, target_toxins) * 100L
          set.seed(combo_downsample_seed) 
          
          down_class_0_idx <- smart_downsample(class_0_idx, target_size, w_train)
          down_idx <- c(down_class_0_idx, class_1_idx)
          
          X_train_raw <- X_train_raw[down_idx, , drop = FALSE]
          y_train     <- y_train[down_idx]
          if (use_weights) w_train <- w_train[down_idx]
        }

        y_test_raw   <- unlist(nirs_data$y_test[[material]][[i]][[toxin]]$Class)
        y_test       <- ifelse(y_test_raw == "HC", "1", "0")
        
        # 4. Proceed with Preprocessing & Modeling
        for (prep_name in preps) {
          
          X_train_prep <- prep_methods[[prep_name]](X_train_raw)
          X_test_prep  <- prep_methods[[prep_name]](X_test_raw)
          
          for (varsel_name in varsel_names){
            
            combo_seed <- current_seed +
              match(material, materials)       * 10000L +
              match(toxin, target_toxins)      * 1000L +
              match(prep_name, preps)          * 100L +
              match(varsel_name, varsel_names) * 10L
            set.seed(combo_seed)
            
            selected_vars <- varsel_methods[[varsel_name]](
              x = X_train_prep, 
              y = as.numeric(y_train), 
              max_comp = max_nlv, k_folds = 10, cur_seed = combo_seed)
          
            X_train_prep_sel <- X_train_prep[, selected_vars, drop = FALSE]
            X_test_prep_sel  <- X_test_prep[, selected_vars, drop = FALSE]
            
            res_nlv <- plsr_plsda_allsteps(
                      X = X_train_prep_sel, Xname = NULL, Xscaling = "none", 
                      Y = y_train, Yscaling = "none", weights = w_train,
                      newX = NULL, newXname = NULL, method = "plsrda",
                      step = "nlvtest", nlv = max_nlv, 
                      modeloutput = c("scores","loadings","coef","vip"), 
                      cvmethod = "kfolds", nbrep = 3, seed = combo_seed, 
                      nfolds = 10, criterion = "err", selection = "globalmin"
            )
            
            opt_idx <- which(res_nlv$optimum == 1)[1]
            error_cv    <- res_nlv$err_mean[[opt_idx]]
            error_cv_sd <- res_nlv$err_sd[[opt_idx]]
            
            final_mod <- plsrda(X = X_train_prep_sel, y = y_train, weights = w_train, nlv = opt_idx)
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
            
            # --- UPDATED RES_ROW ---
            res_row <- data.frame(
              Sample_source = material,
              Target_toxin = toxin,
              Preprocessing = prep_name,
              Variable_Selection = varsel_name,
              Downsampled = apply_downsampling,
              Split = i,
              LV = opt_idx,
              Err_CV = round(error_cv, 4),
              Err_CV_SD = round(error_cv_sd, 4),
              TN = tn, FP = fp, FN = fn, TP = tp,
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
            ds_label <- ifelse(apply_downsampling, "YES", "NO")
            cat(sprintf("Done: Split %d | Mat: %s | Tox: %s | DS: %s | Prep: %s | VarSel: %s\n", 
                        i, material, toxin, ds_label, prep_name, varsel_name))
          }
        }
      }
    }
  }
  
  final_df <- do.call(rbind, results_list)
  return(final_df)
}

# ==============================================================================
# --- Execution Block ---
# ==============================================================================

# Define the configurations for the 4 runs
run_configs <- list(
  list(threshold = 1000, weighted = "none",     toxin = "DON",       output = "plsda_1000_all.csv"),
  list(threshold = 1250, weighted = "none",     toxin = "Total_DON", output = "plsda_1250_all.csv"),
  list(threshold = 1000, weighted = "prob", toxin = "DON",       output = "weighted_plsda_1000_all.csv"),
  list(threshold = 1250, weighted = "prob", toxin = "Total_DON", output = "weighted_plsda_1250_all.csv")
)

# Loop through each configuration
for (cfg in run_configs) {
  
  message(sprintf("\n--- Starting Run: Threshold = %d, Weighted = %s, Toxin = %s ---", 
                  cfg$threshold, cfg$weighted, cfg$toxin))
  
  # Lock the seed before loading to prevent OS-level file shifting
  set.seed(270725)
  current_nirs_data <- load_nirs_data(weighted = cfg$weighted, threshold = cfg$threshold, base_path = "./data/train_test_set_all/")
  
  use_weights_flag <- (cfg$weighted == "prob")
  
  message("Running Phase 1: Without Downsampling...")
  res_no_ds <- pipeline_binary_plsda_sequential(
    nirs_data = current_nirs_data, 
    target_toxins = cfg$toxin, 
    use_weights = use_weights_flag,
    apply_downsampling = FALSE 
  )
  
  message("Running Phase 2: With Downsampling...")
  res_with_ds <- pipeline_binary_plsda_sequential(
    nirs_data = current_nirs_data, 
    target_toxins = cfg$toxin, 
    use_weights = use_weights_flag,
    apply_downsampling = TRUE 
  )
  
  # Combine both results into one beautiful dataset
  combined_results <- rbind(res_no_ds, res_with_ds)
  
  # Save the results
  output_path <- file.path("./results/mixed_all/", cfg$output)
  write.csv(combined_results, output_path, row.names = FALSE)
  
  message(sprintf("Saved combined results to %s\n", output_path))
}

message("All pipelines completed successfully!")