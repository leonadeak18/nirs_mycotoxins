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
library(recipes)
library(themis)

# Define preprocessing methods
prep_methods <- list(
  'SNV'     = function(x) savitzkyGolay(standardNormalVariate(x), w = 15, p = 2, m = 0),
  '1st_Der' = function(x) savitzkyGolay(x, w = 15, p = 2, m = 1)
)

varsel_methods <- list(
  "None" = function(x, y, opt_nlv, weights) {
    1:ncol(x)
  },
  "VIP" = function(x, y, opt_nlv, weights) {
    fm <- plsrda(X = x, y = y, weights = weights, nlv = opt_nlv)
    vip_scores <- vip(fm, x, y, nlv = opt_nlv)[, sprintf("comp %d", opt_nlv)]
    return(which(vip_scores > 1))
  }
)

pipeline_binary_plsda_sequential <- function(
  nirs_data,
  target_toxins,
  use_weights = FALSE,
  sampling_method = c("none", "downsample", "smote"),
  smote_k = 5,
  smote_distance = "euclidean",
  max_nlv = 25
) {
  sampling_method <- match.arg(sampling_method)
  results_list <- list()
  preps <- names(prep_methods)
  varsel_names <- names(varsel_methods)
  materials <- c("intact", "ground")

  for (material in materials) {
    for (i in 1:100) {

      current_seed <- 270725 + i

      for (toxin in target_toxins) {

        # Extract Raw Data for this split
        X_train_raw <- as.matrix(nirs_data$X_train[[material]][[i]])
        X_test_raw  <- as.matrix(nirs_data$X_test[[material]][[i]])

        y_train_raw <- unlist(nirs_data$y_train[[material]][[i]][[toxin]]$Class)
        y_train     <- ifelse(y_train_raw == "HC", "1", "0")
        names(y_train) <- rownames(X_train_raw)

        # Conditionally assign weights
        if (use_weights) {
          w_train <- as.numeric(unlist(nirs_data$y_train[[material]][[i]][[toxin]]$Weight))
          names(w_train) <- rownames(X_train_raw)
        } else {
          w_train <- NULL
        }

        # Apply Downsampling (on raw data)
        if (sampling_method == "downsample") {
          class_0_idx <- which(y_train == "0")
          class_1_idx <- which(y_train == "1") # HC is always the minority class

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

        y_test_raw <- unlist(nirs_data$y_test[[material]][[i]][[toxin]]$Class)
        y_test     <- ifelse(y_test_raw == "HC", "1", "0")

        # Proceed with Preprocessing & Modeling
        for (prep_name in preps) {

          X_train_prep <- prep_methods[[prep_name]](X_train_raw)
          X_test_prep  <- prep_methods[[prep_name]](X_test_raw)

          # Working copies so SMOTE does not overwrite y_train/w_train across preps
          y_train_cur <- y_train
          w_train_cur <- w_train
          k_used      <- NA_integer_

          # Apply SMOTE directly (using only weight == 1 minority samples when weighted)
          if (sampling_method == "smote") {
            if (use_weights) {
              eligible_min_idx <- which(y_train_cur == "1" & w_train_cur == 1)
              n_uncertain_min  <- sum(y_train_cur == "1" & w_train_cur < 1)
            } else {
              eligible_min_idx <- which(y_train_cur == "1")
              n_uncertain_min  <- 0L
            }

            maj_idx      <- which(y_train_cur == "0")
            n_maj        <- length(maj_idx)
            n_min_elig   <- length(eligible_min_idx)
            n_syn_needed <- n_maj - (n_min_elig + n_uncertain_min)
            k_use        <- min(smote_k, n_min_elig - 1L)

            if (n_syn_needed <= 0L || k_use < 1L) {
              warning(sprintf(
                "Skipping SMOTE on split %d | %s | %s | %s (eligible minority = %d, needed = %d).",
                i, material, toxin, prep_name, n_min_elig, n_syn_needed
              ))
            } else {
              k_used <- k_use
              smote_seed <- current_seed +
                match(material, materials)  * 1000L +
                match(toxin, target_toxins) * 100L +
                match(prep_name, preps)     * 10L
              set.seed(smote_seed)

              keep_idx <- c(maj_idx, eligible_min_idx)
              df_smote <- as.data.frame(X_train_prep[keep_idx, , drop = FALSE])
              df_smote$y_target <- factor(y_train_cur[keep_idx], levels = c("0", "1"))

              target_ratio <- (n_maj - n_uncertain_min) / n_maj

              smote_out <- smote(
                df         = df_smote,
                var        = "y_target",
                k          = k_use,
                over_ratio = target_ratio,
                distance   = smote_distance
              )

              n_orig <- nrow(df_smote)
              if (nrow(smote_out) > n_orig) {
                syn_rows <- smote_out[(n_orig + 1):nrow(smote_out), , drop = FALSE]
                X_syn    <- as.matrix(syn_rows[, setdiff(names(syn_rows), "y_target"), drop = FALSE])
                colnames(X_syn) <- colnames(X_train_prep)

                # Give synthetic rows explicit, matching IDs across X, Y, and weights
                syn_ids <- paste0("SMOTE_", seq_len(nrow(X_syn)))
                rownames(X_syn) <- syn_ids

                y_syn <- as.character(syn_rows$y_target)
                names(y_syn) <- syn_ids

                X_train_prep <- rbind(X_train_prep, X_syn)
                y_train_cur  <- c(y_train_cur, y_syn)

                if (use_weights) {
                  w_syn <- rep(1, nrow(X_syn))
                  names(w_syn) <- syn_ids
                  w_train_cur <- c(w_train_cur, w_syn)
                }
              }
            }
          }

          for (varsel_name in varsel_names) {

            combo_seed <- current_seed +
              match(material, materials)       * 10000L +
              match(toxin, target_toxins)      * 1000L +
              match(prep_name, preps)          * 100L +
              match(varsel_name, varsel_names) * 10L
            set.seed(combo_seed)

            # CV on full variable set to find true optimal LV
            res_nlv_full <- plsr_plsda_allsteps(
              X = X_train_prep, Xname = NULL, Xscaling = "none",
              Y = y_train_cur, Yscaling = "none", weights = w_train_cur,
              newX = NULL, newXname = NULL, method = "plsrda",
              step = "nlvtest", nlv = max_nlv,
              modeloutput = c("scores", "loadings", "coef", "vip"),
              cvmethod = "kfolds", nbrep = 3, seed = combo_seed,
              nfolds = 10, criterion = "err", selection = "globalmin"
            )

            opt_nlv <- which(res_nlv_full$optimum == 1)[1]

            # Variable selection using rchemo VIP at opt_nlv
            selected_vars <- varsel_methods[[varsel_name]](
              x       = X_train_prep,
              y       = y_train_cur,
              opt_nlv = opt_nlv,
              weights = w_train_cur
            )

            # If VIP > 1 selects zero variables, fall back to all
            if (length(selected_vars) == 0) {
              warning(sprintf(
                "VIP selected 0 variables for split %d | %s | %s | %s. Falling back to all variables.",
                i, material, toxin, prep_name
              ))
              selected_vars <- 1:ncol(X_train_prep)
            }

            X_train_prep_sel <- X_train_prep[, selected_vars, drop = FALSE]
            X_test_prep_sel  <- X_test_prep[, selected_vars, drop = FALSE]

            # Refit final model on selected variables at opt_nlv
            final_mod <- plsrda(X = X_train_prep_sel, y = y_train_cur, weights = w_train_cur, nlv = opt_nlv)
            preds     <- predict(final_mod, X_test_prep_sel)

            error_cv    <- res_nlv_full$err_mean[[opt_nlv]]
            error_cv_sd <- res_nlv_full$err_sd[[opt_nlv]]

            y_test_f <- factor(y_test, levels = c("0", "1"))
            pred_f   <- factor(preds$pred, levels = c("0", "1"))
            cm       <- table(Predicted = pred_f, Actual = y_test_f)

            tp <- if ("1" %in% rownames(cm) && "1" %in% colnames(cm)) cm["1", "1"] else 0
            tn <- if ("0" %in% rownames(cm) && "0" %in% colnames(cm)) cm["0", "0"] else 0
            fp <- if ("1" %in% rownames(cm) && "0" %in% colnames(cm)) cm["1", "0"] else 0
            fn <- if ("0" %in% rownames(cm) && "1" %in% colnames(cm)) cm["0", "1"] else 0

            acc    <- (tp + tn) / sum(cm)
            sens   <- if ((tp + fn) > 0) tp / (tp + fn) else 0
            spec   <- if ((tn + fp) > 0) tn / (tn + fp) else 0
            prec   <- if ((tp + fp) > 0) tp / (tp + fp) else 0
            f1_val <- if ((prec + sens) > 0) 2 * ((prec * sens) / (prec + sens)) else 0

            mcc_num <- (as.numeric(tp) * tn) - (as.numeric(fp) * fn)
            mcc_den <- sqrt(as.numeric(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc_val <- if (is.na(mcc_den) || mcc_den == 0) 0 else mcc_num / mcc_den

            scores_pos <- preds$posterior[, 2]
            roc_obj    <- roc(y_test_f, scores_pos, levels = c("0", "1"), direction = "<", quiet = TRUE)
            auc_val    <- as.numeric(auc(roc_obj))

            best_coords <- coords(roc_obj, x = "best", best.method = "youden", ret = "threshold")
            opt_thresh  <- as.numeric(unlist(best_coords))[1]

            scores_hc <- scores_pos[y_test == "1"]
            scores_lc <- scores_pos[y_test == "0"]

            if (length(scores_hc) > 0 && length(scores_lc) > 0) {
              pr_obj     <- pr.curve(scores.class0 = scores_hc, scores.class1 = scores_lc, curve = FALSE)
              pr_auc_val <- pr_obj$auc.integral
            } else {
              pr_auc_val <- NA
            }

            res_row <- data.frame(
              Sample_source      = material,
              Target_toxin       = toxin,
              Preprocessing      = prep_name,
              Variable_Selection = varsel_name,
              Sampling           = sampling_method,
              SMOTE_k            = k_used,
              Split              = i,
              LV                 = opt_nlv,
              N_Vars_Selected    = length(selected_vars),
              Err_CV             = round(error_cv, 4),
              Err_CV_SD          = round(error_cv_sd, 4),
              TN = tn, FP = fp, FN = fn, TP = tp,
              Accuracy           = round(acc, 4),
              Sensitivity        = round(sens, 4),
              Specitifity        = round(spec, 4),
              F1_score           = round(f1_val, 4),
              MCC                = round(mcc_val, 4),
              ROC_AUC            = round(auc_val, 4),
              PR_AUC             = round(pr_auc_val, 4),
              Opt_Threshold      = round(opt_thresh, 4)
            )

            results_list[[length(results_list) + 1]] <- res_row
            cat(sprintf(
              "Done: Split %d | Mat: %s | Tox: %s | Samp: %s | Prep: %s | VarSel: %s | LV: %d | Vars: %d\n",
              i, material, toxin, sampling_method, prep_name, varsel_name, opt_nlv, length(selected_vars)
            ))
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
  list(threshold = 1000, weighted = "none",     toxin = "DON",       output = "plsda_austrian.csv"),
  list(threshold = 1000, weighted = "none",     toxin = "Total_DON", output = "plsda_total_don_austrian.csv"),
  list(threshold = 1000, weighted = "triangle", toxin = "DON",       output = "weighted_plsda_austrian.csv"),
  list(threshold = 1000, weighted = "triangle", toxin = "Total_DON", output = "weighted_plsda_total_don_austrian.csv")
)

# Loop through each configuration
for (cfg in run_configs) {

  message(sprintf(
    "\n--- Starting Run: Threshold = %d, Weighted = %s, Toxin = %s ---",
    cfg$threshold, cfg$weighted, cfg$toxin
  ))

  # Lock the seed before loading to prevent OS-level file shifting
  set.seed(270725)
  current_nirs_data <- load_nirs_data(
    weighted  = cfg$weighted,
    threshold = cfg$threshold,
    base_path = "./data/train_test_set_austrian/"
  )

  use_weights_flag <- (cfg$weighted == "triangle")

  message("Running Phase 1: No Sampling...")
  res_none <- pipeline_binary_plsda_sequential(
    nirs_data       = current_nirs_data,
    target_toxins   = cfg$toxin,
    use_weights     = use_weights_flag,
    sampling_method = "none"
  )

  message("Running Phase 2: With Downsampling...")
  res_ds <- pipeline_binary_plsda_sequential(
    nirs_data       = current_nirs_data,
    target_toxins   = cfg$toxin,
    use_weights     = use_weights_flag,
    sampling_method = "downsample"
  )

  message("Running Phase 3: With SMOTE (Euclidean)...")
  res_smote <- pipeline_binary_plsda_sequential(
    nirs_data       = current_nirs_data,
    target_toxins   = cfg$toxin,
    use_weights     = use_weights_flag,
    sampling_method = "smote",
    smote_k         = 5
  )

  combined_results <- rbind(res_none, res_ds, res_smote)

  output_path <- file.path("./results/austrian/", cfg$output)
  write.csv(combined_results, output_path, row.names = FALSE)

  message(sprintf("Saved combined results to %s\n", output_path))
}

message("All pipelines completed successfully!")