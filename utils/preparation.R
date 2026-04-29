library(dplyr)

# Prepare the dataframe
prepare_df <- function(input, r1, r2){
  if (r1 > r2) {
    temp <- r1
    r1 <- r2
    r2 <- temp
  }
  samples <- input[,1]
  df <- input[,-1]
  colnames(df) <- sub("^X", "", colnames(df))
  cols <- as.numeric(colnames(df))
  output <- df[, cols >= r1 & cols <= r2]
  output <- as.data.frame(output)
  rownames(output) <- samples

  return(output)
}

binary_transform <- function(y, threshold = 1000, targets = c("DON", "Total_DON")) {
  y_class <- list()
  for (target in targets) {
    # Binary classification
    cls <- cut(
      y[[target]],
      breaks = c(-Inf, threshold, Inf),
      labels = c("LC", "HC")
    )
    y_class[[target]] <- data.frame(
      Class = cls,
      Weight = 1,  # Assign a default weight of 1 for all samples
      row.names = rownames(y)
    )
  }  
  return(y_class)
}

binary_transform_with_prob <- function(y, threshold = 1000, uncertainty = 0.3, targets = c("DON", "Total_DON")) {
  y_class <- list()
  for (target in targets) {
    # Extract the column as a numeric vector
    val <- y[[target]]
    
    # Binary classification
    cls <- cut(
      val,
      breaks = c(-Inf, threshold, Inf),
      labels = c("LC", "HC")
    )
    
    # determine the weight
    stdev <- threshold * (uncertainty / 2)
    z_score <- abs(threshold - val) / stdev

    #scale the 0.5 to 1 probability range to 0 - 1 weight range
    raw_weight <- (pnorm(z_score) - 0.5) * 2
    final_weight <- ifelse(raw_weight > 0.95, 1, raw_weight)

    # 5. Store as a data frame within the list to hold both values
    y_class[[target]] <- data.frame(
      Class = cls,
      Weight = final_weight,
      row.names = rownames(y)
    )
  }
  
  return(y_class)
}

binary_transform_with_triangle <- function(y, threshold = 1000, uncertainty = 0.3, targets = c("DON", "Total_DON")) {
  y_class <- list()
  
  for (target in targets) {
    # Extract the column
    val <- y[[target]]
    
    # Binary classification
    cls <- cut(
      val,
      breaks = c(-Inf, threshold, Inf),
      labels = c("LC", "HC")
    )
    
    # Determine the upper and lower bounds based on relative uncertainty
    upper_bound <- threshold + (threshold * uncertainty)
    lower_bound <- threshold - (threshold * uncertainty)
    
    # Calculate the slopes
    lower_slope <- 1 / (threshold - lower_bound)
    upper_slope <- 1 / (upper_bound - threshold)

    # Fully vectorized weight calculation
    weight <- case_when(
      val <= lower_bound ~ 1,                             # 100% certain it is LC
      val >= upper_bound ~ 1,                             # 100% certain it is HC
      val < threshold    ~ lower_slope * (threshold - val),  # Scales down to 0 as it approaches threshold from below
      val > threshold    ~ upper_slope * (val - threshold),  # Scales down to 0 as it approaches threshold from above
      val == threshold   ~ 0                              # Exactly on the line = 0% certainty
    )
    
    # Store as a data frame
    y_class[[target]] <- data.frame(
      Class = cls,
      Weight = weight,
      row.names = rownames(y)
    )
  }
  
  return(y_class)
}