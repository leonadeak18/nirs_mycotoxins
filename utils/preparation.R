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
  rownames(output) <- samples

  return(output)
}

binary_transform <- function(y, threshold = 1000, targets = c("DON", "Total_DON")) {
  y_class <- list()
  for (target in targets) {
    # Binary classification
    y_class[[target]] <- cut(
      y[[target]],
      breaks = c(-Inf, threshold, Inf),
      labels = c("LC", "HC")
    )
    names(y_class[[target]]) <- rownames(y)
  }  
  return(y_class)
}

three_level_transform <- function(df, cutoff = 1000, uncertainty_percent = 0.5) {
  output <- data.frame(
    X = df[, 1],
    class = factor(ifelse(df[, 2] > cutoff * (1 + uncertainty_percent), "HC",
                          ifelse(df[, 2] < cutoff * (1 - uncertainty_percent), "LC", "UC")))
  )
  rownames(output) <- output$X
  output <- output[, -1, drop = FALSE]
  
  return(output)
}

binary_transform_with_weight <- function(y, threshold = 1000, uncertainty = 0.5, targets = c("DON", "Total_DON")) {
  y_class <- list()
  
  certainty_multiplier <- 1.0 - uncertainty
  gray_area_width <- threshold * uncertainty
  
  for (target in targets) {
    # 1. Extract the column as a numeric vector
    val <- y[[target]]
    
    # 2. Binary classification
    cls <- cut(
      val,
      breaks = c(-Inf, threshold, Inf),
      labels = c("LC", "HC")
    )
    
    # 3. Distance from threshold (using the numeric vector 'val')
    raw_dist <- abs(val - threshold)
    
    # 4. Scale and clip between 0 and 1
    dist_weight <- pmin(pmax(raw_dist / gray_area_width, 0), 2)
    
    # 5. Combine with certainty
    final_weight <- dist_weight * certainty_multiplier
    
    # 6. Store as a data frame within the list to hold both values
    y_class[[target]] <- data.frame(
      Class = cls,
      Weight = final_weight,
      row.names = rownames(y)
    )
  }
  
  return(y_class)
}
