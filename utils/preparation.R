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

binary_transform <- function(df, cutoff = 1000) {
  output <- data.frame(
    X = df[, 1],
    class = factor(ifelse(df[, 2] > cutoff, "HC", "LC"))
  )
  rownames(output) <- output$X
  output <- output[, -1, drop = FALSE]
  
  return(output)
}
