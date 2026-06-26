library(ggplot2)
library(donutsk)
library(dplyr)
library(ggpubr)
library(viridis)

plot_spectra <- function(spectra, title, y_label = "log (1/R)", color) {
  n_groups <- length(unique(spectra[[color]]))

  output <-ggplot(spectra, aes(x = as.numeric(as.character(Wavelength)), y = Reflectance, group = Sample_ID, color = .data[[color]])) +
    geom_line(alpha = 0.5) +
    labs(title = title, x = "Wavelength (nm)", y = y_label) +
    theme_classic() +
    theme(legend.title = element_blank(), 
          legend.position = "top")

  if (n_groups == 2) {
    output <- output + scale_color_manual(values = c("#1982c4", "#ef233c"))
  } else if (n_groups == 4) {
    output <- output + scale_color_manual(values = c("#1982c4", "#ef233c", "#b6d5e9", "#f7c1bb"))
  } else {
    output <- output + scale_color_viridis_d()
  }
  
  return(output)
}

plot_pca <- function(input, input2, title, color, shape = NULL) {
  # Calculate the farthest distance from the origin
  max_x <- 1.25 * max(abs(input$`Comp 1`))
  max_y <- 1.25 * max(abs(input$`Comp 2`))

  n_groups <- length(unique(input[[color]]))
  
  # Base plot (without shape)
  output <- ggplot(data = input, aes(x = `Comp 1`, y = `Comp 2`, color = .data[[color]])) 
  
  # Shape identification
  if (!is.null(shape)) {
    output <- output + 
      geom_point(aes(shape = .data[[shape]]), size = 3) +
      scale_shape_manual(values = c(16, 17))
  } else {
    output <- output + 
      geom_point(size = 3)
  }
  
  output <- output +
    theme_minimal(base_family = "Arial") +
    xlab(paste("PC1 (", round(input2[1], 2), "%)", sep = "")) +
    ylab(paste("PC2 (", round(input2[2], 2), "%)", sep = "")) +
    labs(color = NULL, title = title, shape = NULL) +
    xlim(-max_x, max_x) + 
    ylim(-max_y, max_y) + 
    theme_bw() +
    theme(legend.position = "top",
          legend.text = element_text(size = 12),
          plot.title = element_text(hjust = 0.5, margin = margin(b = 1), size = 12, face = "bold"),
          panel.grid = element_blank(),
          axis.text = element_text(size = 12),
          axis.title = element_text(size = 12),
          axis.title.x = element_text(margin = margin(t = 1)),
          axis.title.y = element_text(margin = margin(r = 1))) +
    geom_hline(yintercept = 0, linetype = "dotted", color = "black") + 
    geom_vline(xintercept = 0, linetype = "dotted", color = "black")

  # Apply colors
  if (n_groups == 2) {
    output <- output + scale_color_manual(values = c("#1982c4", "#ef233c"))
  } else if (n_groups == 4) {
    output <- output + scale_color_manual(values = c("#1982c4", "#ef233c", "#b6d5e9", "#f7c1bb"))
  } else {
    output <- output + scale_color_viridis_d() # default color palette for more than 4 categories
  }
  
  return(output)
}


plot_stacked_counts <- function(data, x_var, fill_var, title = NULL, x_label = NULL) {
  
  # Auto-generate titles and labels if none are provided
  if (is.null(title)) {
    title <- paste("Distribution of", fill_var, "by", x_var)
  }
  if (is.null(x_label)) {
    x_label <- x_var
  }
  
  n_groups <- length(unique(data[[fill_var]]))

  output <- ggplot(data, aes(x = .data[[x_var]], fill = .data[[fill_var]])) +
            geom_bar() +
            geom_text(
              stat = "count", 
              aes(label = after_stat(count)), 
              position = position_stack(vjust = 0.5), 
              size = 4,
              color = "white"
            ) +
            theme_classic() +
            labs(title = title, x = x_label, y = "Count") +
            theme(
              legend.position = "top",
              legend.title = element_blank(),
              legend.text = element_text(size = 14),
              plot.title = element_text(size = 16),
              axis.text = element_text(size = 14),
              axis.title = element_text(size = 16)
            )
  if (n_groups == 2) {
    output <- output + scale_fill_manual(values = c("#1982c4", "#ef233c"))
  } else if (n_groups == 4) {
    output <- output + scale_fill_manual(values = c("#1982c4", "#b6d5e9", "#f7c1bb", "#ef233c"))
  } else {
    output <- output + scale_fill_viridis_d()
  }
  
  return(output)
}

plot_top_pipelines <- function(pipeline_emmeans, 
                               metric_name = "F1 Score", 
                               backtransform_fn = function(x) { plogis(x) }) { 
  
  # Load necessary libraries
  require(tidyverse)
  require(patchwork)
  require(multcomp)
  
  # Get indices of top 10 from the emmeans object
  top10_idx <- order(as.data.frame(pipeline_emmeans)$emmean, 
                     decreasing = TRUE)[1:10]
  top10_means <- pipeline_emmeans[top10_idx]
  
  # Run cld() on the top 10
  cld_top10 <- cld(top10_means,
                   adjust  = "sidak",
                   Letters = letters,
                   decreasing = TRUE) %>%
    as.data.frame()
  
  # Apply custom back-transformation and prepare the data
  df_clean <- cld_top10 %>%
    mutate(
      pipeline = paste(Grinding, Toxin, Preprocessing,
                       Variable_Selection, Downsampling,
                       Dataset, Weighting, sep = " | "),
      
      # Apply the provided mathematical transformation
      Est = backtransform_fn(emmean),
      LCL = backtransform_fn(asymp.LCL),
      UCL = backtransform_fn(asymp.UCL),
      
      .group  = trimws(.group),
      is_best = .group == .group[which.max(Est)]
    ) %>%
    arrange(desc(Est)) %>% 
    mutate(
      Pipeline_ID = paste("Pipeline", row_number()),
      Pipeline_ID = factor(Pipeline_ID, levels = rev(paste("Pipeline", row_number())))
    )
  
  # Create the Left Plot: The Parameter Matrix
  param_data <- df_clean %>%
    dplyr::select(
      G = Grinding, 
      T = Toxin, 
      P = Preprocessing, 
      Vs = Variable_Selection, 
      W = Weighting,
      B = Downsampling,
      Ds = Dataset,
      Pipeline_ID
    ) %>%
    pivot_longer(-Pipeline_ID, names_to = "Parameter", values_to = "Setting")
  
  param_data$Parameter <- factor(param_data$Parameter, 
                                 levels = c("Ds", "B", "G", "T", "W", "P", "Vs"))
  
  p_matrix <- ggplot(param_data, aes(x = Parameter, y = Pipeline_ID)) +
    geom_tile(fill = "white", color = "white", linewidth = 0.5, alpha = 0.3) +
    geom_text(aes(label = Setting), size = 3, color = "black") +
    scale_fill_discrete(guide = "none") + 
    scale_x_discrete(position = "top") +
    labs(
      title    = paste("Top 10 Pipelines by Estimated", metric_name),
      subtitle = "Shared letters = not significantly different\n",
      x = NULL, y = NULL) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title      = element_text(face = "bold", size = 13),
      plot.subtitle   = element_text(color = "grey40", size = 9),
      axis.text.x     = element_text(hjust = 0.5, face = "bold"),
      axis.text.y     = element_text(face = "bold"),
      panel.grid      = element_blank()
    )
  
  # Create the Right Plot
  x_min <- min(df_clean$LCL, na.rm = TRUE)
  x_max <- max(df_clean$UCL, na.rm = TRUE)
  x_range <- x_max - x_min
  
  p_scores <- df_clean %>%
    ggplot(aes(x = Est, y = Pipeline_ID, color = is_best)) +
    geom_point(size = 4) +
    geom_errorbar(aes(xmin = LCL, xmax = UCL),
                  width = 0.25, linewidth = 0.8) +
    geom_text(aes(label = .group, x = UCL + (x_range * 0.05)),
              vjust = 0.4, size = 4, 
              color = "black", fontface = "bold") +
    scale_color_manual(
      values = c("TRUE" = "#1a9850", "FALSE" = "gray"),
      labels = c("TRUE"  = "Best pipelines",
                 "FALSE" = "Other"),
      name   = ""
    ) +
    scale_x_continuous(
      limits = c(x_min - (x_range * 0.05), x_max + (x_range * 0.15))
    ) +
    labs(
      x        = paste("Estimated", metric_name),
      y        = NULL,
      caption  = "Abbreviations: G (Grinding), T (Toxin), P (Spectral Preprocessing), Vs (Variable Selection), W (Weighting), B (Balancing), Ds (Dataset)\nError bars = 95% Confidence Interval"
    ) +
    theme_bw(base_size = 11) +
    theme(
      axis.text.y     = element_blank(), 
      axis.ticks.y    = element_blank(),
      legend.position = c(0.01, 0.99),
      legend.direction = "horizontal",
      legend.justification = c(0, 1),
      plot.caption    = element_text(size = 8, hjust = 1, color = "grey30"),
      panel.grid      = element_blank()
    )
  
  final_visualization <- p_matrix + p_scores + plot_layout(widths = c(1, 1.5))
  
  return(list(plot = final_visualization, data = df_clean))
}

plot_effect <- function(model, factor, title, X= NULL, y = "Estimated F1-Score", 
                        backtransform_fn = function(x) { plogis(x) }) {
  library(emmeans)
  emms <- emmeans(model, specs = factor)
  emms_df <- as.data.frame(emms) %>%
    mutate(
      Factor = as.character(.data[[factor]]),
      Estimate = backtransform_fn(emmean),
      LCL = backtransform_fn(asymp.LCL),
      UCL = backtransform_fn(asymp.UCL)
    )
  
  output <- ggplot(emms_df, aes(x = Factor, y = Estimate, color = Factor)) +
            geom_errorbar(aes(ymin = LCL, ymax = UCL), width = 0.1, linewidth = 1) +
            geom_point(size = 4) +
            scale_color_viridis_d() +
            theme_classic() +
            labs(title = title, x = X, y = y) +
            theme(
              plot.title = element_blank(),
              legend.position = "none"
            )
  return(list(plot = output, data = emms_df))
}

hist_plot <- function(data, x = "Total DON") {
  output <- ggplot(data, aes(x = .data[[x]], fill = Uncertainty)) +
    geom_histogram(boundary = 1000, binwidth = 100, position = "identity") + 
    geom_vline(xintercept = 1000, linetype = "dashed", color = "black", linewidth = 1) + 
    theme_classic(base_size = 14) +
    labs(
      x = paste(x, "(µg/kg)"),
      y = "Count",
    ) +
    scale_fill_manual(values = c("False Positive (FP)" = "#ef233c",
                                "Uncertain False Positive (UFP)" = "#f7c1bb",
                                "False Negative (FN)" = "#1982c4",
                                "Uncertain False Negative (UFN)" = "#b6d5e9")) +
    theme(
      legend.position = "bottom",
      plot.title = element_text(face = "bold"),
      legend.title = element_blank()
    )
  return(output)
}

density_plot <- function(data, var, color, label, title){
  plot <- ggplot(data, aes(x = var, fill = color)) +
    geom_density(alpha = 0.5) +
    theme_classic() +
    scale_y_continuous(limits = c(0, 5)) +
    labs(title = title, x = label, fill = color) +
    scale_fill_viridis(discrete = TRUE) +
    theme(legend.position = "top",
    legend.title = element_blank())
  return(plot)
}