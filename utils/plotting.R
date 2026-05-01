library(ggplot2)
library(donutsk)
library(dplyr)

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
    output <- output + scale_color_viridis_d() # default color palette for more than 4 categories
  }
  
  return(output)
}

plot_pca <- function(input, input2, title, color, shape = NULL) {
  # Calculate the farthest distance from the origin
  max_x <- 1.25 * max(abs(input$`Comp 1`))
  max_y <- 1.25 * max(abs(input$`Comp 2`))

  n_groups <- length(unique(input[[color]]))
  
  # 1. Base plot (without shape)
  output <- ggplot(data = input, aes(x = `Comp 1`, y = `Comp 2`, color = .data[[color]])) 
  
  # 2. Add points conditionally based on whether 'shape' is provided
  if (!is.null(shape)) {
    output <- output + 
      geom_point(aes(shape = .data[[shape]]), size = 3) +
      scale_shape_manual(values = c(15:18))
  } else {
    output <- output + 
      geom_point(size = 3)
  }
  
  # 3. Add the rest of your formatting
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

  # 4. Apply colors
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
    output <- output + scale_fill_viridis_d() # default color palette for more than 4 categories
  }
  
  return(output)
}

plot_piedonut <- function(main, inner_donut, outer_donut, title = NULL) {
  df <- main %>%
        filter(.data[[inner_donut]] %in% c("C", "NC")) %>%
        mutate(
          Class = ifelse(.data[[inner_donut]] == "C", "Compliant", "Non-Compliant"),
          Uncertainty = ifelse(.data[[outer_donut]] %in% c("UNC", "UC"), "Uncertain", "Certain")
        ) %>%
        count(Class, Uncertainty, name = "Count") %>%
        mutate(total_count = sum(Count)) %>%
        mutate(Outer_Pct = paste0(round((Count / total_count) * 100, 1), "%")) %>%
        group_by(Class) %>%
        mutate(Inner_Pct = paste0(round((sum(Count) / total_count) * 100, 1), "%")) %>%
        ungroup()

  output <- ggplot(df, aes(value = Count, fill = Class)) +
    # --- Inner Ring  ---
    geom_donut_int(r_int = 0.1, color = "white", linewidth = 1) +
    geom_text_int(aes(label = Inner_Pct), r = 0.5, color = "white", fontface = "bold", size = 5) +
    
    # --- Outer Ring ---
    geom_donut_ext(aes(alpha = Uncertainty), color = "white", linewidth = 1, r_int = 1.2, r_ext = 1.6) +
    geom_text_ext(aes(label = Outer_Pct), layout = circle(r = 1.4), color = "black", size = 5) +
    
    scale_fill_manual(values = c("Compliant" = "#2E86C1", "Non-Compliant" = "#E74C3C")) +
    scale_alpha_manual(values = c("Certain" = 1.0, "Uncertain" = 0.35)) +
    coord_radial(theta = "y", expand = FALSE) +
    theme_void() +
    labs(title = title) +
    theme(
      legend.position = "top",
      legend.title = element_blank(),
      plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
      legend.text = element_text(size = 10)
    )
  
  return(output)
}

plot_top_pipelines <- function(pipeline_emmeans, 
                               metric_name = "F1 Score", 
                               backtransform_fn = function(x) { plogis(x) }) { 
  
  # Load necessary libraries
  require(tidyverse)
  require(patchwork)
  require(multcomp)
  
  # 1. Get indices of top 10 from the emmeans object
  top10_idx <- order(as.data.frame(pipeline_emmeans)$emmean, 
                     decreasing = TRUE)[1:10]
  top10_means <- pipeline_emmeans[top10_idx]
  
  # 2. Run cld() on the top 10
  cld_top10 <- cld(top10_means,
                   adjust  = "sidak",
                   Letters = letters,
                   decreasing = TRUE) %>%
    as.data.frame()
  
  # 3. Apply custom back-transformation and prepare the data
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
  
  # 4. Create the Left Plot: The Parameter Matrix
  param_data <- df_clean %>%
    dplyr::select(
      G = Grinding, 
      T = Toxin, 
      P = Preprocessing, 
      Vs = Variable_Selection, 
      W = Weighting,
      D = Downsampling,
      Ds = Dataset,
      Pipeline_ID
    ) %>%
    pivot_longer(-Pipeline_ID, names_to = "Parameter", values_to = "Setting")
  
  param_data$Parameter <- factor(param_data$Parameter, 
                                 levels = c("G", "T", "P", "Vs", "W", "D", "Ds"))
  
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
  
  # 5. Create the Right Plot
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
      caption  = "Abbreviations: G (Grinding), T (Toxin), P (Spectral Preprocessing), Vs (Variable Selection), W (Weighting), D (Downsampling), Ds (Dataset)\nError bars = 95% Confidence Interval"
    ) +
    theme_bw(base_size = 11) +
    theme(
      axis.text.y     = element_blank(), 
      axis.ticks.y    = element_blank(),
      legend.position = "bottom",
      plot.caption    = element_text(size = 8, hjust = 1, color = "grey30"),
      panel.grid      = element_blank()
    )
  
  final_visualization <- p_matrix + p_scores + plot_layout(widths = c(1, 1.5))
  
  return(final_visualization)
}

plot_effect <- function(model, factor, title, y = "Estimated F1 Score", 
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
            geom_errorbar(aes(ymin = LCL, ymax = UCL), width = 0.1, size = 1) +
            geom_point(size = 4) +
            scale_color_viridis_d() +
            theme_classic() +
            labs(title = title, x = factor, y = y) +
            theme(
              plot.title = element_blank(),
              legend.position = "none"
            )
  return(output)
}

hist_plot <- function(data){
  output <- ggplot(data, aes(x = Deoxynivalenol, fill = Uncertainty)) +
    geom_histogram(boundary = 1000, binwidth = 100, position = "identity") + 
    geom_vline(xintercept = 1000, linetype = "dashed", color = "black", linewidth = 1) + 
    theme_classic(base_size = 14) +
    labs(
      x = "DON (ppb)",
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