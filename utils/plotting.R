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
    output <- output + scale_fill_manual(values = c("#1982c4", "#ef233c", "#b6d5e9", "#f7c1bb"))
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
    output <- output + scale_fill_manual(values = c("#1982c4", "#ef233c", "#b6d5e9", "#f7c1bb"))
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
              legend.title = element_blank()
            )
  if (n_groups == 2) {
    output <- output + scale_fill_manual(values = c("#1982c4", "#ef233c"))
  } else if (n_groups == 4) {
    output <- output + scale_fill_manual(values = c("#1982c4", "#ef233c", "#b6d5e9", "#f7c1bb"))
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