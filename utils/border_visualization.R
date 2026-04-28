library(ggplot2)
library(dplyr)

# 1. Set your parameters
threshold <- 1000
uncertainty <- 0.30

# 2. Create a sequence of simulated sample concentrations
df <- data.frame(val = seq(500, 1500, by = 5))

# 3. Calculate Probability Weights
df <- df %>%
  mutate(
    stdev = threshold * (uncertainty / 2),
    z_score = abs(threshold - val) / stdev,
    weight_prob = (pnorm(z_score) - 0.5) * 2
  )

# 4. Calculate Triangle Weights
lower_bound <- threshold - (threshold * uncertainty)
upper_bound <- threshold + (threshold * uncertainty)
lower_slope <- 1 / (threshold - lower_bound)
upper_slope <- 1 / (upper_bound - threshold)

df <- df %>%
  mutate(
    weight_triangle = case_when(
      val <= lower_bound ~ 1,
      val >= upper_bound ~ 1,
      val < threshold    ~ lower_slope * (threshold - val),
      val > threshold    ~ upper_slope * (val - threshold),
      val == threshold   ~ 0
    )
  )

# 5. Plot the comparison
ggplot(df, aes(x = val)) +
  # The probability curve
  geom_line(aes(y = weight_prob, color = "Probability (pnorm)"), linewidth = 1.2) +
  # The triangle curve
  geom_line(aes(y = weight_triangle, color = "Linear Triangle"), linewidth = 1.2, linetype = "dashed") +
  # The threshold line
  geom_vline(xintercept = threshold, linetype = "dotted", color = "black", linewidth = 0.8) +
  
  # Formatting
  scale_color_manual(values = c("Probability (pnorm)" = "#2ca02c", "Linear Triangle" = "#d62728")) +
  labs(
    title = "PLS-DA Sample Weighting: Probability vs. Triangle",
    subtitle = paste0("Threshold = ", threshold, " µg/kg | Uncertainty = ", uncertainty * 100, "%"),
    x = "Measured Concentration (µg/kg)",
    y = "Assigned Model Weight",
    color = "Penalty Method"
  ) +
  geom_hline(yintercept = 0.95, color = "red", linewidth = 1) +
  geom_vline(xintercept = c(lower_bound, upper_bound), linetype = "dashed", color = "blue", linewidth = 0.8) +
  scale_x_continuous(breaks = seq(500, 1500, by = 100)) +
  theme_classic(base_size = 14) +
  theme(legend.position = "bottom")