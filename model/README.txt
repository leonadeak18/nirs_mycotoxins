Explanation regarding each file
1. preparation.qmd consists as follows:
	a. outliers identification
	b. various plots from the spectra to the metadata
	c. dataset splitting
2. There are 4 files to run the pipeline, which the output is in result folder:
	a. PLS_DA_austrian.R is an R script to run Austrian dataset without weighting treatment.
	b. PLS_DA_austrian_triangle.R is an R script to run Austrian dataset with weighting treatment.
	c. PLS_DA_mixed_all.R is an R script to run combination of Austrian and French dataset without weighting treatment.
	d. PLS_DA_mixed_all_triangle.R is an R script to run combination of Austrian and French dataset with weighting treatment.
3. pipeline_config.R is a helper for the pipeline, it works by importing all files and applying the observation weighting.
4. evaluation.qmd is a quarto document to evaluate the results with a GLMM model.
5. F1.R is an R script to evaluate the pipeline 1 and 2, which aimed to identify the effect of weighting.