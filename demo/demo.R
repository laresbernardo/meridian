library(reticulate)
# install_python(version = '3.11')
virtualenv_create("r-meridian", version = "3.11")
use_virtualenv("r-meridian", required = TRUE)
py_config()

# Install meridian (local dir) and some other deps
py_install("meridian", ignore_installed = FALSE, pip = TRUE)
# py_install("immutabledict", pip = TRUE, upgrade = TRUE)
# py_install("tf_keras", pip = TRUE, upgrade = TRUE)
# py_install("joblib", pip = TRUE, upgrade = TRUE)
# py_install("altair", pip = TRUE, upgrade = TRUE)

# Import libraries
np <- import("numpy", delay_load = FALSE)
pd <- import("pandas", delay_load = FALSE)
tf <- import("tensorflow", delay_load = FALSE)
tfp <- import("tensorflow_probability", delay_load = FALSE)
az <- import("arviz", delay_load = FALSE)
meridian <- import("meridian", delay_load = FALSE)

# Import meridian's modules
if (TRUE) {
  constants <- import("meridian.constants")
  load <- import("meridian.data.load")
  test_utils <- import("meridian.data.test_utils")
  model <- import("meridian.model.model")
  spec <- import("meridian.model.spec")
  prior_distribution <- import("meridian.model.prior_distribution")
  optimizer <- import("meridian.analysis.optimizer")
  analyzer <- import("meridian.analysis.analyzer")
  visualizer <- import("meridian.analysis.visualizer")
  summarizer <- import("meridian.analysis.summarizer")
  formatter <- import("meridian.analysis.formatter") 
}

# Check if GPU is available
try({
  suppressPackageStartupMessages(library(tensorflow))
  tf_gpu_configured()
})

######## STEP 1: LOAD THE DATA
# Docs: https://developers.google.com/meridian/docs/user-guide/collect-data

# Load (dummy) data to inspect
csv_path <- "meridian/data/simulated_data/csv/geo_all_channels.csv"
df <- read.csv(csv_path)
head(df, 10)

# Map the column names to their corresponding variable types
coord_to_columns <- load$CoordToColumns(
  time = 'time',
  geo = 'geo',
  controls = c('GQV', 'Competitor_Sales'),
  population = 'population',
  kpi = 'revenue_per_conversion',
  revenue_per_kpi = 'revenue_per_conversion',
  media = c(
    'Channel0_impression',
    'Channel1_impression',
    'Channel2_impression',
    'Channel3_impression',
    'Channel4_impression'
  ),
  media_spend = c(
    'Channel0_spend',
    'Channel1_spend',
    'Channel2_spend',
    'Channel3_spend',
    'Channel4_spend'
  ),
  organic_media = c('Organic_channel0_impression'),
  non_media_treatments = c('Promo')
)
# Create the dictionaries
# Map the media variables and the media spends to the 
# designated channel names intended for outputs
correct_media_to_channel <- dict(
  'Channel0_impression' = 'Channel_0',
  'Channel1_impression' = 'Channel_1',
  'Channel2_impression' = 'Channel_2',
  'Channel3_impression' = 'Channel_3',
  'Channel4_impression' = 'Channel_4'
)
correct_media_spend_to_channel <- dict(
  'Channel0_spend' = 'Channel_0',
  'Channel1_spend' = 'Channel_1',
  'Channel2_spend' = 'Channel_2',
  'Channel3_spend' = 'Channel_3',
  'Channel4_spend' = 'Channel_4'
)

# Load the data
loader <- load$CsvDataLoader(
  csv_path = csv_path,
  kpi_type = 'non_revenue',
  coord_to_columns = coord_to_columns,
  media_to_channel = correct_media_to_channel,
  media_spend_to_channel = correct_media_spend_to_channel
)
# Error in py_call_impl(callable, call_args$unnamed, call_args$named) : 
#   AttributeError: 'Series' object has no attribute 'stack'
data <- loader$load()

######## STEP 2: CONFIGURE THE MODEL
# Docs: https://developers.google.com/meridian/docs/user-guide/configure-model

# Set the ROI prior Mu and Sigma for each media channel
roi_mu <- c(0.2, 0.3, 0.4, 0.3, 0.3)
roi_sigma <- c(0.7, 0.9, 0.6, 0.7, 0.6)
# Create the Prior Distribution
prior <- prior_distribution$PriorDistribution(
  roi_m = tfp$distributions$LogNormal(roi_mu, roi_sigma, name = constants$ROI_M)
)
# Create the Model Specification
n_times <- length(unique(df$time))
knots <- round(0.8 * n_times)
model_spec <- spec$ModelSpec(
  prior = prior,
  media_effects_dist = 'log_normal',
  hill_before_adstock = FALSE,
  max_lag = 8L, # Default: 8. NULL for abs flex
  unique_sigma_for_each_geo = FALSE,
  paid_media_prior_type = 'roi',
  roi_calibration_period = NULL,
  rf_roi_calibration_period = NULL,
  knots = knots, # to control for seasonality. 1 for intercept only, max is number of periods
  baseline_geo = NULL,
  holdout_id = NULL,
  control_population_scaling_id = NULL)
# Assuming `data` is already loaded as per your previous steps
mmm <- model$Meridian(input_data = data, model_spec = model_spec)

## NOTE: If you are using T4 GPU runtime, this step may take about 
## 10 minutes for the provided demo data set.

# n_chains: The number of chains to be sampled in parallel. 
# To reduce memory consumption, you can use a list of integers to allow for sequential 
# MCMC sampling calls. Given a list, each element in the sequence corresponds to the 
# n_chains argument for a call to windowed_adaptive_nuts.
#
# n_adapt: The number of MCMC draws per chain, during which step size and kernel 
# are adapted. These draws are always excluded.
# 
# n_burnin: An additional number of MCMC draws, per chain, to be excluded after 
# the step size and kernel are fixed. These additional draws may be needed to ensure 
# that all chains reach the stationary distribution after adaptation is completed, 
# but in practice we often find that the chains reach the stationary distribution 
# during adaptation and that n_burnin=0 is sufficient.
# 
# n_keep: The number of MCMC draws, per chain, to keep for the model analysis and results.

# Sample from the prior distribution
system.time(mmm$sample_prior(500))
# Sample from the posterior distribution
system.time(mmm$sample_posterior(
  n_chains = 7L, n_adapt = 500L, 
  n_burnin = 500L, n_keep = 1000L))

######## STEP 3: MODEL DISGNOSTICS

# Assess convergence
model_diagnostics <- visualizer$ModelDiagnostics(mmm)
model_diagnostics$plot_rhat_boxplot()

# Assess model's fit
model_fit <- visualizer$ModelFit(mmm)
model_fit$plot_model_fit()

# Create Media Summary
media_summary <- visualizer$MediaSummary(mmm)
summary_table <- media_summary$summary_table(
  include_prior = TRUE,
  include_posterior = TRUE,
  include_non_paid_channels = TRUE
)
print(summary_table)

######## STEP 4: GENERATE RESULTS & 2-PAGE HTML OUTPUT
# Docs: https://developers.google.com/meridian/docs/user-guide/generate-model-results-output

# Generate HTML file
mmm_summarizer <- summarizer$Summarizer(mmm)

# Define file path and dates
filepath <- 'demo'
start_date <- '2021-01-25' # min(df$time)
end_date <- '2024-01-15' # max(df$time)

# Output Model Results Summary
mmm_summarizer$output_model_results_summary(
  'summary_output.html', filepath, start_date, end_date)

######## STEP 5: BUDGET OPTIMIZATION REPORT
# Docs: https://developers.google.com/meridian/docs/user-guide/budget-optimization-scenarios

# use_posterior: Boolean. If `True`, then the budget is optimized based on
# the posterior distribution of the model. Otherwise, the prior
# distribution is used.
# selected_times: Tuple containing the start and end time dimension
# coordinates for the duration to run the optimization on. Selected time
# values should align with the Meridian time dimension coordinates in the
# underlying model. By default, all times periods are used. Either start
# or end time component can be `None` to represent the first or the last
# time coordinate, respectively.
# fixed_budget: Boolean indicating whether it's a fixed budget optimization
# or flexible budget optimization. Defaults to `True`. If `False`, must
# specify either `target_roi` or `target_mroi`.
# budget: Number indicating the total budget for the fixed budget scenario.
# Defaults to the historical budget.
# pct_of_spend: Numeric list of size `n_total_channels` containing the
# percentage allocation for spend for all media and RF channels. The order
# must match `InputData.media` with values between 0-1, summing to 1. By
# default, the historical allocation is used. Budget and allocation are
# used in conjunction to determine the non-optimized media-level spend,
# which is used to calculate the non-optimized performance metrics (for
# example, ROI) and construct the feasible range of media-level spend with
# the spend constraints.
# spend_constraint_lower: Numeric list of size `n_total_channels` or float
# (same constraint for all channels) indicating the lower bound of
# media-level spend. The lower bound of media-level spend is `(1 -
# spend_constraint_lower) * budget * allocation)`. The value must be
# between 0-1. Defaults to `0.3` for fixed budget and `1` for flexible.
# spend_constraint_upper: Numeric list of size `n_total_channels` or float
# (same constraint for all channels) indicating the upper bound of
# media-level spend. The upper bound of media-level spend is `(1 +
# spend_constraint_upper) * budget * allocation)`. Defaults to `0.3` for
# fixed budget and `1` for flexible.
# target_roi: Float indicating the target ROI constraint. Only used for
# flexible budget scenarios. The budget is constrained to when the ROI of
# the total spend hits `target_roi`.
# target_mroi: Float indicating the target marginal ROI constraint. Only
# used for flexible budget scenarios. The budget is constrained to when
# the marginal ROI of the total spend hits `target_mroi`.
# gtol: Float indicating the acceptable relative error for the budget used
# in the grid setup. The budget will be rounded by `10*n`, where `n` is
# the smallest integer such that `(budget - rounded_budget)` is less than
# or equal to `(budget * gtol)`. `gtol` must be less than 1.
# use_optimal_frequency: If `True`, uses `optimal_frequency` calculated by
# trained Meridian model for optimization. If `False`, uses historical
# frequency.
# confidence_level: The threshold for computing the confidence intervals.
# batch_size: Maximum draws per chain in each batch. The calculation is run
# in batches to avoid memory exhaustion. If a memory error occurs, try
# reducing `batch_size`. The calculation will generally be faster with
# larger `batch_size` values.

# Perform Optimization
budget_optimizer <- optimizer$BudgetOptimizer(mmm)
optimization_results <- budget_optimizer$optimize(
  use_posterior = TRUE,
  selected_times = NULL,
  fixed_budget = TRUE,
  budget = NULL,
  pct_of_spend = NULL,
  spend_constraint_lower = 0.5,
  spend_constraint_upper = 2,
  target_roi = NULL,
  target_mroi = NULL
)
# Plot results
optimization_results$plot_budget_allocation(optimized = TRUE)
optimization_results$plot_response_curves(n_top_channels = 3L)

# Export HTML report
filepath <- 'demo'
optimization_results$output_optimization_summary(
  'optimization_output.html', filepath)

######## STEP 6: SAVE AND EXPORT MODEL

# Define file path
file_path <- 'demo/saved_mmm.pkl'
# Save Model
model$save_mmm(mmm, file_path)
# Load Model
mmm2 <- model$load_mmm(file_path)
