library(reticulate)

virtualenv_create("r-reticulate") 
use_virtualenv("r-reticulate", required = TRUE)

# Install from local directory to have more control on versions
py_install("meridian", ignore_installed = FALSE, pip = TRUE)

# # Upgrade pandas to the latest version
# py_install("pandas", pip = TRUE, upgrade = TRUE)
# pd <- import("pandas", delay_load = FALSE)
# pd$`__version__`

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
  max_lag = 8, # Default: 8. NULL for abs flex
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
  n_chains = 7, n_adapt = 500, n_burnin = 500, n_keep = 1000))

######## STEP 3: MODEL DISGNOSTICS

# Assess convergence
model_diagnostics <- visualizer$ModelDiagnostics(mmm)
model_diagnostics$plot_rhat_boxplot()

# Assess model's fit
model_fit <- visualizer$ModelFit(mmm)
model_fit$plot_model_fit()

# Create Media Summary
media_summary <- visualizer$MediaSummary(mmm)
summary_table <- media_summary$summary_table()
print(summary_table)

######## STEP 4: GENERATE RESULTS & 2-PAGE HTML OUTPUT
# Docs: https://developers.google.com/meridian/docs/user-guide/generate-model-results-output

# Generate HTML file
mmm_summarizer <- summarizer$Summarizer(mmm)

# Define file path and dates
filepath <- '/folder'
start_date <- '2021-01-25' # min(df$time)
end_date <- '2024-01-15' # max(df$time)

# Output Model Results Summary
mmm_summarizer$output_model_results_summary(
  'summary_output.html', filepath, start_date, end_date)

######## STEP 5: BUDGET OPTIMIZATION REPORT
# Docs: https://developers.google.com/meridian/docs/user-guide/budget-optimization-scenarios

# Perform Optimization
budget_optimizer <- optimizer$BudgetOptimizer(mmm)
optimization_results <- budget_optimizer$optimize()

# Export HTML report
filepath <- '/folder'
optimization_results$output_optimization_summary('optimization_output.html', filepath)

######## STEP 6: SAVE AND EXPORT MODEL

# Define file path
file_path <- '/saved_mmm.pkl'
# Save Model
model$save_mmm(mmm, file_path)
# Load Model
mmm <- model$load_mmm(file_path)
