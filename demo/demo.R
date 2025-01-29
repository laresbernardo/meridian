library(reticulate)

virtualenv_create("r-reticulate")
use_virtualenv("r-reticulate", required = TRUE)

# Install from local directory to have more control on versions
py_install("meridian", ignore_installed = FALSE, pip = TRUE)

# Import libraries
np <- import("numpy", delay_load = FALSE)
pd <- import("pandas", delay_load = FALSE)
tf <- import("tensorflow", delay_load = FALSE)
tfp <- import("tensorflow_probability", delay_load = FALSE)
az <- import("arviz", delay_load = FALSE)
meridian <- import("meridian", delay_load = FALSE)

# Import the necessary modules
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

# Check if GPU is available
try(tensorflow::tf_gpu_configured())

######## STEP 1: LOAD THE DATA

# Load (dummy) data to inspect
csv_path <- "meridian/data/simulated_data/csv/geo_all_channels.csv"
df <- read.csv(csv_path)
head(df, 10)

# Map the column names to their corresponding variable types
# Defs: https://developers.google.com/meridian/docs/user-guide/collect-data
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
