# Load necessary libraries
library(shiny)
library(reticulate)
library(DT)

# Ensure the Python environment is set up correctly
use_virtualenv("r-reticulate", required = TRUE)

# Import necessary Python packages
np <- import("numpy", delay_load = FALSE)
pd <- import("pandas", delay_load = FALSE)
tf <- import("tensorflow", delay_load = FALSE)
tfp <- import("tensorflow_probability", delay_load = FALSE)
az <- import("arviz", delay_load = FALSE)
meridian <- import("meridian", delay_load = FALSE)

# Import meridian's modules
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

# Define UI
ui <- fluidPage(
  titlePanel("Meridian Test App"),
  tabsetPanel(
    tabPanel("Load Data", 
             fileInput("csv_file", "Upload CSV File", accept = ".csv"),
             actionButton("load_dummy", "Load Dummy Dataset"),
             hr(),
             tableOutput("data_table")
    ),
    tabPanel("Configure Model",
             numericInput("roi_mu", "ROI Mu", 0.2),
             numericInput("roi_sigma", "ROI Sigma", 0.9),
             actionButton("configure_model", "Configure Model")
    ),
    tabPanel("Model Diagnostics",
             plotOutput("rhat_boxplot"),
             plotOutput("model_fit")
    ),
    tabPanel("Generate Results",
             tableOutput("summary_table"),
             textInput("start_date", "Start Date", "2021-01-25"),
             textInput("end_date", "End Date", "2024-01-15"),
             actionButton("generate_results", "Generate HTML Report")
    ),
    tabPanel("Budget Optimization",
             actionButton("optimize_budget", "Perform Optimization"),
             tableOutput("optimization_results")
    ),
    tabPanel("Save Model",
             textInput("save_path", "Save Path", "/saved_mmm.pkl"),
             actionButton("save_model", "Save Model")
    )
  )
)

# Define server logic
server <- function(input, output, session) {
  data <- reactiveVal(NULL)
  data_path <- reactiveVal(NULL)
  
  observeEvent(input$csv_file, {
    tryCatch({
      req(input$csv_file)
      df <- read.csv(input$csv_file$datapath)
      data(df)
      data_path(input$csv_file$datapath)
      output$data_table <- renderTable({
        dplyr::sample_n(df, 10)
      })
    }, error = function(e) {
      showNotification(paste("Error loading CSV file:", e$message), type = "error")
    })
  })
  
  observeEvent(input$load_dummy, {
    tryCatch({
      csv_path <- "C:/Users/bl896211/OneDrive - GSK/Documents/meridian/meridian/data/simulated_data/csv/geo_all_channels.csv"
      df <- read.csv(csv_path)
      data(df)
      data_path(csv_path)
      output$data_table <- renderTable({
        dplyr::sample_n(df, 10)
      })
    }, error = function(e) {
      showNotification(paste("Error loading dummy dataset:", e$message), type = "error")
    })
  })
  
  observeEvent(input$configure_model, {
    tryCatch({
      req(data())
      df <- data()
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
      loader <- load$CsvDataLoader(
        csv_path = data_path(),
        kpi_type = 'non_revenue',
        coord_to_columns = coord_to_columns,
        media_to_channel = correct_media_to_channel,
        media_spend_to_channel = correct_media_spend_to_channel
      )
      data_loaded <- loader$load()
      
      roi_mu <- input$roi_mu
      roi_sigma <- input$roi_sigma
      prior <- prior_distribution$PriorDistribution(
        roi_m = tfp$distributions$LogNormal(roi_mu, roi_sigma, name = constants$ROI_M)
      )
      model_spec <- spec$ModelSpec(prior = prior)
      mmm <- model$Meridian(input_data = data_loaded, model_spec = model_spec)
      
      system.time(mmm$sample_prior(500))
      system.time(mmm$sample_posterior(
        n_chains = 7, n_adapt = 500, n_burnin = 500, n_keep = 1000
      ))
      
      model_diagnostics <- visualizer$ModelDiagnostics(mmm)
      output$rhat_boxplot <- renderPlot({
        model_diagnostics$plot_rhat_boxplot()
      })
      
      model_fit <- visualizer$ModelFit(mmm)
      output$model_fit <- renderPlot({
        model_fit$plot_model_fit()
      })
      
      media_summary <- visualizer$MediaSummary(mmm)
      summary_table <- media_summary$summary_table()
      output$summary_table <- renderTable({
        summary_table
      })
    }, error = function(e) {
      showNotification(paste("Error configuring model:", e$message), type = "error")
    })
  })
  
  observeEvent(input$generate_results, {
    tryCatch({
      req(data())
      mmm_summarizer <- summarizer$Summarizer(mmm)
      filepath <- tempfile()
      start_date <- input$start_date
      end_date <- input$end_date
      mmm_summarizer$output_model_results_summary(
        'summary_output.html', filepath, start_date, end_date
      )
    }, error = function(e) {
      showNotification(paste("Error generating results:", e$message), type = "error")
    })
  })
  
  observeEvent(input$optimize_budget, {
    tryCatch({
      req(data())
      budget_optimizer <- optimizer$BudgetOptimizer(mmm)
      optimization_results <- budget_optimizer$optimize()
      output$optimization_results <- renderTable({
        optimization_results
      })
    }, error = function(e) {
      showNotification(paste("Error optimizing budget:", e$message), type = "error")
    })
  })
  
  observeEvent(input$save_model, {
    tryCatch({
      req(data())
      file_path <- input$save_path
      model$save_mmm(mmm, file_path)
    }, error = function(e) {
      showNotification(paste("Error saving model:", e$message), type = "error")
    })
  })
}

# Run the application 
shinyApp(ui = ui, server = server)