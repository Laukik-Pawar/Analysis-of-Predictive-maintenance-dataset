# Load required libraries
library(shiny)
library(ggplot2)
library(plotly)
library(DT)
library(randomForest)
library(dplyr)
library(caret)  # For additional evaluation metrics

# Load your data (modify path as needed)
data <- read.csv("C:/Users/Pawar's/Desktop/Predictive Maintanence/dataset/ds.csv")
# Preprocess the data
colnames(data) <- c("UDI", "Product.ID", "Type", "AirTemp", "ProcessTemp", "RotationalSpeed", "Torque", "ToolWear", "Target", "FailureType")

# Check for missing values
missing_values <- colSums(is.na(data))
print(missing_values)

# Label encoding for categorical variables
data$Type <- as.numeric(as.factor(data$Type))
data$FailureType <- as.factor(data$FailureType)  # Keep it as a factor for original labels
data$Target <- as.factor(data$Target)  # Keep Target as a factor for classification

# Fill missing values for numeric columns with the mean
data <- data %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Fill missing values for categorical columns with the mode
data$Type[is.na(data$Type)] <- as.numeric(names(sort(table(data$Type), decreasing = TRUE)[1]))

# Remove duplicates
data <- data %>% distinct()

# Split the data into training and test sets
set.seed(123)
train_indices <- sample(1:nrow(data), 0.7 * nrow(data))  # 70% training, 30% testing
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Ensure Target has the same levels in both train and test data
train_data$Target <- factor(train_data$Target)
test_data$Target <- factor(test_data$Target, levels = levels(train_data$Target))

# Train Random Forest on training data
model_rf <- randomForest(Target ~ ., data = train_data, ntree = 100)

# Define UI for dashboard
ui <- fluidPage(
  titlePanel("Predictive Maintenance Dashboard"),
  sidebarLayout(
    sidebarPanel(
      h4("Maintenance Prediction Overview"),
      p("This dashboard presents insights into equipment performance and failure prediction."),
      selectInput("feature", "Choose Feature for Plotting", 
                  choices = c("AirTemp", "ProcessTemp", "RotationalSpeed", "Torque", "ToolWear"))
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Summary Statistics", verbatimTextOutput("summary")),
        tabPanel("Failure Type Distribution", plotlyOutput("failurePlot")),
        tabPanel("Feature Correlation", plotlyOutput("featurePlot")),
        tabPanel("Data Table", DTOutput("dataTable")),
        tabPanel("Model Predictions", plotlyOutput("predictionPlot")),
        tabPanel("Model Accuracy", verbatimTextOutput("accuracy"))
      )
    )
  )
)

# Define server logic
# Define server logic
server <- function(input, output) {
  # Summary of dataset
  output$summary <- renderPrint({
    summary(data)
  })
  
  # Plot Failure Type Distribution
  output$failurePlot <- renderPlotly({
    p <- ggplot(data, aes(x = FailureType)) + 
      geom_bar(fill = "skyblue") + 
      labs(title = "Failure Type Distribution", x = "Failure Type", y = "Count") +
      theme_minimal()
    ggplotly(p)
  })
  
  # Plot selected feature against Target
  output$featurePlot <- renderPlotly({
    p <- ggplot(data, aes_string(x = input$feature, y = "Target")) +
      geom_point(aes(color = as.factor(Target))) +
      labs(title = paste("Feature vs Target:", input$feature), x = input$feature, y = "Target") +
      theme_minimal()
    ggplotly(p)
  })
  
  # Data table view
  output$dataTable <- renderDT({
    datatable(data)
  })
  
  # Model Prediction Plot
  output$predictionPlot <- renderPlotly({
    predictions_rf <- predict(model_rf, test_data)
    
    actual_vs_predicted <- data.frame(
      Actual = test_data$Target,
      Predicted = predictions_rf
    )
    
    p <- ggplot(actual_vs_predicted, aes(x = Actual, y = Predicted)) +
      geom_point(color = "blue") +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
      labs(title = "Actual vs Predicted Failures", x = "Actual Failure", y = "Predicted Failure") +
      theme_minimal()
    
    ggplotly(p)
  })
  
  # Model Accuracy and Evaluation Metrics
  output$accuracy <- renderPrint({
    # Make predictions on test data
    predictions_rf <- predict(model_rf, test_data)
    
    # Ensure the predicted values have the same levels as the test target
    predictions_rf <- factor(predictions_rf, levels = levels(test_data$Target))
    
    # Generate confusion matrix
    confusion_matrix <- confusionMatrix(predictions_rf, test_data$Target)
    
    # Extract and print only relevant metrics
    cat("Confusion Matrix:\n")
    print(confusion_matrix$table)
    
    cat("\nAccuracy:", round(confusion_matrix$overall['Accuracy'] * 100, 2), "%\n")
    
    cat("\nOther Performance Metrics:\n")
    print(confusion_matrix$byClass)
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
