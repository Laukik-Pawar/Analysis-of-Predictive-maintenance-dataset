# Load required libraries
library(shiny)
library(ggplot2)
library(plotly)
library(DT)
library(e1071)  # For SVM
library(dplyr)
library(caret)  # For confusion matrix and other metrics

# Load the dataset
data <- read.csv("C:/Users/Pawar's/Desktop/Predictive Maintanence/dataset/ds.csv")

# Preprocess the data
colnames(data) <- c("UDI", "Product.ID", "Type", "AirTemp", "ProcessTemp", "RotationalSpeed", "Torque", "ToolWear", "Target", "FailureType")


# Label encoding for categorical variables
data$Type <- as.numeric(as.factor(data$Type))
data$FailureType <- as.factor(data$FailureType)  # Keep as factor for original labels
data$Target <- as.factor(data$Target)  # Ensure Target is a factor for multi-class classification

# Remove duplicates
# Fill missing values for numeric columns with the mean
data <- data %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Fill missing values for categorical columns with the mode
data$Type[is.na(data$Type)] <- as.numeric(names(sort(table(data$Type), decreasing = TRUE)[1]))

# Remove duplicates
data <- data %>% distinct()


# Split the data into training and test sets
set.seed(123)
train_indices <- sample(1:nrow(data), 0.7 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Remove Product.ID column from both train and test data
train_data <- train_data %>% select(-Product.ID)
test_data <- test_data %>% select(-Product.ID)

# Train SVM model for multi-class classification

model_svm <- svm(Target ~ ., data = train_data, cost = 0.1, kernel = "polynomial", probability = TRUE)
#model_svm <- svm(Target ~ ., data = train_data, cost = 0.01, probability = TRUE)

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
      geom_point(aes(color = Target)) +
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
    predictions_svm <- predict(model_svm, test_data, decision.values = TRUE)
    actual_vs_predicted <- data.frame(
      Actual = test_data$Target,
      Predicted = predictions_svm
    )
    p <- ggplot(actual_vs_predicted, aes(x = Actual, y = Predicted)) +
      geom_point(color = "blue") +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
      labs(title = "Actual vs Predicted Failures", x = "Actual Failure", y = "Predicted Failure") +
      theme_minimal()
    ggplotly(p)
  })
  
  # Model Accuracy
  output$accuracy <- renderPrint({
    # Predictions and actual labels
    predictions <- predict(model_svm, test_data)
    
    # Confusion Matrix
    confusion_matrix <- confusionMatrix(predictions, test_data$Target)
    
    # Extracting accuracy and other metrics
    accuracy <- confusion_matrix$overall['Accuracy']
    precision <- confusion_matrix$byClass['Pos Pred Value']
    recall <- confusion_matrix$byClass['Sensitivity']
    f1_score <- 2 * ((precision * recall) / (precision + recall))
    
    # Display confusion matrix and metrics
    print(confusion_matrix)
    cat("Accuracy:", round(accuracy * 100, 2), "%\n")
    cat("Precision:", round(precision, 2), "\n")
    cat("Recall:", round(recall, 2), "\n")
    cat("F1-Score:", round(f1_score, 2), "\n")
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
