LoadPackages <- function() {
    library("h2o")
    library("DBI")
    library("dplyr")
    library("tdplyr")
    library("caret")
}

suppressPackageStartupMessages(LoadPackages())

Connect2Vantage <- function() {
    # Create Vantage connection using tdplyr
    con <- td_create_context(host = Sys.getenv("AOA_CONN_HOST"),
                             uid = Sys.getenv("AOA_CONN_USERNAME"),
                             pwd = Sys.getenv("AOA_CONN_PASSWORD"),
                             dType = 'native'
    )

    # Set connection context
    td_set_context(con)

    con
}

evaluate <- function(data_conf, model_conf, ...) {
    print("Evaluating model...")

    # Connect to Vantage
    con <- Connect2Vantage()

    # Create tibble from table in Vantage
    if ("schema" %in% data_conf) {
        table_name <- in_schema(data_conf$schema, data_conf$table)
    } else {
        table_name <- data_conf$table
    }
    table <- tbl(con, table_name)
    
    predictors <- c('age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'y')
    target <- 'y'

    # Create dataframe from tibble, selecting the necessary columns and mutating as necessary
    print("Load dataset")
    data <- table %>% select(all_of(predictors)) %>% as.data.frame()
    data$age <- as.integer(data$age)
    data$job <- as.factor(data$job)
    data$marital <- as.factor(data$marital)
    data$education <- as.factor(data$education)
    data$default <- as.factor(data$default)
    data$balance <- as.integer(data$balance)
    data$housing <- as.factor(data$housing)
    data$loan <- as.factor(data$loan)
    data$y <- as.factor(data$y)
    
    # Initialize and convert dataframe to h2o
    print("Init h2o")
    h2o.init(nthreads = -1)
    eval.hex <- as.h2o(data)
    splits <- h2o.splitFrame(eval.hex, 0.75, seed=1234)
    data <- splits[[2]]
    
    # Load trained model
    print("Loading h2o model")
    output.dir <- getwd()
    path.value <- file.path(output.dir, "artifacts/input")
    name <- file.path(path.value, "model.h2o")
    model <- h2o.loadModel(name)
    
    print("Running evaluation of h2o model")
    pred <- h2o.predict(model, data)
    pred_df <- as.data.frame(pred)
    tester <- as.data.frame(data)
    #cm <- confusionMatrix(tester$y, pred_df$predict)
    cm <- confusionMatrix(table(pred_df$predict, tester$y))
    png("artifacts/output/confusion_matrix.png", width = 860, height = 860)
    fourfoldplot(cm$table)
    dev.off()
    
    # Save model metrics
    metrics <- cm$overall
    write(jsonlite::toJSON(metrics, auto_unbox = TRUE, null = "null", keep_vec_names=TRUE), "artifacts/output/metrics.json")
    print("Model Evaluated!")
}
