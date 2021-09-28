LoadPackages <- function() {
    library("h2o")
    library("DBI")
    library("dplyr")
    library("tdplyr")
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

train <- function(data_conf, model_conf, ...) {
    print("Training model...")

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
    data
    
    # Load hyperparameters from model configuration
    hyperparams <- model_conf[["hyperParameters"]]

    # Initialize and convert dataframe to h2o
    print("Init h2o")
    h2o.init(nthreads = -1)
    train.hex <- as.h2o(data)
    splits <- h2o.splitFrame(train.hex, 0.75, seed=1234)

    # Train model
    model <- h2o.gbm(
        x = predictors,
        y = target, 
        training_frame = splits[[1]],
        categorical_encoding = 'auto', 
        ntrees = hyperparams$ntrees, 
        max_depth = hyperparams$max_depth, 
        min_rows = hyperparams$min_rows, 
        learn_rate = hyperparams$learn_rate
    )
    print("Model Trained!")

    # Save trained model
    print("Saving trained model...")
    #saveRDS(model, "artifacts/output/model.rds")
    
    # Save trained model in h2o format
    output.dir <- getwd()
    path.value <- file.path(output.dir, "artifacts/output")
    h2o.saveModel(object = model, path = path.value, force = TRUE)
    name <- file.path(path.value, "model.h2o") # destination file name at the same folder location
    file.rename(file.path(path.value, model@model_id), name)
}
