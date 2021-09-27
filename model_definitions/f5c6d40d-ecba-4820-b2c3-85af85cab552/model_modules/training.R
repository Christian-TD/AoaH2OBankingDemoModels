library("gbm")


train <- function(data_conf, model_conf, ...) {

    # implement training logic

    # clean the model (R stores the dataset on the model..
    # this changes per model so be careful
    model$data <- NULL

    saveRDS(model, "artifacts/output/model.rds")
}
