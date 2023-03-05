from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from aoa.stats import stats
from aoa.util.artefacts import save_plot
import os
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator


def train(context: ModelContext, **kwargs):
    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    categorical_names = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'y']

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame.from_query(context.dataset_info.sql)
    train_pdf = train_df.to_pandas(all_rows=True)

    # split data into X and y
    X_train = train_pdf[feature_names]
    y_train = train_pdf[target_name]

    print("Starting training...")

    # read training dataset from Teradata and convert to pandas
    h2o.init()
    train_df = DataFrame(data_conf["table"])
    train_df = train_df.select([feature_names + [target_name]])
    train_pdf = train_df.to_pandas(all_rows=True)
    train_hdf = h2o.H2OFrame(train_pdf)

    # split data into X and y and factorize
    X_train, y_train = train_hdf.split_frame(ratios=[.7])
    X_train[target_name] = X_train[target_name].asfactor()
    y_train[target_name] = y_train[target_name].asfactor()

    print("Starting training...")

    # Execute AutoML on training data
    model = H2OGradientBoostingEstimator(categorical_encoding = 'auto', ntrees = context.hyperparams['ntrees'], max_depth = context.hyperparams['max_depth'], min_rows = context.hyperparams['min_rows'], learn_rate = context.hyperparams['learn_rate'])
    model.train(x=feature_names, y=target_name, training_frame=X_train)

    print("Finished training")

    # export model artefacts
    current_path = os.path.abspath(os.getcwd())
    old_model = h2o.save_model(model, context.artifact_output_path)
    new_model = os.path.join(current_path, context.artifact_output_path, "model.h2o")
    if os.path.isfile(new_model):
        print("The file already exists")
    else:
        # Rename the file
        os.rename(old_model, new_model)
        
    # Saving as h2o mojo
    mojo = model.download_mojo(path=context.artifact_output_path, get_genmodel_jar=True)
    new_mojo = os.path.join(current_path, context.artifact_output_path, "mojo.zip")
    if os.path.isfile(new_mojo):
        print("The file already exists")
    else:
        # Rename the file
        os.rename(mojo, new_mojo)

    # Convert mojo to pmml
    cmd = "wget https://aoa-public-files.s3.amazonaws.com/jpmml-h2o-executable-1.1-SNAPSHOT.jar && java -jar ./jpmml-h2o-executable-1.1-SNAPSHOT.jar --mojo-input {} --pmml-output {}".format(new_mojo, os.path.join(current_path, context.artifact_output_path, "model.pmml"))
    result = os.system(cmd)
    if result > 0:
        raise OSError(result, "Error while trying to convert mojo to pmml")
    
    print("Saved trained model")

    model.varimp_plot()
    save_plot('Feature Importance')
    fi = model.varimp(True)
    fix = fi[['variable','scaled_importance']]
    fis = fix.to_dict('records')
    feature_importance = {v['variable']:v['scaled_importance'] for (k,v) in enumerate(fis)}

    stats.record_training_stats(train_df,
                       features=feature_names,
                       predictors=[target_name],
                       categorical=[target_name],
                       importance=feature_importance,
                       context=context)
