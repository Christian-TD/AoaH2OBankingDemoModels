from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from aoa.stats import stats
from aoa.util.artefacts import save_plot
import os
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    feature_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan']
    target_name = 'y'
    categorical_names = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'y']

    # read training dataset from Teradata and convert to pandas
    h2o.init()
    train_df = DataFrame(data_conf["table"])
    train_df = train_df.select([feature_names + [target_name]])
    train_hdf = h2o.H2OFrame(train_df.to_pandas())

    # split data into X and y and factorize
    X_train, y_train = train_hdf.split_frame(ratios=[.7])
    X_train[target_name] = X_train[target_name].asfactor()
    y_train[target_name] = y_train[target_name].asfactor()

    print("Starting training...")

    # Execute AutoML on training data
    model = H2OGradientBoostingEstimator(categorical_encoding = 'auto', ntrees = hyperparams['ntrees'], max_depth = hyperparams['max_depth'], min_rows = hyperparams['min_rows'], learn_rate = hyperparams['learn_rate'])
    model.train(x=feature_names, y=target_name, training_frame=X_train)

    print("Finished training")

    # export model artefacts
    current_path = os.path.abspath(os.getcwd())
    artifacts_path = "artifacts/output"
    old_model = h2o.save_model(model, artifacts_path)
    new_model = os.path.join(current_path, artifacts_path, "model.h2o")
    if os.path.isfile(new_model):
        print("The file already exists")
    else:
        # Rename the file
        os.rename(old_model, new_model)
        
    # Saving as h2o mojo
    mojo = model.download_mojo(path=artifacts_path, get_genmodel_jar=True)
    new_mojo = os.path.join(current_path, artifacts_path, "mojo.zip")
    if os.path.isfile(new_mojo):
        print("The file already exists")
    else:
        # Rename the file
        os.rename(mojo, new_mojo)

    # Convert mojo to pmml
    cmd = "wget https://aoa-public-files.s3.amazonaws.com/jpmml-h2o-executable-1.1-SNAPSHOT.jar && java -jar ./jpmml-h2o-executable-1.1-SNAPSHOT.jar --mojo-input {} --pmml-output {}".format(new_mojo, os.path.join(current_path, artifacts_path, "model.pmml"))
    result = os.system(cmd)
    if result > 0:
        raise OSError(result, "Error while trying to convert mojo to pmml")
    
    print("Saved trained model")

    model.varimp_plot()
    save_plot('feature_importance.png')
    fi = model.varimp(True)
    fix = fi[['variable','scaled_importance']]
    fis = fix.to_dict('records')
    feature_importance = {v['variable']:v['scaled_importance'] for (k,v) in enumerate(fis)}

    stats.record_training_stats(train_df,
                       features=feature_names,
                       predictors=[target_name],
                       categorical=categorical_names,
                       importance=feature_importance)
