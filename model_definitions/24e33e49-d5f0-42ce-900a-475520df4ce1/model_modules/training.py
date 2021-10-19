from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from aoa.stats import stats
from aoa.util.artefacts import save_plot
import os
import h2o
from h2o.automl import H2OAutoML


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
    aml = H2OAutoML(max_models=hyperparams['max_models'], seed=hyperparams['seed'])
    aml.train(x=feature_names, y=target_name, training_frame=X_train)
    # model = aml.leader
    # Here we are getting the best GBM algorithm for demo purposes
    model = aml.get_best_model(algorithm="gbm")

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

    print("Saved trained model")

    import matplotlib.pyplot as plt
    try:
        ra_plot = model.varimp_plot()
        plt.savefig(os.path.join(current_path, artifacts_path, 'feature_importance.png'))
        fi = model.varimp(True)
        fix = fi[['variable','scaled_importance']]
        fis = fix.to_dict('records')
        feature_importance = {v['variable']:v['scaled_importance'] for (k,v) in enumerate(fis)}
    except:
        print("Warning: This model doesn't have variable importances (Stacked Ensemble)")
        ra_plot = aml.varimp_heatmap()
        plt.savefig(os.path.join(current_path, artifacts_path, 'feature_heatmap.png'))
        feature_importance = {}

    stats.record_training_stats(train_df,
                       features=feature_names,
                       predictors=[target_name],
                       categorical=categorical_names,
                       importance=feature_importance)