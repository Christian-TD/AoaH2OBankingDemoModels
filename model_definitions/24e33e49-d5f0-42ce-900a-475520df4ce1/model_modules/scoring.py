from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from teradataml.dataframe.copy_to import copy_to_sql
from aoa.stats import stats
from aoa.util.artefacts import save_plot
import os
import json
import h2o
import pandas as pd


def score(data_conf, model_conf, **kwargs):
    current_path = os.path.abspath(os.getcwd())
    input_path = "artifacts/input"
    h2o.init()
    model = h2o.load_model(os.path.join(current_path, input_path, 'model.h2o'))

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    feature_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan']
    target_name = 'y'

    # read scoring dataset from Teradata and convert to pandas
    features_tdf = DataFrame(data_conf["table"])
    features_tdf = features_tdf.select([feature_names])
    features_hdf = h2o.H2OFrame(features_tdf.to_pandas())

    print("Scoring")
    y_pred = model.predict(features_hdf)
    y_pred_pd = y_pred.as_data_frame()

    print("Finished Scoring")

    # create result dataframe and store in Teradata
    y_pred_pd = y_pred.as_data_frame()
    y_pred_tdf = y_pred_pd['predict']
    y_pred_tdf = pd.DataFrame(y_pred_tdf, columns=['predict'])
    y_pred_tdf = y_pred_tdf.rename(columns={'predict': target_name})
    copy_to_sql(df=y_pred_tdf, table_name=data_conf["predictions"], index=False, if_exists="replace")

    predictions_tdf = DataFrame(data_conf["predictions"])

    stats.record_scoring_stats(features_tdf, predictions_tdf)


# Add code required for RESTful API
class ModelScorer(object):

    def __init__(self, config=None):
        print("Initializing RESTful Model...")
        current_path = os.path.abspath(os.getcwd())
        input_path = "artifacts/input"
        h2o.init()
        self.model = h2o.load_model(os.path.join(current_path, input_path, 'model.h2o'))
        
        print("The RESTful model is ready to accept requests.")

        from prometheus_client import Counter
        self.pred_class_counter = Counter('model_prediction_classes',
                                          'Model Prediction Classes', ['model', 'version', 'clazz'])

    def predict(self, data):
        print("Prediction request received with data: ")
        print(data)
        feature_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan']
        data_df = pd.DataFrame(data=[data], columns=feature_names)
        data_h2o = h2o.H2OFrame(data_df)
        pred = self.model.predict(data_h2o).as_data_frame()

        # record the predicted class so we can check model drift (via class distributions)
        self.pred_class_counter.labels(model=os.environ["MODEL_NAME"],
                                       version=os.environ.get("MODEL_VERSION", "1.0"),
                                       clazz=str(int(1 if pred['predict'] == 'yes' else 0)).inc()

        return pred