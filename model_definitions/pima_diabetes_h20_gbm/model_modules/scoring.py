from teradataml import copy_to_sql, DataFrame
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)
import os
import h2o
import pandas as pd


def score(context: ModelContext, **kwargs):
    aoa_create_context()
    
    current_path = os.path.abspath(os.getcwd())
    h2o.init()
    model = h2o.load_model(os.path.join(current_path, context.artifact_input_path, 'model.h2o'))

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    # read scoring dataset from Teradata and convert to pandas
    features_tdf = DataFrame.from_query(context.dataset_info.sql)
    features_pdf = features_tdf.to_pandas(all_rows=True)
    features_hdf = h2o.H2OFrame(features_pdf)

    print("Scoring")
    predictions_hdf = model.predict(features_hdf[feature_names])

    print("Finished Scoring")

    # create result dataframe and store in Teradata
    predictions_pdf = predictions_hdf.as_data_frame()
    predictions_pdf = predictions_pdf.rename(columns={'predict': target_name})
    predictions_pdf[entity_key] = features_pdf.index.values
    predictions_pdf["job_id"] = context.job_id
    predictions_pdf["json_report"] = ""
    predictions_pdf = predictions_pdf[["job_id", entity_key, target_name, "json_report"]]

    copy_to_sql(df=predictions_pdf,
                schema_name=context.dataset_info.predictions_database,
                table_name=context.dataset_info.predictions_table,
                index=False,
                if_exists="append")

    print("Saved predictions in Teradata")

    # calculate stats
    predictions_df = DataFrame.from_query(f"""
        SELECT 
            * 
        FROM {context.dataset_info.get_predictions_metadata_fqtn()} 
            WHERE job_id = '{context.job_id}'
    """)

    record_scoring_stats(features_df=features_tdf, predicted_df=predictions_df, context=context)


# Add code required for RESTful API
class ModelScorer(object):

    def __init__(self, config=None):
        print("Initializing RESTful Model...")
        current_path = os.path.abspath(os.getcwd())
        h2o.init()
        self.model = h2o.load_model(os.path.join(current_path, context.artifact_input_path, 'model.h2o'))

        print("The RESTful model ready to accept requests.")

        from prometheus_client import Counter
        self.pred_class_counter = Counter('model_prediction_classes',
                                          'Model Prediction Classes', ['model', 'version', 'clazz'])

    def predict(self, data):
        print("Received prediction request with data: ")
        print(data)
        feature_names = context.dataset_info.feature_names
        data_df = pd.DataFrame(data=[data], columns=feature_names)
        data_h2o = h2o.H2OFrame(data_df)
        pred = self.model.predict(data_h2o)

        return pred.as_data_frame()
