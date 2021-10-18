from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from teradataml.dataframe.copy_to import copy_to_sql
from aoa.stats import stats
from aoa.util.artefacts import save_plot
import os
import json
import h2o
import pandas as pd
from h2o.automl import H2OAutoML


def evaluate(data_conf, model_conf, **kwargs):
    current_path = os.path.abspath(os.getcwd())
    input_path = "artifacts/input"
    h2o.init()
    model = h2o.load_model(os.path.join(current_path, input_path, 'model.h2o'))

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    
    # read training dataset from Teradata and convert to pandas
    test_df = DataFrame(data_conf["table"])
    test_df = test_df.select([model.feature_names + [model.target_name]])
    test_hdf = h2o.H2OFrame(test_df.to_pandas())

    # split data into X and y and factorize
    X_test, y_test = test_hdf.split_frame(ratios=[.7])
    y_test = X_test
    y_test[model.target_name] = y_test[model.target_name].asfactor()
    X_test = X_test[model.feature_names]

    print("Scoring")
    y_pred = model.predict(X_test)
    y_pred_pd = y_pred.to_pandas()

    y_pred_tdf = pd.DataFrame(y_pred_pd['predict'], columns=[model.target_name])

    eval_metrics=model.model_performance(y_test)
    evaluation = {
        'Gini': '{:.6f}'.format(eval_metrics.gini()),
        'MSE': '{:.6f}'.format(eval_metrics.mse()),
        'RMSE': '{:.6f}'.format(eval_metrics.rmse()),
        'LogLoss': '{:.6f}'.format(eval_metrics.logloss()),
        'AUC': '{:.6f}'.format(eval_metrics.auc()),
        'AUCPR': '{:.6f}'.format(eval_metrics.aucpr()),
        'Accuracy': '{:.6f}'.format(eval_metrics.accuracy()[0][1]),
        'Mean Per-Class Error': '{:.6f}'.format(eval_metrics.mean_per_class_error()[0][1]),
        'F1 score': '{:.6f}'.format(eval_metrics.F1()[0][1]),
        'Precision': '{:.6f}'.format(eval_metrics.precision()[0][1]),
        'Sensitivity': '{:.6f}'.format(eval_metrics.sensitivity()[0][1]),
        'Specificity': '{:.6f}'.format(eval_metrics.specificity()[0][1]),
        'Recall': '{:.6f}'.format(eval_metrics.recall()[0][1])
    }

    artifacts_path = "artifacts/output"
    
    with open(os.path.join(current_path, artifacts_path, 'metrics.json'), "w+") as f:
        json.dump(evaluation, f)

    roc_plot = eval_metrics.plot(type = "roc")
    plt.savefig(os.path.join(current_path, artifacts_path, 'roc_curve.png'))

    pr_plot = eval_metrics.plot(type = "pr")
    plt.savefig(os.path.join(current_path, artifacts_path, 'aucpr.png'))
    
    lc_plot = model.learning_curve_plot()
    plt.savefig(os.path.join(current_path, artifacts_path, 'learning_curve.png'))
    
    import matplotlib.pyplot as plt
    try:
        vi_plot = model.varimp_plot()
        plt.savefig(os.path.join(current_path, artifacts_path, 'feature_importance.png'))
        shap_plot = model.shap_summary_plot(y_test)
        plt.savefig(os.path.join(current_path, artifacts_path, 'shap_summary.png'))
        shapr_plot = model.shap_explain_row_plot(y_test, row_index=0)
        plt.savefig(os.path.join(current_path, artifacts_path, 'shap_explain.png'))
        fi = model.varimp(True)
        fix = fi[['variable','scaled_importance']]
        fis = fix.to_dict('records')
        feature_importance = {v['variable']:v['scaled_importance'] for (k,v) in enumerate(fis)}
    except:
        print("Warning: This model doesn't have variable importances (Stacked Ensemble)")
        feature_importance = {}

    predictions_table = "{}_tmp".format(data_conf["predictions"]).lower()
    copy_to_sql(df=y_pred_tdf, table_name=predictions_table, index=False, if_exists="replace", temporary=True)

    stats.record_evaluation_stats(test_df, DataFrame(predictions_table), feature_importance)