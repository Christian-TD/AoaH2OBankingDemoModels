from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from teradataml.dataframe.copy_to import copy_to_sql
from aoa.stats import stats
from aoa.util.artefacts import save_plot
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import json
import h2o
import pandas as pd


def evaluate(data_conf, model_conf, **kwargs):
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

    # read training dataset from Teradata and convert to pandas
    test_df = DataFrame(data_conf["table"])
    test_tdf = test_df.select([feature_names + [target_name]])
    test_pdf = test_tdf.to_pandas()
    test_hdf = h2o.H2OFrame(test_pdf)

    # split data into X and y and factorize
    X_test, y_test = test_hdf.split_frame(ratios=[.7])
    y_test = X_test
    y_test[target_name] = y_test[target_name].asfactor()
    X_test = X_test[feature_names]

    print("Scoring")
    y_pred = model.predict(X_test)
    y_pred_pd = y_pred.as_data_frame()
    y_pred_tdf = y_pred_pd['predict']
    y_pred_tdf = pd.DataFrame(y_pred_tdf, columns=['predict'])
    y_pred_tdf = y_pred_tdf.rename(columns={'predict': 'y'})

    eval_metrics = model.model_performance(y_test)
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

    eval_metrics.plot(type="roc")
    save_plot('roc_curve.png')

    eval_metrics.plot(type="pr")
    save_plot('aucpr.png')

    model.learning_curve_plot()
    save_plot('learning_curve.png')

    cm = confusion_matrix(y_test.as_data_frame()['y'].values, y_pred_tdf.values)
    labels = ['no', 'yes']
    values = [0,1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion Matrix')
    fig.colorbar(cax)
    ax.set_xticks(values)
    ax.set_xticklabels(labels)
    ax.set_yticks(values)
    ax.set_yticklabels(labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    save_plot('confusion_matrix.png')

    try:
        model.varimp_plot()
        save_plot('feature_importance.png')
        model.shap_summary_plot(y_test)
        save_plot('shap_summary.png')
        model.shap_explain_row_plot(y_test, row_index=0)
        save_plot('shap_explain_row.png')
        fi = model.varimp(True)
        fix = fi[['variable', 'scaled_importance']]
        fis = fix.to_dict('records')
        feature_importance = {v['variable']: v['scaled_importance'] for (k, v) in enumerate(fis)}
    except:
        print("Warning: This model doesn't support feature importance (Stacked Ensemble)")
        feature_importance = {}
        model.residual_analysis_plot(y_test)
        save_plot('residual_analysis.png')

    predictions_table = "{}_tmp".format(data_conf["predictions"]).lower()
    copy_to_sql(df=y_pred_tdf, table_name=predictions_table, index=False, if_exists="replace", temporary=True)

    stats.record_evaluation_stats(test_tdf.iloc[:len(y_pred_tdf)], DataFrame(predictions_table), feature_importance)
