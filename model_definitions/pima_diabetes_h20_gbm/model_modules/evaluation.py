from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from teradataml.dataframe.copy_to import copy_to_sql
from aoa import (
    record_evaluation_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import json
import h2o
import pandas as pd


def evaluate(context: ModelContext, **kwargs):

    aoa_create_context()
    
    current_path = os.path.abspath(os.getcwd())
    h2o.init()
    model = h2o.load_model(os.path.join(current_path, context.artifact_input_path, 'model.h2o'))

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    # read training dataset from Teradata and convert to pandas
    test_df = DataFrame.from_query(context.dataset_info.sql)
    test_pdf = test_df.to_pandas(all_rows=True)
    test_hdf = h2o.H2OFrame(test_pdf)

    X_test = test_hdf[feature_names]
    y_test = test_hdf
    y_test[target_name] = y_test[target_name].asfactor()

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

    with open(os.path.join(current_path, context.artifact_output_path, 'metrics.json'), "w+") as f:
        json.dump(evaluation, f)

    eval_metrics.plot(type="pr")
    save_plot('AUC Precision Recall')

    eval_metrics.plot(type="roc")
    save_plot('ROC Curve')

    model.learning_curve_plot()
    save_plot('Learning Curve')

    cm = confusion_matrix(y_test.as_data_frame()[target_name].values, y_pred_tdf.values)
    labels = ['no', 'yes']
    values = [0,1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    ax.set_xticks(values)
    ax.set_xticklabels(labels)
    ax.set_yticks(values)
    ax.set_yticklabels(labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    save_plot('Confusion Matrix')
    model.varimp_plot()
    save_plot('Feature Importance')
    model.shap_summary_plot(y_test)
    save_plot('SHAP Summary')
    model.shap_explain_row_plot(y_test, row_index=0)
    save_plot('SHAP Explain Row')
    fi = model.varimp(True)
    fix = fi[['variable', 'scaled_importance']]
    fis = fix.to_dict('records')
    feature_importance = {v['variable']: v['scaled_importance'] for (k, v) in enumerate(fis)}

    predictions_table = "evaluation_preds_tmp"
    copy_to_sql(df=y_pred_tdf, table_name=predictions_table, index=False, if_exists="replace", temporary=True)

    #stats.record_evaluation_stats(test_tdf.iloc[:len(y_pred_tdf)], DataFrame(predictions_table), feature_importance)

    record_evaluation_stats(features_df=test_df,
                        predicted_df=DataFrame.from_query(f"SELECT * FROM {predictions_table}"),
                        importance=feature_importance,
                        context=context)