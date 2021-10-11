# standard imports
import pandas as pd
import numpy as np

# sklearn imports
from sklearn.metrics import mean_squared_error, explained_variance_score


def make_metric_df(y_train_actual, y_train_pred, y_validate_actual, y_validate_pred, model_name, metric_df):
    '''
    Makes metric df to compare regression model metrics side-by-side
    '''
    if metric_df.size == 0:
        metric_df = pd.DataFrame(data=[
            {
                'model': model_name, 
                'RMSE_train': round(mean_squared_error(
                    y_train_actual,
                    y_train_pred,
                    squared=False),4),
                'RMSE_validate': round(mean_squared_error(
                    y_validate_actual,
                    y_validate_pred,
                    squared=False),4),
                'RMSE_diff' : round((mean_squared_error(
                    y_train_actual,
                    y_train_pred,
                    squared=False))
                    -
                    (mean_squared_error(
                    y_validate_actual,
                    y_validate_pred,
                    squared=False)),4),
                'R2_train': round(explained_variance_score(
                    y_train_actual,
                    y_train_pred),4),
                'R2_validate': round(explained_variance_score(
                    y_validate_actual,
                    y_validate_pred),4),
                "R2_diff" : round((explained_variance_score(
                    y_train_actual,
                    y_train_pred))
                    -
                    (explained_variance_score(
                    y_validate_actual,
                    y_validate_pred)),4)
            }])
        return metric_df
    else:
        return metric_df.append(
            {
                'model': model_name, 
                'RMSE_train': round(mean_squared_error(
                    y_train_actual,
                    y_train_pred,
                    squared=False),4),
                'RMSE_validate': round(mean_squared_error(
                    y_validate_actual,
                    y_validate_pred,
                    squared=False),4),
                'RMSE_diff' : round((mean_squared_error(
                    y_train_actual,
                    y_train_pred,
                    squared=False))
                    -
                    (mean_squared_error(
                    y_validate_actual,
                    y_validate_pred,
                    squared=False)),4),
                'R2_train': round(explained_variance_score(
                    y_train_actual,
                    y_train_pred),4),
                'R2_validate': round(explained_variance_score(
                    y_validate_actual,
                    y_validate_pred),4),
                "R2_diff" : round((explained_variance_score(
                    y_train_actual,
                    y_train_pred))
                    -
                    (explained_variance_score(
                    y_validate_actual,
                    y_validate_pred)),4)
            }, ignore_index=True)