import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from scipy import stats

def select_kbest(X, y, k):
    '''
    Takes in predictors, target, and number of features to select and returns that number of best features based on SelectKBest function
    '''
    kbest = SelectKBest(f_regression, k=k)
    kbest.fit(X, y)
    return X.columns[kbest.get_support()].tolist()

def rfe(X, y, n):
    '''
    Takes in predictors, target, and number of features to select and returns that number of best features based on RFE function
    '''
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=n)
    rfe.fit(X, y)
    return X.columns[rfe.get_support()].tolist()

def show_rfe_feature_ranking(X, y):
    '''
    Takes in predictors and target and returns feature ranking based on RFE function
    '''
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=1)
    rfe.fit(X, y)
    rankings = pd.Series(rfe.ranking_, index=X.columns)
    return rankings.sort_values()

def cluster_stats(df, target_col, group_by_col):
    '''
    Returns 1-sample t test for groups and pop mean of target_col grouped by group_by_col
    '''
    for cluster, _ in df.groupby(group_by_col):
        t, p = stats.ttest_1samp(df[target_col][df[group_by_col] == cluster], df[target_col].mean())
        print(f'One-sample T-test Results for Cluster {cluster}:\nT-Statistic: {t:.2f}\nP-value: {p:.3f}')
        if p < 0.05:
            print(f'We reject the null hypothesis, the mean log error for Cluster {cluster} is different than the overall population mean.\n')
        else:
            print(f'We fail to reject the null hypothesis, the mean log error for for Cluster {cluster} is not different than the overall population mean.\n')
            
def group_stats(df, target_col, group_by_col):
    '''
    Returns 1-sample t test for groups and pop mean of target_col grouped by group_by_col
    '''
    for group, _ in df.groupby(group_by_col):
        t, p = stats.ttest_1samp(df[target_col][df[group_by_col] == group], df[target_col].mean())
        print(f'One-sample T-test Results for the {group} subset:\nT-Statistic: {t:.2f}\nP-value: {p:.3f}')
        if p < 0.05:
            print(f'We reject the null hypothesis, the mean log error for the {group} subset is different than the overall population mean.\n')
        else:
            print(f'We fail to reject the null hypothesis, the mean log error for the {group} subset is not different than the overall population mean.\n')