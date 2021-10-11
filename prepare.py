####### PREPARE CAR DATA #######

# standard imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
        
def clean_cars(df):
    '''
    Take in df and eliminates all database key columns, further refines to only single unit properties, handles all nulls with various methods
    '''
    # add general cleaning steps here
    
    # drop columns
    df = df.drop(columns=['calculatedbathnbr', # all present values are same as beds + baths, redundant
                          # add more columns to drop here
                         ]
                )
    
    # fix data types
    unit_type_dict = { # created dictionary for all unit type conversions
        'bedroomcnt' : 'int',
        # add more unit type conversions here
                    }
    df = df.astype(unit_type_dict) # convert unit types
    
    # remove any outliers
    cols_w_outliers = ['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'taxvaluedollarcnt'] # of remaining columns, these need outliers removed
    df = df = remove_outliers(df, cols_w_outliers) # call function to remove outliers using Tukey method
    
    # rename columns
    rename_dict = {
    'transactiondate' : 'sale_date',
    # add more renaming key value pairs here
    }
    df = df.rename(columns=rename_dict) # rename columns for readability
    
    # add new columns
    df['sale_month'] = df.sale_date.dt.month # create new columns for month numbers
    df['sale_week'] = df.sale_date.dt.week # create new columns for week numbers
    
    return df

def nulls_by_col(df):
    '''
    Takes in df and shows count of how many rows are null and percentage of total rows that are null
    '''
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = round(num_missing / rows * 100, 2)
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing

def nulls_by_row(df):
    '''
    Takes in df and shows count of how many columns are null and percentage of total columns that are null and value count of each unique combo
    '''
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = round(num_missing / df.shape[1] * 100, 2)
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(columns={'index': 'num_rows'}).reset_index()
    return rows_missing
    
def handle_missing_values(df, prop_required_columns=0.5, prop_required_row=0.5):
    '''
    Takes in df and thresholds for null proportions in each column and row and returns df with only columns and rows below threshold
    '''
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)
    return df
        
def remove_outliers(df, cols):
    '''
    Removes outliers that are outside of 1.5*IQR
    '''
    for col in cols:
        Q1 = np.percentile(df[col], 25, interpolation='midpoint')
        Q3 = np.percentile(df[col], 75, interpolation='midpoint')
        IQR = Q3 - Q1
        UB = Q3 + (1.5 * IQR)
        LB = Q1 - (1.5 * IQR)
        df = df[(df[col] < UB) & (df[col] > LB)]
    return df

def split_60(df):
    '''
    This function takes in a df and splits it into train, validate, and test dfs
    final proportions will be 60/20/20 for train/validate/test
    '''
    train_validate, test = train_test_split(df, test_size=0.2, random_state=527)
    train, validate = train_test_split(train_validate, test_size=.25, random_state=527)
    return train, validate, test

def split_80(df):
    '''
    This function takes in a df and splits it into train, validate, and test dfs
    final proportions will be 80/10/10 for train/validate/test
    '''
    train_validate, test = train_test_split(df, test_size=0.10, random_state=527)
    train, validate = train_test_split(train_validate, test_size=.11, random_state=527)
    return train, validate, test

def encode_scale(df, scaler, target):
    '''
    Takes in df and scaler of your choosing and returns split, encoded, and scaled df with unscaled columns dropped
    Doesn't scale specified target
    '''
    cat_cols = df.select_dtypes('object').columns.tolist()
    num_cols = df.select_dtypes('number').columns.tolist()
    num_cols.remove(target)
    df = pd.get_dummies(data=df, columns=cat_cols)
    train, validate, test = split_80(df)
    new_column_names = [c + '_scaled' for c in num_cols]
    
    # Fit the scaler on the train
    scaler.fit(train[num_cols])
    
    # transform train validate and test
    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[num_cols]), columns=new_column_names, index=train.index),
    ], axis=1)
    
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[num_cols]), columns=new_column_names, index=validate.index),
    ], axis=1)
 
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[num_cols]), columns=new_column_names, index=test.index),
    ], axis=1)
    
    # drop scaled columns
    train = train.drop(columns=num_cols)
    validate = validate.drop(columns=num_cols)
    test = test.drop(columns=num_cols)
    
    return train, validate, test

def encode_scale_final(df, scaler, target, cols_not_scale):
    '''
    Takes in df and scaler of your choosing and returns split, encoded, and scaled df with unscaled columns dropped
    Doesn't scale specified target and allows user to enter list of columns not to scale, if desired
    '''
    cat_cols = df.select_dtypes('object').columns.tolist()
    num_cols = df.select_dtypes('number').columns.tolist()
    num_cols.remove(target)
    num_cols = [col for col in num_cols if col not in cols_not_scale]
    df = pd.get_dummies(data=df, columns=cat_cols)
    train, validate, test = split_80(df)
    new_column_names = [c + '_scaled' for c in num_cols]
    
    # Fit the scaler on the train
    scaler.fit(train[num_cols])
    
    # transform train validate and test
    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[num_cols]), columns=new_column_names, index=train.index),
    ], axis=1)
    
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[num_cols]), columns=new_column_names, index=validate.index),
    ], axis=1)
    
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[num_cols]), columns=new_column_names, index=test.index),
    ], axis=1)
    
    # drop scaled columns
    train = train.drop(columns=num_cols)
    validate = validate.drop(columns=num_cols)
    test = test.drop(columns=num_cols)
    
    return train, validate, test