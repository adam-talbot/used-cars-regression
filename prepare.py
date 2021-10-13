####### PREPARE CAR DATA #######

# standard imports
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
        
def clean_cars(df):
    '''
    Get data ready for exploration and modeling, see comments below for more details
    '''
    df = df.reset_index(drop=True) # reset index for this subset of the dataset
    df = df.drop_duplicates() # drop any duplicate rows
    df = handle_missing_values(df) # drop all columns and rows with more than half of data missing
    df.back_legroom = df.back_legroom.str.split(' ', expand=True)[0] # split and only keep number
    df.back_legroom = pd.to_numeric(df.back_legroom, errors='coerce') # convert to float
    df.back_legroom = df.back_legroom.fillna(round(df.back_legroom.mean(),2)) # fill missing values with mean
    df.franchise_dealer = np.where(df.franchise_dealer == True, 'Yes', 'No') # change from bool to 1 or 0
    df.front_legroom = df.front_legroom.str.split(' ', expand=True)[0] # split and only keep number
    df.front_legroom = pd.to_numeric(df.front_legroom, errors='coerce') # convert to float
    df.front_legroom = df.front_legroom.fillna(round(df.front_legroom.mean(),2)) # fill missing values with mean
    df.fuel_tank_volume = df.fuel_tank_volume.str.split(' ', expand=True)[0] # split and only keep number
    df.fuel_tank_volume = pd.to_numeric(df.fuel_tank_volume, errors='coerce') # convert to float
    df.fuel_tank_volume = df.fuel_tank_volume.fillna(round(df.fuel_tank_volume.mean(),2)) # fill missing values with mean
    df.height = df.height.str.split(' ', expand=True)[0] # split and only keep number
    df.height = pd.to_numeric(df.height, errors='coerce') # convert to float
    df.height = df.height.fillna(round(df.height.mean(),2)) # fill missing values with mean
    df.length = df.length.str.split(' ', expand=True)[0] # split and only keep number
    df.length = pd.to_numeric(df.length, errors='coerce') # convert to float
    df.length = df.length.fillna(round(df.length.mean(),2)) # fill missing values with mean
    top_six = ['Ford', 'Chevrolet', 'Toyota', 'Honda', 'Nissan', 'Jeep'] # biggest proportion, top 6
    df.make_name = df.make_name.apply(lambda x: x if x in top_six else 'Other') # group all others as 'Other'
    df.maximum_seating = df.maximum_seating.str.split(' ', expand=True)[0] # split and only keep number
    df.maximum_seating = pd.to_numeric(df.maximum_seating, errors='coerce') # convert to float
    df.maximum_seating = df.maximum_seating.fillna(df.maximum_seating.median()) # fill missing values with median
    df.wheelbase = df.wheelbase.str.split(' ', expand=True)[0] # split and only keep number
    df.wheelbase = pd.to_numeric(df.wheelbase, errors='coerce') # convert to float
    df.wheelbase = df.wheelbase.fillna(round(df.wheelbase.mean(),2)) # fill missing values with mean
    df.width = df.width.str.split(' ', expand=True)[0] # split and only keep number
    df.width = pd.to_numeric(df.width, errors='coerce') # convert to float
    df.width = df.width.fillna(round(df.width.mean(),2)) # convert to float
    cols_to_drop = [
        'vin', 
        'dealer_zip',
        'description',
        'engine_cylinders',
        'engine_type',
        'exterior_color',
        'interior_color',
        'interior_color',
        'daysonmarket',
        'listed_date',
        'listing_id',
        'main_picture_url',
        'major_options',
        'franchise_make',
        'model_name',
        'power',
        'savings_amount',
        'seller_rating',
        'sp_id',
        'sp_name',
        'torque',
        'transmission_display',
        'trimId',
        'trim_name',
        'wheel_system_display',
        'is_new']
    df = df.drop(columns=cols_to_drop)
    # impute remaining missing values
    df.city_fuel_economy = df.city_fuel_economy.fillna(value=round(df.city_fuel_economy.mean(),0)) # mean
    df.engine_displacement = df.engine_displacement.fillna(value=round(df.engine_displacement.mean(),0)) # mean
    df.highway_fuel_economy = df.highway_fuel_economy.fillna(value=round(df.highway_fuel_economy.mean(),0)) # mean
    df.horsepower = df.horsepower.fillna(value=round(df.horsepower.mean(),0)) # mean
    df.wheel_system = df.wheel_system.fillna(value=df.wheel_system.mode()[0]) # mode
    df.mileage = df.mileage.fillna(value=round(df.mileage.mean(),0)) # mean
    df.fuel_type = df.fuel_type.fillna(value=df.fuel_type.mode()[0]) # mode
    df.transmission = df.transmission.fillna(value=df.transmission.mode()[0]) # mode
    df.body_type = df.body_type.fillna(value=df.body_type.mode()[0]) # mode
    # remove outliers
    cols_w_outliers = [
        'back_legroom', 
        'city_fuel_economy', 
        'engine_displacement', 
        'front_legroom',
        'fuel_tank_volume',
        'height',
        'highway_fuel_economy',
        'horsepower',
        'length',
        'maximum_seating',
        'mileage',
        'price',
        'wheelbase',
        'width',
        'year']
    df = remove_outliers(df, cols_w_outliers, 2)
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
        
def remove_outliers(df, cols, k):
    '''
    Removes outliers that are outside of 1.5*IQR
    '''
    for col in cols:
        Q1 = np.percentile(df[col], 25, interpolation='midpoint')
        Q3 = np.percentile(df[col], 75, interpolation='midpoint')
        IQR = Q3 - Q1
        UB = Q3 + (k * IQR)
        LB = Q1 - (k * IQR)
        df = df[(df[col] <= UB) & (df[col] >= LB)]
    return df

def split_60(df):
    '''
    This function takes in a df and splits it into train, validate, and test dfs
    final proportions will be 60/20/20 for train/validate/test
    '''
    train_validate, test = train_test_split(df, test_size=0.2, random_state=527)
    train, validate = train_test_split(train_validate, test_size=.25, random_state=527)
    train_prop = train.shape[0] / df.shape[0]
    val_prop = validate.shape[0] / df.shape[0]
    test_prop = test.shape[0]/df.shape[0]
    print(f'Train Proportion: {train_prop:.2f} ({train.shape[0]} rows)\nValidate Proportion: {val_prop:.2f} ({validate.shape[0]} rows)\
    \nTest Proportion: {test_prop:.2f} ({test.shape[0]} rows)')
    return train, validate, test

def split_80(df):
    '''
    This function takes in a df and splits it into train, validate, and test dfs
    final proportions will be 80/10/10 for train/validate/test
    '''
    train_validate, test = train_test_split(df, test_size=0.10, random_state=527)
    train, validate = train_test_split(train_validate, test_size=.11, random_state=527)
    train_prop = train.shape[0] / df.shape[0]
    val_prop = validate.shape[0] / df.shape[0]
    test_prop = test.shape[0]/df.shape[0]
    print(f'Train Proportion: {train_prop:.2f} ({train.shape[0]} rows)\nValidate Proportion: {val_prop:.2f} ({validate.shape[0]} rows)\
    \nTest Proportion: {test_prop:.2f} ({test.shape[0]} rows)')
    return train, validate, test

def scale(train, validate, test, scaler, cols_to_scale):
    '''
    Returns dfs with indicated columns scaled using scaler passed and original columns dropped
    '''
    new_column_names = [col + '_scaled' for col in cols_to_scale]
    
    # Fit the scaler on the train
    scaler.fit(train[cols_to_scale])
    
    # transform train validate and test
    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[cols_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[cols_to_scale]), columns=new_column_names, index=validate.index),
    ], axis=1)
    
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[cols_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)
    
    # drop scaled columns
    train = train.drop(columns=cols_to_scale)
    validate = validate.drop(columns=cols_to_scale)
    test = test.drop(columns=cols_to_scale)
    
    return train, validate, test