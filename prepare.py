####### PREPARE CAR DATA #######

# standard imports
import pandas as pd
import numpy as np

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
    df.franchise_dealer = np.where(df.franchise_dealer == True, 'Yes', 'No') # change from bool to Yes or No
    df.front_legroom = df.front_legroom.str.split(' ', expand=True)[0] # split and only keep number
    df.front_legroom = pd.to_numeric(df.front_legroom, errors='coerce') # convert to float
    df.fuel_tank_volume = df.fuel_tank_volume.str.split(' ', expand=True)[0] # split and only keep number
    df.fuel_tank_volume = pd.to_numeric(df.fuel_tank_volume, errors='coerce') # convert to float
    df.height = df.height.str.split(' ', expand=True)[0] # split and only keep number
    df.height = pd.to_numeric(df.height, errors='coerce') # convert to float
    df.length = df.length.str.split(' ', expand=True)[0] # split and only keep number
    df.length = pd.to_numeric(df.length, errors='coerce') # convert to float
    top_six = ['Ford', 'Chevrolet', 'Toyota', 'Honda', 'Nissan', 'Jeep'] # biggest proportion, top 6
    df.make_name = df.make_name.apply(lambda x: x if x in top_six else 'Other') # group all others as 'Other'
    df.maximum_seating = df.maximum_seating.str.split(' ', expand=True)[0] # split and only keep number
    df.maximum_seating = pd.to_numeric(df.maximum_seating, errors='coerce') # convert to float
    df.wheelbase = df.wheelbase.str.split(' ', expand=True)[0] # split and only keep number
    df.wheelbase = pd.to_numeric(df.wheelbase, errors='coerce') # convert to float
    df.width = df.width.str.split(' ', expand=True)[0] # split and only keep number
    df.width = pd.to_numeric(df.width, errors='coerce') # convert to float
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
    
    return df

def impute(train, validate, test):
    '''
    Gets imputation values from train and then imputes them for train, validate, and test
    '''
    
    # make copies of passed splits so originals aren't modified
    train = train.copy()
    validate = validate.copy()
    test = test.copy()
    
    # get imputation values
    back_legroom_impute = round(train.back_legroom.mean(),2)
    front_legroom_impute = round(train.front_legroom.mean(),2)
    fuel_tank_volume_impute = round(train.fuel_tank_volume.mean(),2)
    height_impute = round(train.height.mean(),2)
    length_impute = round(train.length.mean(),2)
    maximum_seating_impute = train.maximum_seating.median()
    wheelbase_impute = round(train.wheelbase.mean(),2)
    width_impute = round(train.width.mean(),2)
    city_fuel_economy_impute = round(train.city_fuel_economy.mean(),0)
    engine_displacement_impute = round(train.engine_displacement.mean(),0)
    highway_fuel_economy_impute = round(train.highway_fuel_economy.mean(),0)
    horsepower_impute = round(train.horsepower.mean(),0)
    wheel_system_impute = train.wheel_system.mode()[0]
    mileage_impute = round(train.mileage.mean(),0)
    fuel_type_impute = train.fuel_type.mode()[0]
    transmission_impute = train.transmission.mode()[0]
    body_type_impute = train.body_type.mode()[0]
    
    # train
    train.back_legroom = train.back_legroom.fillna(value=back_legroom_impute) # fill missing values with mean
    train.front_legroom = train.front_legroom.fillna(front_legroom_impute) # fill missing values with mean
    train.fuel_tank_volume = train.fuel_tank_volume.fillna(fuel_tank_volume_impute) # fill missing values with mean
    train.height = train.height.fillna(height_impute) # fill missing values with mean
    train.length = train.length.fillna(length_impute) # fill missing values with mean
    train.maximum_seating = train.maximum_seating.fillna(maximum_seating_impute) # fill missing values with median
    train.wheelbase = train.wheelbase.fillna(wheelbase_impute) # fill missing values with mean
    train.width = train.width.fillna(width_impute) # convert to float
    # impute remaining missing values
    train.city_fuel_economy = train.city_fuel_economy.fillna(city_fuel_economy_impute) # mean
    train.engine_displacement = train.engine_displacement.fillna(engine_displacement_impute) # mean
    train.highway_fuel_economy = train.highway_fuel_economy.fillna(highway_fuel_economy_impute) # mean
    train.horsepower = train.horsepower.fillna(horsepower_impute) # mean
    train.wheel_system = train.wheel_system.fillna(wheel_system_impute) # mode
    train.mileage = train.mileage.fillna(mileage_impute) # mean
    train.fuel_type = train.fuel_type.fillna(fuel_type_impute) # mode
    train.transmission = train.transmission.fillna(transmission_impute) # mode
    train.body_type = train.body_type.fillna(body_type_impute) # mode
    
    # validate
    validate.back_legroom = validate.back_legroom.fillna(value=back_legroom_impute) # fill missing values with mean
    validate.front_legroom = validate.front_legroom.fillna(front_legroom_impute) # fill missing values with mean
    validate.fuel_tank_volume = validate.fuel_tank_volume.fillna(fuel_tank_volume_impute) # fill missing values with mean
    validate.height = validate.height.fillna(height_impute) # fill missing values with mean
    validate.length = validate.length.fillna(length_impute) # fill missing values with mean
    validate.maximum_seating = validate.maximum_seating.fillna(maximum_seating_impute) # fill missing values with median
    validate.wheelbase = validate.wheelbase.fillna(wheelbase_impute) # fill missing values with mean
    validate.width = validate.width.fillna(width_impute) # convert to float
    # impute remaining missing values
    validate.city_fuel_economy = validate.city_fuel_economy.fillna(city_fuel_economy_impute) # mean
    validate.engine_displacement = validate.engine_displacement.fillna(engine_displacement_impute) # mean
    validate.highway_fuel_economy = validate.highway_fuel_economy.fillna(highway_fuel_economy_impute) # mean
    validate.horsepower = validate.horsepower.fillna(horsepower_impute) # mean
    validate.wheel_system = validate.wheel_system.fillna(wheel_system_impute) # mode
    validate.mileage = validate.mileage.fillna(mileage_impute) # mean
    validate.fuel_type = validate.fuel_type.fillna(fuel_type_impute) # mode
    validate.transmission = validate.transmission.fillna(transmission_impute) # mode
    validate.body_type = validate.body_type.fillna(body_type_impute) # mode
    
    # test
    test.back_legroom = test.back_legroom.fillna(value=back_legroom_impute) # fill missing values with mean
    test.front_legroom = test.front_legroom.fillna(front_legroom_impute) # fill missing values with mean
    test.fuel_tank_volume = test.fuel_tank_volume.fillna(fuel_tank_volume_impute) # fill missing values with mean
    test.height = test.height.fillna(height_impute) # fill missing values with mean
    test.length = test.length.fillna(length_impute) # fill missing values with mean
    test.maximum_seating = test.maximum_seating.fillna(maximum_seating_impute) # fill missing values with median
    test.wheelbase = test.wheelbase.fillna(wheelbase_impute) # fill missing values with mean
    test.width = test.width.fillna(width_impute) # convert to float
    # impute remaining missing values
    test.city_fuel_economy = test.city_fuel_economy.fillna(city_fuel_economy_impute) # mean
    test.engine_displacement = test.engine_displacement.fillna(engine_displacement_impute) # mean
    test.highway_fuel_economy = test.highway_fuel_economy.fillna(highway_fuel_economy_impute) # mean
    test.horsepower = test.horsepower.fillna(horsepower_impute) # mean
    test.wheel_system = test.wheel_system.fillna(wheel_system_impute) # mode
    test.mileage = test.mileage.fillna(mileage_impute) # mean
    test.fuel_type = test.fuel_type.fillna(fuel_type_impute) # mode
    test.transmission = test.transmission.fillna(transmission_impute) # mode
    test.body_type = test.body_type.fillna(body_type_impute) # mode
    
    return train, validate, test

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
        
def remove_outliers(train, validate, test, cols, k):
    '''
    Removes outliers that are outside of k*IQR using train to get bounds and then applying to all splits
    '''
     # make copies of passed splits so originals aren't modified
    train = train.copy()
    validate = validate.copy()
    test = test.copy()
    
    # remove outliers
    for col in cols:
        # get bounds from train
        Q1 = np.percentile(train[col], 25, interpolation='midpoint')
        Q3 = np.percentile(train[col], 75, interpolation='midpoint')
        IQR = Q3 - Q1
        UB = Q3 + (k * IQR)
        LB = Q1 - (k * IQR)
        # apply bounds to train to eliminate outliers
        train = train[(train[col] <= UB) & (train[col] >= LB)]
        # apply bounds to validate to eliminate outliers
        validate = validate[(validate[col] <= UB) & (validate[col] >= LB)]
        # apply bounds to test to eliminate outliers
        test = test[(test[col] <= UB) & (test[col] >= LB)]
        
    return train, validate, test

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