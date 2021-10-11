import pandas as pd

##################### Acquire Car Data #####################
    
def get_tx_car_data():
    '''
    This function reads the data from the a local .csv file into a df and returns the df.
    '''
    df = pd.read_csv('texas_car_data.csv', index_col=0)
    return df

def get_car_data():
    '''
    This function reads in raw data from .csv, filters to only TX cities, saves new df to local .csv, returns filtered df
    '''
    df = pd.read_csv('used_cars_data.csv', index_col=0) # get all data from local .csv
    # filter to just data from 7 largest cities in TX
    df = df[(df.city == 'San Antonio') | 
            (df.city == 'Houston') |
            (df.city == 'Dallas') |
            (df.city == 'Austin') |
            (df.city == 'El Paso') |
            (df.city == 'Arlington') |
            (df.city == 'Fort Worth')]
    df.to_csv('texas_car_data.csv') # save the .csv to use for rest of project
    return df