import pandas as pd
import os
from darts.dataprocessing.transformers import (
    Scaler,
    MissingValuesFiller,
    Mapper,
    InvertibleMapper,
)
from darts import TimeSeries
import darts.metrics as metrics
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler

def create_df(file_name):
    """ 
    
    """
    return pd.read_table(os.path.abspath(file_name))
    
def split_column(df, delimiter, column):
    """ 
    
    """
    split_df = df[column].str.split(delimiter, expand=True)
    split_df.columns = column.split(delimiter)
    split_df = split_df.iloc[:,1:]
    
    return pd.concat([df.drop(column, axis=1), split_df], axis=1)

def convert_columns_to_numeric(df):
    """ 
    
    """
    for col in df.columns:
        try:
            pd.to_numeric(df[col], errors='raise')
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass

    return df

def train_target_slicing(df, target='', time_col='date'):
    """
    
    """
    
    assert target != '', "Target column must be specified"
    assert target in df.columns, "Target column must be in Data Frame columns"
    
    cols_train = list(df.columns)
    cols_train.remove(target)
    col_target = [time_col, target]
    
    X, y = df[cols_train], df[col_target]
    return  X, y

def train_test_split(df, proportion_train, col_target='', time_col='date'):
    """ 
    
    """
    df['date'] = pd.to_datetime(df['date'], format='%d %b %Y')
    df['date'].dt.to_period('D')
   
    X, y = train_target_slicing(df=df, target=col_target, time_col=time_col)
    n = X.shape[0]
    n_train = int( n * proportion_train )
    return X.iloc[:n_train,:], y.iloc[:n_train,:], X.iloc[n_train:,:], y.iloc[n_train:,:]

def set_date(df):
    """ 
    
    """
    df['date'] = pd.to_datetime(df['date'], format='%d %b %Y')
    # df['date'] =  df['date'].dt.to_period('D')
    return df

def create_darts_series_from_df(df, time_col='date', freq='D'):
    """  
    
    """
    return TimeSeries.from_dataframe(df=df, time_col=time_col, freq=freq)

def series_scale(series):
    scaler = Scaler()
    return scaler.fit_transform(series)

def series_rescale(series):
    scaler = Scaler()
    return scaler.inverse_transform(series)

def series_fill_missing_vals(series):
    filler = MissingValuesFiller()
    return filler.transform(series=series, method='quadratic')
    
def series_train_test(series, proportion=0.75):
    train, validation = series.split_before(proportion)
    return train, validation