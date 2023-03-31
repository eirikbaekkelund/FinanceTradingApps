import pandas as pd
import yfinance as yf
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os

def get_data(file_name='effr.xlsx'):
    """ 
    Create a dataframe of S&P500 ETF & the Effective Federal Funds Rate (EFFR)
    from 1 Jan 2014 to 31 December 2019

    Args:
        file_name (str): file name of the excel file containing the effective federal funds rate

    Returns:
        df_rf (pd.DataFrame): dataframe of effective federal funds rate
    """
    # get absolute path of the file
    file_name = os.path.join(os.path.dirname(__file__), file_name)

    df_rf = pd.read_excel(file_name)
    # drop cols with missing values
    df_rf = df_rf.dropna(axis=1)
    df_rf = df_rf.drop('Rate Type', axis=1)
    # set effective date to be the index
    df_rf = df_rf.set_index('Effective Date')
    df_rf.index = pd.to_datetime(df_rf.index)
    df_rf.index = df_rf.index.strftime('%Y-%m-%d')

    df_rf = df_rf.iloc[::-1]
        
    # convert percentage to decimal and annual rate to daily rate
    df_rf = df_rf / (100 * 252 )
 

    # rename columns
    df_rf.columns = [col[:-4] for col in df_rf.columns]
    # drop columns with missing values
    df_rf = df_rf.drop(['Target Rate From', 'Target Rate To'], axis=1)
    # rename rate column
    df_rf = df_rf.rename(columns={'Rate': 'EFFR'})
   

    df_sp500 = yf.download('SPY', start='2014-01-01', end='2019-12-31', progress=False)
    df_sp500 = df_sp500.drop(['Open', 'High', 'Low', 'Close'], axis=1)
    df_sp500 = df_sp500.rename(columns={'Adj Close': 'SPY'})

    df_sp500.index = pd.to_datetime(df_sp500.index)
    df_sp500.index = df_sp500.index.strftime('%Y-%m-%d')

    # merge the two dataframes
    df = df_sp500.merge(df_rf, how='left', left_index=True, right_index=True)
    df = df.dropna(axis=0)
    df['Date'] = df.index
    df['Time'] = np.arange(len(df))
    df['Price Change'] = df['SPY'].pct_change()
    # reset index to be 0 to len(df)
    df = df.reset_index(drop=True)
    
    return df

def excess_return_unit(df):
    """ 
    Calculate the excess return of S&P500 ETF 
    as compared to the risk-free rate.

    Args:
        df_sp500 (pd.DataFrame): dataframe of S&P500 ETF
        df_rf (pd.DataFrame): dataframe of effective federal funds rate
    Returns:
        df (pd.DataFrame): dataframe including excess return
    """
    # add excess return
    df['Excess Return'] = ( np.diff( df['SPY'] ) / df['SPY'][1:] ) - df['EFFR'][1:]
    # fill 0 for the first row that has no return value
    df = df.fillna(0)
    return df

def moving_averages(df, col='SPY'):
    """ 
    Calculate the 10-day, 20-day, 30-day moving averages of S&P500 ETF.

    Args:
        df (pd.DataFrame): dataframe including excess return
    Returns:
        df (pd.DataFrame): dataframe including excess return and moving averages
    """
    df['10 MA' + f' {col}'] = df[col].rolling(window=10).mean()
    df['20 MA' + f' {col}'] = df[col].rolling(window=20).mean()
    df['30 MA' + f' {col}'] = df[col].rolling(window=30).mean()
    
    return df

def exponential_moving_averages(df, col='SPY'):
    """ 
    Calculate the 10-day, 20-day, 30-day exponential moving averages of S&P500 ETF.

    Args:
        df (pd.DataFrame): dataframe including excess return
    Returns:
        df (pd.DataFrame): dataframe including excess return and moving averages
    """
    df['10 EMA' + f' {col}'] = df[col].ewm(span=10, adjust=False).mean()
    df['20 EMA' + f' {col}'] = df[col].ewm(span=20, adjust=False).mean()
    df['30 EMA' + f' {col}'] = df[col].ewm(span=30, adjust=False).mean()
    return df

def bollinger_bands(df, col='SPY', window=20, sigma=2):
    
    """
    Calculate the Bollinger Bands of the Gaussian Process model.

    Args:
        df (pd.DataFrame): dataframe including excess return
        col (str): column name of the data for which to calculate the Bollinger Bands
        window (int): window size for the moving average
        sigma (int): number of standard deviations to use for the upper and lower bands
    Returns:
        df (pd.DataFrame): dataframe including bolinger bands
    """
    ma = df[col].rolling(window=window).mean()
    std = df[col].rolling(window=window).std()
    df['BB Upper'] = ma + (std * sigma)
    df['BB Lower'] = ma - (std * sigma)

    return df

def kelly_fraction(df, col='Excess Return'):
    """
    Computes the mean and variance of the column for all values up to the index
    for the entire dataframe.

    Args:
        df (pd.DataFrame): dataframe with data
    
    Returns:
        df (pd.DataFrame): dataframe with means and variances
    """
    # compute mean from GP predictions based on all values up to the index
    mean = df[f'{col}'].expanding().mean()
    std = df[f'{col}'].expanding().std()

    df['Kelly Fraction'] =  ( df[col] - mean ) / std

    return df

