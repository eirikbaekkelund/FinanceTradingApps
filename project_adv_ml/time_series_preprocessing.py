from darts import TimeSeries
import numpy as np
from darts.dataprocessing.transformers import Scaler
import pandas as pd
import random

def set_index(df, freq='Q'):
    """ 
    Sets the index of the df to a pd.date_range with quarterly frequency.
    Uses the time column as start_time and the number of rows as num_periods.
    This assumes frequency of the data is within supported frequencies of pd.date_range.
    Args:
        df (pd.DataFrame): DataFrame with a time column.
    
    Returns:
        df (pd.DataFrame): DataFrame with a pd.date_range as index with quarterly frequency.
    """
    assert 'time' in df.columns, 'time column must be in df.columns'
    
    start_time = df.time.min()
    num_periods = df.shape[0]

    # create pd.date_range with quarterly frequency based on start_time and num_periods
    df.index = pd.date_range(start=start_time, periods=num_periods, freq=freq)
    df = df.drop(columns=['time'])
    
    return df

def scale_series(series):
    """ 
    Scale the series with a Scaler from darts.
    """
    # TODO add option to choose scaler, add option to choose inverse scaler
    scaler = Scaler()
    scaler.fit_transform(series)
    series = scaler.transform(series)

    series = series.astype(np.float64)
    

    return series

def convert_df_to_series(df, covariates=None, stationary_covariates = ['ticker'], target='Sales_Actual_fiscal'):
    """ 
    Convert a df to a TimeSeries object from darts.
    The df must have a target column.

    Args:
        df (pd.DataFrame): DataFrame with a target column.
        covariates (list): List of covariates to be used in the model.
        stationary_covariates (list): List of stationary covariates to be used in the model.
        target (str): Column name of the target column. Default is 'Sales_Actual_fiscal'.
    Returns:
        covs (TimeSeries): TimeSeries object with the covariates.
        target (TimeSeries): TimeSeries object with the target column.
    """
    df = df.copy()
    
    if covariates is not None:
        assert all([col in df.columns for col in covariates]), 'covariates must be in df.columns'
        assert target not in covariates, 'target must not be in covariates'
        assert all([col not in covariates for col in stationary_covariates]), 'stationary_covariates must not be in covariates'

    
    elif covariates is None:
        # get the covariates that are not the target column or the stationary covariates
        covariates = [col for col in df.columns if col not in stationary_covariates and col != target and df[col].dtype in ['int64', 'float64']]
 

    covs = TimeSeries.from_dataframe(df[covariates], value_cols=covariates)
    target = TimeSeries.from_dataframe(df[[target]],value_cols=target, static_covariates=df[stationary_covariates].iloc[0])
    
    return covs, target

def dict_series(df, inv_mapper, covariates=None, stationary_covariates = ['ticker'], target='Sales_Actual_fiscal'):
    """ 
    Create a dictionary with TimeSeries objects for each ticker. 
    The dictionary contains a tuple with the covariates, stationary covariate and target.
    The order of the tuple is (covariates, stationary_covariate, target).

    Args:
        df (pd.DataFrame): DataFrame with a target column.
        inv_mapper (dict): Dictionary with the inverse mapping of the tickers that have been made numeric for stationary covariate usage.
        covariates (list): List of covariates to be used in the model.
        stationary_covariates (list): List of stationary covariates to be used in the model.
        target (str): Column name of the target column. Default is 'Sales_Actual_fiscal'.
    Returns:
        dict_tickers_series (dict): Dictionary with TimeSeries objects for each ticker.
        (covariates, target)
    """
    # creates data frames for each ticker with time as index
    dict_tickers_df =  {inv_mapper[tic] : set_index(df[df.ticker == tic]) for tic in df.ticker.unique()}
    
    return {tic : convert_df_to_series(dict_tickers_df[tic], covariates=covariates, stationary_covariates=stationary_covariates, target=target) for tic in dict_tickers_df.keys()}

def past_future_split(series_dict, n_preds=2):
    """
    Split the time series into past and future covariate sets. Make lists for the train and test sets.
    It makes lists for past covariates and future covariates.
    The series dict consist of the time series for each company. Its values are the tuples of (covariates, target).

    Args:
        series_dict (dict): dictionary of time series. The keys are the tickers and the values are the tuples of (covariates, target).
        n_preds (int): number of predictions to make. Default is 2.
    
    Returns:
        past_cov (list): list of past covariates.
        future_cov (list): list of future covariates.
        target (list): list of targets.
        tickers (list): list of tickers.
    They share the same indices with respect to the company.
    """
    result = [(value[0][:-n_preds], value[0][-n_preds:], value[1], key) for key, value in series_dict.items()]
    past_cov, future_cov, target, tickers = zip(*result)
    return past_cov, future_cov, target, tickers

def match_input_length(past_cov, future_cov, target, tickers, min_train = 10):
    """
    Match the input and output length of the train covariates and train target.
    All the components of train_cov should have the same length.
    The function removes the early values of components that are longer than the minimum length.
    The function removes the instances that have less than the minimum length from the covariates, target and the ticker
    as they all have matching indices with respect to company name.

    Args:
        past_cov (list): list of past covariates time series objects.
        future_cov (list): list of future covariates time series objects.
        target (list): list of target time series objects.
        tickers (list): list of tickers.
        min_train (int): minimum length of the train covariates. Default is 10.
    
    Returns:
        past_cov (list): list of past covariates with the same length.
        future_cov (list): list of future covariates.
        target (list): list of target.
        tickers (list): list of tickers.
    The indices of the lists are ordered with respect to the same company.
    """
    min_length_series = min([len(x) for x in past_cov])

    if min_length_series > min_train:
        min_train = min_length_series

    past_cov = [x[-min_train:] for x in past_cov]

    # Remove instances that have less than the minimum length from train and test sets
    mask = [len(x) >= min_train for x in past_cov]
    past_cov, future_cov, target, tickers = [[lst[i] for i in range(len(lst)) if mask[i]] for lst in [past_cov, future_cov, target, tickers]]
    
    combined_length = min_train + len(future_cov[0])
    target = [target[-combined_length:] for target in target]

    return past_cov, future_cov, target, tickers

def train_test_split(past_cov, future_cov, target, tickers, scaler_cov, scaler_target, test_size=0.2):
    """
    Splits the covariates and target into train and test sets.
    The split is performed by index to account for the fact that the covariates and target have the same indices with respect to the company.
    It randomly selects n_test elements from each list but at matching indices.

    Args:
        past_cov (list): list of past covariates
        future_cov (list): list of future covariates
        target (list): list of targets
        test_size (float): proportion of data to be used for testing
        tickers (list): list of tickers
        scaler_cov (list of darts.preprocessing.StandardScaler): list of scalers for the covariates
        scaler_target (list of darts.preprocessing.StandardScaler): list of scalers for the target
        test_size (float): proportion of data to be used for testing
    
    Returns:
        train_past_cov (list): list of past covariates for training
        test_past_cov (list): list of past covariates for testing
        train_future_cov (list): list of future covariates for training
        test_future_cov (list): list of future covariates for testing
        train_target (list): list of targets for training
        test_target (list): list of targets for testing
        tickers_train (list): list of tickers for training
        tickers_test (list): list of tickers for testing
    """
    n_test = int(len(past_cov) * test_size)

    test_indices = random.sample(range(len(past_cov)), n_test)
    train_indices = [i for i in range(len(past_cov)) if i not in test_indices]
    
    train_past_cov = [past_cov[i] for i in train_indices]
    test_past_cov = [past_cov[i] for i in test_indices]
    
    train_future_cov = [future_cov[i] for i in train_indices]
    test_future_cov = [future_cov[i] for i in test_indices]
    
    train_target = [target[i] for i in train_indices]
    test_target = [target[i] for i in test_indices]
    
    tickers_train = [tickers[i] for i in train_indices]
    tickers_test = [tickers[i] for i in test_indices]

    scaler_cov_train = [scaler_cov[i] for i in train_indices]
    scaler_cov_test = [scaler_cov[i] for i in test_indices]

    scaler_target_train = [scaler_target[i] for i in train_indices]
    scaler_target_test = [scaler_target[i] for i in test_indices]

    return train_past_cov, test_past_cov, train_future_cov, test_future_cov, train_target, test_target, tickers_train, tickers_test, scaler_cov_train, scaler_cov_test, scaler_target_train, scaler_target_test

def scale_series(past_cov, future_cov, target):
    """
    Scale the time series using the Scaler class from darts.
    Scalers are fit on the covariates and  target for each company.
    The scalers are applied on the past and future covariates and the target.
    
    It scales seperately the covariates and the target as well as seperately for each company.
    The inverse scalers are also returned to be used for the inverse transformation of the predictions.
    Thus, the results can be interpreted in the original scale.

    Args:
        past_cov (list): list of past covariates.
        future_cov (list): list of future covariates.
        target (list): list of target.

    Returns:
        past_cov (list): list of scaled past covariates.
        future_cov (list): list of scaled future covariates.
        target (list): list of scaled target.
        scaler_cov (list): list of scalers for the covariates.
        scaler_target (list): list of scalers for the target.
    """
    scaler_cov = [Scaler().fit(x) for x in past_cov]
    past_cov = [scaler.transform(x) for x, scaler in zip(past_cov, scaler_cov)]
    future_cov = [scaler.transform(x) for x, scaler in zip(future_cov, scaler_cov)]

    scaler_target = [Scaler().fit(x) for x in target]
    target = [scaler.transform(x) for x, scaler in zip(target, scaler_target)]

    return past_cov, future_cov, target, scaler_cov, scaler_target

def remove_n_predictions(test_target, test_future_cov):
    """
    Removes points that we try to predict from the target.

    Args:
        test_target (list): list of targets for testing.
        test_future_cov (list): list of future covariates for testing.
    
    Returns:
        test_target (list): list of targets for testing with the last n points removed.
    """
    n_remove = len(test_future_cov[0])
    test_target = [target[:-n_remove] for target in test_target]

    return test_target