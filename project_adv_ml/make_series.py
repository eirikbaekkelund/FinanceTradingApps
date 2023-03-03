from darts import TimeSeries
import numpy as np
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller


def set_df_index(df):
    """ 
    
    """
    df = df.copy()
    df = df.reset_index(drop=True)

    return df

def scale_series(series):
    """ 
    
    """
    scaler = Scaler()
    scaler.fit_transform(series)
    series = scaler.transform(series)

    series = series.astype(np.float64)

    return series


def convert_df_to_series(df,covariates=['nw_total_sales_a_total','nw_total_sales_b_total','Sales_Estimate_fiscal'], target='Sales_Actual_fiscal'):
    """ 
    
    """
    df = df.copy()
    df = df.reset_index(drop=True)
    
    if covariates is None:
        covariates = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        
        try:
            covariates.remove(target)        
        except ValueError:
            pass
    
    covs = TimeSeries.from_dataframe(df, value_cols=covariates, freq='Q')
    target = TimeSeries.from_dataframe(df,value_cols=target, freq='Q')
    
    covs = scale_series(covs)
    target = scale_series(target)
    
    return covs, target

def get_covs_target_dict(df, covariates=None, target='Sales_Actual_fiscal'):
    """ 
    
    """
    
    ticker_series =  {tic : set_df_index(df[df['ticker'] == tic]) for tic in df.ticker.unique()}
    
    return {tic : convert_df_to_series(ticker_series[tic], covariates=covariates, target=target) for tic in ticker_series.keys()}

def drop_short_sequences(series_dict, min_length = 15):
    """ 
    
    """
    return {key : vals for key, vals in series_dict.items() if vals[0].data_array().shape[0] > min_length}
    
def slice_series(covariates, target, proportion=0.9):
    """ 
    
    """
    n_split = int( len(target.data_array())*proportion )
    
    target_train, target_test = target[:n_split], target[n_split:]
    past_covariates, future_covariates = covariates[:n_split], covariates[n_split:]

    return past_covariates, future_covariates, target_train, target_test

def split_covariates_target(covariates, target, proportion=0.9):
    """ 
    
    """
    past_covariates, future_covariates = covariates.split_before(proportion)
    target_train, target_test = target.split_before(proportion)
    
    return past_covariates, future_covariates, target_train, target_test

def model_input(series_dict, train_style='slice'):
    """ 
    
    """
    assert train_style in ['slice', 'split'], "need to have train_style set to either slice or split"

    if train_style == 'slice':
        covs_past, covs_future, targets_train, targets_test = zip(*[slice_series(series_dict[tic][0], series_dict[tic][1]) for tic in series_dict.keys()])
        covs_past = list(covs_past)
        covs_future = list(covs_future)
        targets_train = list(targets_train)
        targets_test = list(targets_test)
        tickers = list(series_dict.keys())
    else:
        covs_past, covs_future, targets_train, targets_test = split_covariates_target(series_dict[tic][0], series_dict[tic][1])
        tickers = list(series_dict.keys())  
    
    return covs_past, covs_future, targets_train, targets_test, tickers

def get_input_output_chunks(series_train, series_test):
    
    min_length_train, min_length_test = np.inf, np.inf

    for train, test in zip(series_train, series_test):
        length_train = len(train.data_array())
        length_test = len(test.data_array())
        
        if length_train < min_length_train:
            min_length_train = length_train
        
        if length_test < min_length_test:
            min_length_test = length_test
        
    min_length_train = min_length_train - min_length_test
    
    return min_length_train, min_length_test

def set_test_length(series_test, t_test):
    return [serie[:t_test] for serie in series_test]

def series_scale( series):
    """ 
    
    """
    scaler = Scaler()
    return scaler.fit_transform(series)

def series_rescale( series):
    """ 
    
    """
    scaler = Scaler()
    return scaler.inverse_transform(series)

def series_fill_missing_vals( series):
    """ 
    
    """
    filler = MissingValuesFiller()
    return filler.transform(series=series, method='quadratic')
