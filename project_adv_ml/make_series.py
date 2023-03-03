from darts import TimeSeries
import numpy as np
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller


def set_df_index( df):
    """ 
    
    """
    df = df.copy()
    df = df.reset_index(drop=True)
    
    return df

def convert_df_to_series( df, covariates=['nw_total_sales_a_total','nw_total_sales_b_total','Sales_Estimate_fiscal'], target='Sales_Actual_fiscal'):
    """ 
    
    """
    df = df.copy()
    df = df.reset_index(drop=True)
    
    if covariates is None:
        covariates = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        
        try:
            covariates.remove(target)
            covariates.remove('mic')
        
        except ValueError:
            pass
    
    covs = TimeSeries.from_dataframe(df, value_cols=covariates, freq='Q')
    target = TimeSeries.from_dataframe(df,value_cols=target, freq='Q')
    
    return covs, target

def get_covs_target_dict( df, covariates=None, target='Sales_Actual_fiscal'):
    """ 
    
    """
    
    ticker_series =  {tic : set_df_index(df[df['ticker'] == tic]) for tic in df.ticker.unique()}
    
    return {tic : convert_df_to_series(ticker_series[tic], covariates=covariates, target=target) for tic in ticker_series.keys()}

def drop_short_sequences( series_dict, min_length = 12):
    """ 
    
    """
    new_dict = {}
    for key, vals in series_dict.items():

        length = vals[0].data_array().shape[0]
        if length < min_length:
            continue
        else:
            new_dict[key] = vals
    return new_dict

def split_covariates_target( covariate_series, target, proportion=0.9):
    """ 
    
    """
    n_split = int( len(target.data_array())*proportion )
    target_train, target_test = target[:n_split], target[n_split:]
    past_covariates, future_covariates = covariate_series[:n_split], covariate_series[n_split:]

    return past_covariates, future_covariates, target_train, target_test

def model_input( series_dict):
    """ 
    
    """
    covs_past, covs_future, targets_train, targets_test = [], [], [], []
    
    for tic in series_dict.keys():
        cov_p, cov_f, target_tr, target_te = split_covariates_target(series_dict[tic][0], series_dict[tic][1])
        covs_past.append(cov_p)
        covs_future.append(cov_f)
        targets_train.append(target_tr)
        targets_test.append(target_te)

    
    return covs_past, covs_future, targets_train, targets_test

def get_input_output_chunks( series, t_predict=1):
    min_length = np.inf

    for serie in series:
        length = len(serie.data_array())
        if length < min_length:
            min_length = length
    
    input_length = min_length - t_predict
    
    return input_length, t_predict

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
