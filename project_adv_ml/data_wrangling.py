import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def path_finder(name):
    """ 
    Finds the path to a file or folder in the project directory
    
    Args:
        name (str): Name of the file or folder.
    Returns:
        path (str): Path to the file or folder.
    """
    return os.path.abspath(name)

def files_from_folder(folder='exabel_data'):
    """ 
    Creates a list of files in a folder.

    Args:
        folder (str): Name of the folder. Default is 'exabel_data'.
    
    Returns:
        files (list): List of files in the folder.
    """
    folder_path = path_finder(folder)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    
    return files

def create_df(file_name, folder='exabel_data'):
    """ 
    Creates a Pandas data frame from a file in a folder.

    Args:
        file_name (str): Name of the file.
        folder (str): Name of the folder. Default is 'exabel_data'.
    
    Returns:
        df (pd.DataFrame): DataFrame with the data from the file.
    """

    assert file_name in files_from_folder(), 'File not found in folder'

    # Note - these columns are specific to the files in the exabel_data folder
    col_list_spendings = ['mic', 'ticker', 'time', 'nw_total_sales_a_total','nw_total_sales_b_total']
    col_list_revenue = ['mic', 'ticker', 'time', 'Sales_Actual_fiscal','Sales_Estimate_fiscal']
    
    path = str(path_finder(folder) + '/' + file_name)
    # Note - this is specific to the files in the exabel_data folder    
    df =  pd.read_excel(path_finder(path))

    if file_name == 'revenue.xlsx':
        return df[col_list_revenue]
    
    elif file_name == 'spend_amounts_aggregated.xlsx':
        return df[col_list_spendings]
    
    else:
        raise ValueError('File name not recognized')
    

def split_column( df, delimiter, column):
    """ 
    Splits a column in a df based on a delimiter and returns a new df with the split columns.
    Specifically for csv files with multiple columns in one column.

    Args:
        df (pd.DataFrame): DataFrame with the data from the file.
        delimiter (str): Delimiter used to split the column.
    
    Returns:
        df (pd.DataFrame): DataFrame with the data from the file.
    """
    split_df = df[column].str.split(delimiter, expand=True)
    split_df.columns = column.split(delimiter)
    split_df = split_df.iloc[:,1:]
    
    return pd.concat([df.drop(column, axis=1), split_df], axis=1)

def add_time_cols(df):
    """ 
    Add year, month and quarter columns to a df based on the time column.

    Args:
        df (pd.DataFrame): DataFrame with the data from the file.
    Returns:
        df_copy (pd.DataFrame): DataFrame with the data from the file.
    """
    df_copy = df.copy()
    df_copy['year'] = df_copy['time'].dt.year
    df_copy['month'] = df_copy['time'].dt.month
    df_copy['quarter'] = df_copy['time'].dt.quarter

    return df_copy

def encode_index( df, column='mic', mapper = None):
    """  
    
    """

    if mapper is not None:
        df[column] = df[column].map(encoding)
    else:
        unique_names = df[column].unique()
        encoding = {name: i for i, name in enumerate(unique_names)}
        df[column] = df[column].map(encoding)
    
    return df

def merge_dataframes(df_left, df_right, on=['ticker', 'time'], how='left', cols_right=['ticker', 'time', 'Sales_Actual_fiscal', 'Sales_Estimate_fiscal']):
    """ 
    
    """

    return df_left.merge(df_right[cols_right], on=on, how=how)

# ORIGINAL
def print_nans_companies( df):
    """ 

    """
    for tic in np.unique(df.ticker):
        df_copy = df[df['ticker'] == tic]
        
        if df_copy.isnull().values.any():
            print('\n')
            print(f"Ticker: {tic}, # Data points: {df_copy.shape[0]}")
        df_copy = df_copy.reset_index(drop=False)
        
        for col in df_copy.columns:
            nan_count = df_copy[col].isnull().sum()
            if nan_count > 0:
                nan_indices = df_copy[df_copy[col].isnull()].index.tolist()
                print(f"Column: {col}, NaN Indices: {nan_indices}")

def get_nan_columns(df):
    """
    Return a dictionary of columns with NaN values grouped by ticker.

    Args:
        df (pandas.DataFrame): DataFrame to search for NaN values.

    Returns:
        A dictionary with tickers as keys and lists of columns with NaN values as values.
    """
    nan_dict = {}
    nan_groups = df.groupby('ticker').apply(lambda x: x.isnull().any())
    for ticker, nan_cols in nan_groups.items():
        if nan_cols.any():
            nan_dict[ticker] = [col for col in nan_cols.index if nan_cols[col]]
    return nan_dict

def remove_missing_ground_truth(df, thresh_proportion=0.4):
    """
    Remove rows for companies with a high proportion of missing ground truth data.

    Args:
        df (pandas.DataFrame): DataFrame to search for missing ground truth data.
        thresh_proportion (float, optional): Proportion of missing ground truth data required to remove a company. Defaults to 0.4.

    Returns:
        A new DataFrame with rows removed for companies with a high proportion of missing ground truth data.
    """
    nan_companies = get_nan_columns(df)
    for tic, nan_cols in nan_companies.items():
        if 'Sales_Actual_fiscal' in nan_cols and 'Sales_Estimate_fiscal' in nan_cols:
            tic_df = df[df['ticker'] == tic]
            prop_actual = tic_df['Sales_Actual_fiscal'].isna().mean()
            prop_estimate = tic_df['Sales_Estimate_fiscal'].isna().mean()
            if prop_actual >= thresh_proportion and prop_estimate >= thresh_proportion:
                df = df[df['ticker'] != tic]
    return df

def impute_least_squares(df, company, column):
    """ 
    Imputes missing values in a column of a dataframe by OLS.

    Args:
        df (pd.DataFrame): dataframe of the entire dataset
        company (str): company ticker
        column (str): column for which the missing values are imputed
    
    Returns:
        df (pd.DataFrame): dataframe with imputed values
    """
    # get the subset of the dataframe which contains the company rows
    df_company = df[df['ticker'] == company]
    # get the index of the nan values
    nan_index = df_company[df_company[column].isna()].index
    # get the values of the other columns
    non_nan_index = df_company[df_company[column].notna()].index
    # create the X and y from the non nan values
    X = df_company.loc[non_nan_index, ['nw_total_sales_a_total', 'Sales_Estimate_fiscal', 'Sales_Actual_fiscal']]
    y = df_company.loc[non_nan_index, column]
    # add bias term to X
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    # get the coefficients
    weights = np.linalg.lstsq(X, y, rcond=None)[0]
    # predict the values for the nan values
    X_nan = df_company.loc[nan_index, ['nw_total_sales_a_total', 'Sales_Estimate_fiscal', 'Sales_Actual_fiscal']]
    X_nan = np.concatenate((np.ones((X_nan.shape[0], 1)), X_nan), axis=1)
    y_pred = np.dot(X_nan, weights)
    # replace the nan values with the predicted values
    df.loc[nan_index, column] = y_pred

    return df

def impute_normal_sample(df, company, column):
    """ 
    Imputes missing values in a column of a dataframe by sampling from a normal distribution.

    Args:
        df (pd.DataFrame): dataframe of the entire dataset
        company (str): company ticker
        column (str): column for which the missing values are imputed
    
    Returns:
        df (pd.DataFrame): dataframe with imputed values
    """
    assert column in ['Sales_Actual_fiscal', 'Sales_Estimate_fiscal']

    if column == 'Sales_Actual_fiscal':
        column_estimate = 'Sales_Estimate_fiscal'
    else:
        column_estimate = 'Sales_Actual_fiscal'
    
    # get the subset of the dataframe which contains the company rows
    df_company = df[df['ticker'] == company]
    # get the index of the nan values
    nan_index = df_company[df_company[column].isna()].index
    # get the values of the other columns
    non_nan_index = df_company[df_company[column].notna()].index
    
    # get the y estimate values at the nan index
    y_estimate = df_company.loc[nan_index, column_estimate]
    print(y_estimate, nan_index)
    # get the standard deviation as the difference between the estimate and the actual at the non nan index
    std = np.std(df_company.loc[non_nan_index, column] - df_company.loc[non_nan_index, column_estimate])
    y_pred = np.random.multivariate_normal(y_estimate, std * np.eye(len(y_estimate)) )
    # replace the nan values with the predicted values
    df.loc[nan_index, column] = y_pred

    return df

def impute_values(df):
    # get the list of companies with nan values
    nan_companies = dw.get_nan_columns(df)
    company_list_panel_B = nan_companies['nw_total_sales_b_total']
    company_list_panel_A = nan_companies['nw_total_sales_a_total']

    company_list_sales_actual = nan_companies['Sales_Actual_fiscal']
    company_list_sales_estimated = nan_companies['Sales_Estimate_fiscal']

    company_list = company_list_panel_B + company_list_panel_A + company_list_sales_actual + company_list_sales_estimated
    company_list = list(set(company_list))


    for company in company_list:
        # check if is in set b and not in actual, estimated, and a
        if company in company_list_panel_B and company not in company_list_sales_actual and company not in company_list_sales_estimated and company not in company_list_panel_A:
            df = impute_least_squares(df, company, 'nw_total_sales_b_total')
        # check if is in set a and not in set b, actual and estimated
        elif company in company_list_panel_A and company not in company_list_panel_B and company not in company_list_sales_actual and company not in company_list_sales_estimated:
            df = impute_least_squares(df, company, 'nw_total_sales_a_total')
        
        # check if is in actual and not in estimated
        elif company in company_list_sales_actual and company not in company_list_sales_estimated:
            df = impute_normal_sample(df, company, 'Sales_Actual_fiscal')
        
        # check if is in estimated and not in actual
        elif company in company_list_sales_estimated and company not in company_list_sales_actual:
            df = impute_normal_sample(df, company, 'Sales_Estimate_fiscal')
    
    return df

def run_imputation(df):
    """ 
    Iteratively imputes missing values in a dataframe.

    Args:
        df (pd.DataFrame): dataframe of the entire dataset
    
    Returns:
        df (pd.DataFrame): dataframe with imputed values
    """
    while True:
        n_nans = df.isna().sum().sum()
        df = impute_values(df)
        if n_nans == df.isna().sum().sum():
            return df
        
def remove_short_series(df, n=9):
    """ 
    
    """
    ticker_counts = df['ticker'].value_counts()
    df = df[df['ticker'].isin(ticker_counts[ticker_counts >= n].index)]
    
    return df

def add_prod( df):
    """ 
    
    """
    df['prod_sales'] = df['Sales_Actual_fiscal'] * df['Sales_Estimate_fiscal']
    df['prod_n_customers'] = df['nw_total_sales_a_total'] * df['nw_total_sales_b_total']
    
    return df

def add_proportion_ab( df):
    df['proportion_ab'] = df['Sales_Actual_fiscal'] / ( df['nw_total_sales_a_total'] +  df['nw_total_sales_b_total'] )
    
    return df


def add_quarterly_yoy( df):
    df['quarterly_yoy'] = df.groupby(['ticker'])['Sales_Actual_fiscal'].pct_change(periods=4)
    # replace first 4 quarters with zeros as there is no yoy
    df['quarterly_yoy'] = df['quarterly_yoy'].fillna(0)
    
    return df

def drop_low_correlation_features( df):
    """ 
    Drop features with low correlation with Sales_Actual_fiscal
    or nans in the correlation
    """
    corr = df.groupby('ticker').apply(lambda x: x.corrwith(x['Sales_Actual_fiscal'], numeric_only=True)).mean()
    # to ensure we do not drop ticker from dataframe
    corr['ticker'] = 1
    corr['time'] = 1
    df = df[corr.keys().tolist()]
    df = df.drop(columns=corr[abs(corr) < 0.1].index, axis=1)
    # manually drop mic 
    df = df.drop(columns=['mic'], axis=1)
    
    return df

def get_numeric_cols(df):
    return df[[col for col in df.columns if df[col].dtype in ['int64', 'float64']]]

def encode_to_int(df, col):
    """ 
    Map ticker to a number
    """
    mapper = {tic: i + 1 for i, tic in enumerate(df[col].unique())}
    inv_mapper = {v: k for k, v in mapper.items()}
    df[col] = df[col].map(mapper)    

    return df, mapper, inv_mapper

def encode_one_hot(df, col):
    """ 
    One hot encode the dataframe categorical column
    """
    df = df.copy()
    df = pd.get_dummies(df, columns=[col])
    
    return df

def encode_float(df, col):
    """ 
    Encode the dataframe column to a float
    """
    unique_vals = df[col].unique()
    mapper = {val: i for i, val in zip(np.linspace(0.1,1, len(unique_vals)), unique_vals)}
    inv_mapper =  {v: k for k, v in mapper.items()}
    df[col] = df[col].map(mapper)

    return df, inv_mapper

def rmse_residual_vector(predictions, target):
    """
    Calculate elementwise difference between predictions and target in a flattened array.

    Args: 
        predictions (TimeSeries): TimeSeries of predictions where each entry is a vector of length n_preds.
        target (TimeSeries): TimeSeries of target values where the last n_preds entries are the target values.
    """
    return [ np.sqrt( (( target[-1:].data_array().squeeze() - pred[-1:].data_array().squeeze() )**2).to_numpy() ) for target, pred in zip(target, predictions) ]

def rmse_df(preds_train_nbeats, preds_test_nbeats, preds_val_nbeats, 
            preds_train_rf, preds_test_rf, preds_val_rf, 
            preds_train_xgb, preds_test_xgb, preds_val_xgb, 
            train_target, test_target, val_target):
    """
    Calculate RMSE for each model and each dataset.

    Args:
        preds_train_nbeats (TimeSeries): TimeSeries of predictions for the train set for the NBEATS model.
        preds_test_nbeats (TimeSeries): TimeSeries of predictions for the test set for the NBEATS model.
        preds_val_nbeats (TimeSeries): TimeSeries of predictions for the validation set for the NBEATS model.
        preds_train_rf (TimeSeries): TimeSeries of predictions for the train set for the Random Forest model.
        preds_test_rf (TimeSeries): TimeSeries of predictions for the test set for the Random Forest model.
        preds_val_rf (TimeSeries): TimeSeries of predictions for the validation set for the Random Forest model.
        preds_train_xgb (TimeSeries): TimeSeries of predictions for the train set for the XGBoost model.
        preds_test_xgb (TimeSeries): TimeSeries of predictions for the test set for the XGBoost model.
        preds_val_xgb (TimeSeries): TimeSeries of predictions for the validation set for the XGBoost model.
        train_target (TimeSeries): TimeSeries of target values for the train set.
        test_target (TimeSeries): TimeSeries of target values for the test set.
        val_target (TimeSeries): TimeSeries of target values for the validation set.
    
    Returns:
        df_res (pd.DataFrame): DataFrame with RMSE for each model and each dataset.
    """
    def calculate_rmse(predictions, target):
        return rmse_residual_vector(predictions=predictions, target=target)

    models = {
        'nbeats': {
            'train': preds_train_nbeats,
            'test': preds_test_nbeats,
            'val': preds_val_nbeats,
        },
        'rf': {
            'train': preds_train_rf,
            'test': preds_test_rf,
            'val': preds_val_rf,
        },
        'xgb': {
            'train': preds_train_xgb,
            'test': preds_test_xgb,
            'val': preds_val_xgb,
        },
    }

    df_res = pd.DataFrame()

    for model_name, model_data in models.items():
        rmse_train = calculate_rmse(model_data['train'], train_target)
        rmse_test = calculate_rmse(model_data['test'], test_target)
        rmse_val = calculate_rmse(model_data['val'], val_target)
        rmse_total = rmse_train + rmse_test + rmse_val
        
        model_results = {
            f'res_{model_name}_train': rmse_train,
            f'res_{model_name}_test': rmse_test,
            f'res_{model_name}_val': rmse_val,
            f'res_{model_name}': rmse_total,
        }
        
        df_res = df_res.append(model_results, ignore_index=True)
        
    df_res = df_res[['res_nbeats_train', 'res_nbeats_test', 'res_nbeats_val', 'res_nbeats',
                    'res_rf_train', 'res_rf_test', 'res_rf_val', 'res_rf',
                    'res_xgb_train', 'res_xgb_test', 'res_xgb_val', 'res_xgb']]
    
    return df_res