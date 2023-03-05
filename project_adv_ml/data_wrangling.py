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

def merge_spendings_revenue( df_spendings, df_revenue):
    """ 
    
    """
    return df_spendings.merge(
                df_revenue[['ticker', 'time', 'Sales_Actual_fiscal', 'Sales_Estimate_fiscal']], 
                on=['ticker', 'time'], how='left'
                )

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

def get_nan_columns( df):
    """ 
    
    """
    nans_dict = {}
    # TODO make more efficient?
    for tic in np.unique(df.ticker):
        
        df_copy = df[df['ticker'] == tic]
        df_copy = df_copy.reset_index(drop=False)
        nans_dict[tic] = []
        
        for col in df_copy.columns:
            nan_count = df_copy[col].isnull().sum()
            if nan_count > 0:
                nans_dict[tic].append(col)
    
    return nans_dict

def remove_missing_ground_truth( df, tresh_proportion=0.4):
    """ 
    
    """
    nan_companies = get_nan_columns(df)
    # TODO make more efficient?
    
    for tic in nan_companies.keys():
    
        if 'Sales_Actual_fiscal' in nan_companies[tic] and 'Sales_Actual_fiscal' in nan_companies[tic]:
    
            df_copy = df[df['ticker'] == tic]
    
            proportion_actual = df_copy[df_copy['Sales_Actual_fiscal'].isna()].shape[0] / df_copy.shape[0]
    
            proportion_estimate = df_copy[df_copy['Sales_Estimate_fiscal'].isna()].shape[0] / df_copy.shape[0]
            if proportion_actual >= tresh_proportion and proportion_estimate >= tresh_proportion:

                df = df[df['ticker'] != tic]
    
    return df


def linear_least_squares( df, plot, col='nw_total_sales_b_total'):
    """ 
    
    """
    df_copy = df.copy()
    # dropping stationary / mutual information / non-numeric columns
    df_copy = df_copy.drop(['month', 'ticker', 'mic', 'time'], axis=1)
    
    nan_rows = df_copy[df_copy[col].isna()]
    value_rows = df_copy[~df_copy[col].isna()]
    
    X, y = value_rows.drop(col, axis=1), value_rows[col]
    
    X['bias'] = np.ones(X.shape[0])
    
    weights = np.linalg.lstsq(X, y, rcond=None)[0]
    
    nan_rows = nan_rows.drop(col, axis=1)
    nan_rows['bias'] = np.ones(nan_rows.shape[0])
    
    new_vals = nan_rows @ weights
    
    df.loc[nan_rows.index, col] = new_vals
    
    if plot:
        plt.scatter(nan_rows.index, new_vals, label='Imputed Values', marker='x', color='red', alpha=.8)
        plt.scatter(y.index, y, label='Actual Values', marker='o', color='blue', alpha=.8)
        plt.plot(df.index, df[col], label='Concatenated', color='black', linestyle='--', alpha=0.5)
        plt.title(f"Company: {df['ticker'].unique()[0]}, column: {col}")
        plt.legend()
        plt.show()
        
    return df

def get_nan_indices( df, ticker, col):
    """
    
    """
    df_copy = df[df['ticker'] == ticker]
    original_index = df_copy.index
    df_copy = df_copy.reset_index(drop=False)
    nan_indices = df_copy[df_copy[col].isnull()].index.tolist()

    return df_copy, original_index, nan_indices

def least_square_imputation( df, df_copy, tic, original_index, plot=False, col='nw_total_sales_b_total'):
    """
    
    """
    df_copy = linear_least_squares(df_copy, plot=plot, col=col)
    df_copy = df_copy.set_index(original_index)
    df.loc[df['ticker'] == tic,col] = df_copy[col]

    return df
    
    
def impute_nans_singular_column( df, col='nw_total_sales_b_total',plot=False, max_plots=10):
    """ 
    
    """
    
    assert max_plots <= df.shape[0], "cannot generate more plots than datapoints"

    n_plots = 0
    nan_companies = get_nan_columns(df)

    # TODO make more efficient?
    for tic in nan_companies.keys():

        if set(nan_companies[tic]) == set([col]):
            df_copy, original_indices, nan_indices = get_nan_indices(df, tic, col=col)
            
            if set(nan_indices) == set([0,1,2]) or set(nan_indices) == set([0,1,2,3]) or set(nan_indices) == set([0,1]) or set(nan_indices) == set([0]) or len(nan_indices) <= 5:
                df = least_square_imputation(df, df_copy, tic, original_indices, plot=plot)
                n_plots += 1
                if n_plots == max_plots:
                    plot = False
    
    return df


def replace_sales( df, tic, company_list, plot, proportion, col_actual='Sales_Actual_fiscal', col_estimate = 'Sales_Estimate_fiscal'):
    """ 
    
    """
    def get_index_sets(df_idx, ticker, col_act, col_est):
        """ 
        
        """
        _, _, nan_indice_actual = get_nan_indices(df_idx, ticker, col=col_act)
        _, _, nan_indices_estimate = get_nan_indices(df_idx, ticker, col=col_est)
        actual_set, estimate_set = set(nan_indice_actual), set(nan_indices_estimate)
        
        return actual_set.difference(estimate_set), estimate_set.difference(actual_set), actual_set.intersection(estimate_set)
    
    actual_not_estimate, estimate_not_actual, actual_and_estimate = get_index_sets(df, tic, col_actual, col_estimate)
    
    # replace NaNs with normals when one column has values and the other doesn't
    
    if len(actual_not_estimate) != 0 or len(estimate_not_actual) != 0:
        mean_abs_diff = np.mean(abs(df[col_actual] - df[col_estimate]))
        std_dev = np.sqrt(mean_abs_diff)

        for col in [col_estimate, col_actual]:
            mask = df[col].isna()
            other_col = col_estimate if col == col_actual else col_actual
            df.loc[mask, col] = np.random.normal(df[other_col][mask], std_dev, size=(mask.sum(),))
    

    # TODO Nearest Neighbor related for this scenario
    if len(actual_and_estimate) != 0 and set(company_list) == set([col_actual, col_estimate]):
        
        window = len(actual_and_estimate) + 2
        rolling_mean_actual = df[col_actual].rolling(window, min_periods=1).mean()
        rolling_mean_estimate = df[col_estimate].rolling(window, min_periods=1).mean()
            
        if 0 in actual_and_estimate or 1 in actual_and_estimate:    
            df[col_estimate] = df[col_estimate].fillna(rolling_mean_actual).bfill()
        else:
            df[col_estimate] = df[col_estimate].fillna(rolling_mean_estimate).ffill()
        
        df_copy, original_indices, nan_indices = get_nan_indices(df, tic, col=col_actual)
        if len(original_indices) / len(nan_indices) < proportion:
            df = least_square_imputation(df, df_copy, tic, original_indices, plot=plot, col=col_actual)
    
    return df

def fiscal_sales_imputation(df, plot=False, proportion=0.35,col_actual = 'Sales_Actual_fiscal', col_estimate= 'Sales_Estimate_fiscal', n_sales_a = 'nw_total_sales_a_total', n_sales_b = 'nw_total_sales_b_total'):
    """ 
    
    """
    # TODO make more efficient?
    nan_companies = get_nan_columns(df)

    for tic, company_list in nan_companies.items():
        
        if col_actual in company_list or col_estimate in company_list:
            df.loc[df['ticker'] == tic, :] = replace_sales(df, tic, company_list, plot=plot, proportion=proportion)
            df_copy = df[df['ticker'] == tic]
            
            if df_copy[col_estimate].isna().sum() == 0:
                
                if set(nan_companies[tic]) == set([col_actual, n_sales_a]):
                    df_less_nans = df_copy.drop(col_actual, axis=1)
                    _, original_indices, nan_indices = get_nan_indices(df, tic, col=n_sales_a)
                    df = least_square_imputation(df, df_less_nans, tic, original_indices, plot=plot, col=n_sales_a)
                
                elif set(nan_companies[tic]) == set([col_actual, n_sales_b]):
                    df_less_nans = df_copy.drop(col_actual, axis=1)
                    _, original_indices, nan_indices = get_nan_indices(df, tic, col=n_sales_b)
                    
                    if len(original_indices) /len(nan_indices) <= proportion:
                        df = least_square_imputation(df, df_less_nans, tic, original_indices, plot=plot, col=n_sales_b)
    
    return df

def impute_singular_nan_column( df, proportion=0.35, max_plots=10):
    assert max_plots <= df.shape[0], "cannot generate more plots than datapoints"

    # TODO make more efficient?
    nan_companines = get_nan_columns(df)
    n_plots = 0
    
    for tic, column_list in nan_companines.items():
        if len(column_list) == 1:
            df_copy, original_indices, nan_indices = get_nan_indices(df, tic, col=column_list[0])
            if len(nan_indices) / len(original_indices) <= proportion:
                df = least_square_imputation(df, df_copy, tic, original_indices, col=column_list[0],plot=plot)
                n_plots += 1
                if n_plots == max_plots:
                    plot = False
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
