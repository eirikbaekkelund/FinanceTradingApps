import pandas as pd
import os
import holidays
import numpy as np
from matplotlib import pyplot as plt
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

class DataFrameProcessor():
    """ 
    
    """
    def __init__(self, folder):
        self.folder = folder
        self.files = self.files_from_folder()
        if '.DS_Store' in self.files:
            self.files.remove('.DS_Store')
        self.revenue = self.files[0]
        self.spendings = self.files[-1]
        self.files = self.files_from_folder()
        self.col_list_spendings = ['mic', 'ticker', 'time', 'nw_total_sales_a_total','nw_total_sales_b_total']
        self.col_list_revenue = ['mic', 'ticker', 'time', 'Sales_Actual_fiscal','Sales_Estimate_fiscal']
    
    def path_finder(self, name):
        """ 
        
        """
        return os.path.abspath(name)

    def files_from_folder(self):
        """ 
        
        """
        folder_path = self.path_finder(self.folder)
        
        return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    def create_df(self, file_name):
        """ 
        
        """
        path = str(self.path_finder(self.folder) + '/' + file_name)
    
        df =  pd.read_excel(self.path_finder(path))
        
        if file_name == self.revenue:
            return df[self.col_list_revenue]
        
        if file_name == self.spendings:
            return df[self.col_list_spendings]
    
    def split_column(self, df, delimiter, column):
        """ 
        
        """
        split_df = df[column].str.split(delimiter, expand=True)
        split_df.columns = column.split(delimiter)
        split_df = split_df.iloc[:,1:]
        
        return pd.concat([df.drop(column, axis=1), split_df], axis=1)

    # ONLY FOR DAYS estimates
    def add_labor_days(self, df):
        """ 
        
        """
        us_holidays = holidays.US()

        def is_holiday(date):
            if date in us_holidays:
                return 1
            else:
                return 0
            
        df["is_holiday"] = df["time"].apply(lambda x: is_holiday(x))
        df['is_weekend'] = df['time'].dt.weekday.isin([5,6])
        df['is_workday'] = (~df['is_holiday']) & (~df['is_weekend']).astype(int)
        df['is_weekend'] = df['time'].dt.weekday.isin([5,6]).astype(int)
        
        return df
    
    def add_war_to_df(self,df):
        """ 
        
        """
        df['is_war'] = (pd.to_datetime(df['time']) >= '2022-02-24').astype(int)
        
        return df

    def convert_columns_to_numeric(self, df):
        """ 
        
        """
       
        for col in df.columns:
            try:
                pd.to_numeric(df[col], errors='raise')
                df[col] = pd.to_numeric(df[col])
            except ValueError or TypeError:
                pass
        
        return df
    
    def encode_index(self, df, column='mic', encoding = {'XAMS' :  0, 'XLON' : 1, 'XMEX' : 2, 'XNAS' : 3, 'XNYS' : 4, 'XPAR' : 5, 'XTKS' : 6, 'XTSE'  : 7, 'NaN' : 8} ):
        """  
        
        """
    
        if encoding is not None:
            df[column] = df[column].map(encoding)
        else:
            unique_names = df[column].unique()
            encoding = {name: i for i, name in enumerate(unique_names)}
            df[column] = df[column].map(encoding)
        
        return df

    def merge_spendings_revenue(self, df_spendings, df_revenue):
        """ 
        
        """
        return df_spendings.merge(
                    df_revenue[['ticker', 'time', 'Sales_Actual_fiscal', 'Sales_Estimate_fiscal']], 
                    on=['ticker', 'time'], how='left'
                    )
    
    def remove_small_tickers(self, df, n=9):
        """ 
        
        """
        ticker_counts = df['ticker'].value_counts()
        df = df[df['ticker'].isin(ticker_counts[ticker_counts >= n].index)]
        
        return df
    
    def get_nan_columns(self, df):
        """ 
        
        """
        nans_dict = {}
        
        for tic in np.unique(df.ticker):
            
            df_copy = df[df['ticker'] == tic]
            df_copy = df_copy.reset_index(drop=False)
            nans_dict[tic] = []
            
            for col in df_copy.columns:
                nan_count = df_copy[col].isnull().sum()
                if nan_count > 0:
                    nans_dict[tic].append(col)
        
        return nans_dict

    def remove_missing_ground_truth(self, df, tresh_proportion=0.4):
        """ 
        
        """
        nan_companies = self.get_nan_columns(df)

        for tic in nan_companies.keys():
            if 'Sales_Actual_fiscal' in nan_companies[tic] or 'Sales_Actual_fiscal' in nan_companies[tic]:
                df_copy = df[df['ticker'] == tic]
                proportion_actual = df_copy[df_copy['Sales_Actual_fiscal'].isna()].shape[0] / df_copy.shape[0]
                proportion_estimate = df_copy[df_copy['Sales_Estimate_fiscal'].isna()].shape[0] / df_copy.shape[0]

                if proportion_actual <= tresh_proportion and proportion_estimate <= tresh_proportion:

                    df = df[df['ticker'] != tic]
        
        return df
    
    def linear_least_squares(self, df, plot, col='nw_total_sales_b_total'):
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
            plt.legend()
            plt.show()
            
        return df

    def get_nan_indices(self, df, ticker, col):
        """
        
        """
        df_copy = df[df['ticker'] == ticker]
        original_index = df_copy.index
        df_copy = df_copy.reset_index(drop=False)
        nan_indices = df_copy[df_copy[col].isnull()].index.tolist()

        return df_copy, original_index, nan_indices

    def least_square_imputation(self, df, df_copy, tic, original_index, plot=False, col='nw_total_sales_b_total'):
        """
        
        """
        df_copy = self.linear_least_squares(df_copy, plot=plot, col=col)
        df_copy = df_copy.set_index(original_index)
        df.loc[df['ticker'] == tic,col] = df_copy[col]

        return df
        
        
    def impute_bankB_nans(self, df, plot=False):
        """ 
        
        """
        nan_companies = self.get_nan_columns(df)

        for tic in nan_companies.keys():

            if set(nan_companies[tic]) == set(['nw_total_sales_b_total']):
                df_copy, original_index, nan_indices = self.get_nan_indices(df, tic, col='nw_total_sales_b_total')
                
                if set(nan_indices) == set([0,1,2]) or set(nan_indices) == set([0,1,2,3]) or set(nan_indices) == set([0,1]) or set(nan_indices) == set([0]) or len(nan_indices) <= 5:
                    df = self.least_square_imputation(df, df_copy, tic, original_index, plot=plot)
        
        return df

    def create_stationary_covariates(self,df):
        """ 
        
        """
        df['time'] = pd.to_datetime(df['time'])
        # account for starting year
        df['year'] = df['time'].dt.year - 2018 
        df['month'] = df['time'].dt.month
        df['quarter'] = df['time'].dt.quarter
        
        return df


class ModelPipeline():
    """ 
    
    """
    def __init__(self, df):
        self.df = df
    
    def set_df_index(self, df):
        """ 
        
        """
        df = df.copy()
        df.loc[:, 'time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
        df = df.resample('Q').mean(numeric_only=True)
        df = df.asfreq('Q')
        
        return df
    
    def convert_df_to_series(self,df,covariates, target, static):
        """ 
        
        """
        df = self.set_df_index(df)
        if covariates is None:
            covariates = ['nw_total_sales_a_total', 'nw_total_sales_b_total','Sales_Estimate_fiscal', 
                        'year','month', 'quarter', 'is_war',]
        
        # TODO add scaling functions and imputations to missing values

        covs = TimeSeries.from_dataframe(df, value_cols=covariates, static_covariates=df[static], freq='Q')
        target = TimeSeries.from_dataframe(df,value_cols=target, freq='Q')
        
        return covs, target
    
    def get_covs_target_dict(self, covariates=None, target='Sales_Actual_fiscal', static='mic'):
        """ 
        
        """
        return {tag : self.convert_df_to_series(self.df[self.df['ticker'] == tag], covariates=covariates, target=target, static=static) for tag in self.df['ticker']}
    
    def train_test_split(self, series, proportion=0.75):
        """ 
        
        """
        train, validation = series.split_before(proportion)
        return train, validation

    def series_scale(self, series):
        """ 
        
        """
        scaler = Scaler()
        return scaler.fit_transform(series)

    def series_rescale(self, series):
        """ 
        
        """
        scaler = Scaler()
        return scaler.inverse_transform(series)

    def series_fill_missing_vals(self, series):
        """ 
        
        """
        filler = MissingValuesFiller()
        return filler.transform(series=series, method='quadratic')

    