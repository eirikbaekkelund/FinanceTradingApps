import pandas as pd
import os
import holidays
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
import seaborn as sns

class DataFrameProcessor():
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

    def merge_spendings_revenue(self, df_spendings, df_revenue):
        """ 
        
        """
        return df_spendings.merge(
                    df_revenue[['ticker', 'time', 'Sales_Actual_fiscal', 'Sales_Estimate_fiscal']], 
                    on=['ticker', 'time'], how='left'
                    )

    def create_stationary_covariates(self,df):
        """ 
        
        """
        df['time'] = pd.to_datetime(df['time'])
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['quarter'] = df['time'].dt.quarter
        return df
        
    def split_column(self, df, delimiter, column):
        """ 
        
        """
        split_df = df[column].str.split(delimiter, expand=True)
        split_df.columns = column.split(delimiter)
        split_df = split_df.iloc[:,1:]
        
        return pd.concat([df.drop(column, axis=1), split_df], axis=1)

    def add_holiday(self, df):
        """ 
        
        """
        us_holidays = holidays.US()

        def is_holiday(date):
            if date in us_holidays:
                return 1
            else:
                return 0
            
        df['time'] = pd.to_datetime(df['time'])
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

class ModelPipeline():
    
    def __init__(self, df):
        self.df = df
    
    def train_target_slicing(self, target='', time_col='date'):
        """
        
        """
        
        assert target != '', "Target column must be specified"
        assert target in self.df.columns, "Target column must be in Data Frame columns"
        
        cols_train = list(self.df.columns)
        cols_train.remove(target)
        col_target = [time_col, target]
        
        X, y = self.df[cols_train], self.df[col_target]
        return  X, y

    def train_test_split(self, proportion_train, col_target='', time_col='date'):
        """ 
        
        """
       
        X, y = self.train_target_slicing(self.df, target=col_target, time_col=time_col)
        n = X.shape[0]
        n_train = int( n * proportion_train )
        
        return X.iloc[:n_train,:], y.iloc[:n_train,:], X.iloc[n_train:,:], y.iloc[n_train:,:]

    def create_darts_series_from_df(self, time_col='date', freq='D'):
        """  
        
        """
        return TimeSeries.from_dataframe(self.df, time_col=time_col, freq=freq)

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
        
    def series_train_test(self, series, proportion=0.75):
        """ 
        
        """
        train, validation = series.split_before(proportion)
        return train, validation