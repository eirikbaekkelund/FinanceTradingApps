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

class Processor():
    def __init__(self, revenue, spendings, folder):
        self.revenue = revenue
        self.spendings = spendings
        self.folder = folder
        self.files = self.files_from_folder()
    
    def path_finder(self, name):
        
        return os.path.abspath(name)

    def files_from_folder(self):
        """ 
        
        """
        folder_path = self.path_finder(self.folder)
        return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    def create_df(self, folder, file_name):
        """ 
        
        """
        path = str(self.path_finder(folder) + '/' + file_name)
    
        return pd.read_excel(self.path_finder(path))

    def merge_revenue_spendings(self, df_spendings, df_revenue):
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

    def convert_columns_to_numeric(self, df):
        """ 
        
        """
        for col in df.columns:
            try:
                pd.to_numeric(df[col], errors='raise')
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass

        return df

    def train_target_slicing(self, df, target='', time_col='date'):
        """
        
        """
        
        assert target != '', "Target column must be specified"
        assert target in df.columns, "Target column must be in Data Frame columns"
        
        cols_train = list(df.columns)
        cols_train.remove(target)
        col_target = [time_col, target]
        
        X, y = df[cols_train], df[col_target]
        return  X, y

    def train_test_split(self, df, proportion_train, col_target='', time_col='date'):
        """ 
        
        """
       
        X, y = self.train_target_slicing(df=df, target=col_target, time_col=time_col)
        n = X.shape[0]
        n_train = int( n * proportion_train )
        return X.iloc[:n_train,:], y.iloc[:n_train,:], X.iloc[n_train:,:], y.iloc[n_train:,:]

    def set_date(self, df):
        """ 
        
        """
        df['date'] = pd.to_datetime(df['date'], format='%d %b %Y')
        # df['date'] =  df['date'].dt.to_period('D')
        return df

    def create_darts_series_from_df(self, df, time_col='date', freq='D'):
        """  
        
        """
        return TimeSeries.from_dataframe(self, df=df, time_col=time_col, freq=freq)

    def series_scale(self, series):
        scaler = Scaler()
        return scaler.fit_transform(series)

    def series_rescale(self, series):
        scaler = Scaler()
        return scaler.inverse_transform(series)

    def series_fill_missing_vals(self, series):
        filler = MissingValuesFiller()
        return filler.transform(series=series, method='quadratic')
        
    def series_train_test(self, series, proportion=0.75):
        train, validation = series.split_before(proportion)
        return train, validation