import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

############# VISUALIZATION TOOLS ################

def plot_scatter_log(df, col1, col2):
    """
    
    """
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12,5))
    
    ax[0].scatter(df[col1], df[col2], color='blue', marker='x', alpha=0.5);
    ax[0].set_ylabel(f'{col2}')
    ax[0].set_xlabel(f'{col1}')
    ax[0].set_title('Unnormalized')
    
    ax[1].scatter(np.log(df[col1]), np.log(df[col2]), color='red', marker='x', alpha=0.5);
    ax[1].set_ylabel(f'{col2}')
    ax[1].set_xlabel(f'{col1}')
    ax[1].set_title('Log-scaled')
    
def plot_scatter(df, col1, col2):
    """
    
    """
    plt.scatter(df[col1], df[col2], color='blue', marker='x')
    plt.title(f'{col1} vs. {col2}')
    plt.ylabel(f'{col2}')
    plt.xlabel(f'{col2}')

    plt.show()

def plot_hist(df, col1, col2):
    sns.histplot(data=df[[col1,col2]], log_scale=True)

def plot_correlation_matrix(df):
    """
    
    """
    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=17, fontweight='bold');
    plt.show()


def plot_sales_comparison(df, col1 = 'Sales_Actual_fiscal', col2 = 'Sales_Estimate_fiscal', max_plots=10):
    """
    
    """
    assert max_plots <= df.shape[0], "cannot generate more plots than datapoints"
    n_plots = 0
    
    for tic in df['ticker'].unique():
        if n_plots < max_plots:
            df_copy = df[df['ticker'] == tic]
            plt.plot(df_copy['Sales_Actual_fiscal'].index, df_copy['Sales_Actual_fiscal'], label='Actual', marker='x', color='red')
            plt.plot(df_copy['Sales_Estimate_fiscal'].index, df_copy['Sales_Estimate_fiscal'], label='Estimate', marker='o')
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.title(tic)
            plt.legend()
            plt.show()
            n_plots += 1

def plot_anamoly_detection(df, plot=False, max_plots=10):
    """
    
    """
    assert max_plots <= len(df.ticker.unique()), "cannot print more arrays than there are tickers"
    
    def detect_anomaly_ewma(timeseries):
        """ 
        
        """
        span = len(timeseries)
        timeseries = pd.Series(timeseries)
        average = timeseries.ewm(span=span).mean()
        residual = timeseries - average
        
        return np.where(np.abs(residual) > 3 * residual.std(), 1, 0)

    n_plots = 0
    
    for tic in df.ticker.unique():
        df_anomoly = df[df.ticker == tic]
        n_anamolies = 0
        for col in ['nw_total_sales_a_total','nw_total_sales_b_total','Sales_Actual_fiscal','Sales_Estimate_fiscal']:
            anamolies = detect_anomaly_ewma(df_anomoly[col])
            if 1 in list(anamolies) and plot:
                plt.plot(range(df_anomoly.shape[0]) ,df_anomoly[col], label= f"{col}, indices = {np.where(anamolies == 1)}")
                n_anamolies += 1
        if n_anamolies >= 1 and plot:
            plt.title(f"{tic}")
            plt.legend(loc='best')
            plt.show()
            
            n_plots += 1
        if n_plots == max_plots and plot:
            break
