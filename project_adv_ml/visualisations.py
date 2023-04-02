import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from darts.metrics import smape, mse, rmse

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
    plt.matshow(df.corr(numeric_only=True), fignum=f.number, cmap='coolwarm')
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

def plot_predictions(predictions, targets, scalers, tickers):
    n_plots = min(len(predictions), 25)
    n_rows = int(np.ceil(n_plots / 5))
    n_cols = int(np.ceil(n_plots / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_rows* 5,n_cols* 5))
    # set suptitle
    axes = axes.flatten()
    for i in range(n_rows):
        for j in range(n_cols):
            k = i * n_rows + j
            if k >= n_plots:
                break
            pred = scalers[k].inverse_transform(predictions[k]).data_array().squeeze()
            target = scalers[k].inverse_transform(targets[k]).data_array().squeeze()
            axes[k].plot(pred.time, pred, label='prediction', marker = 'o', color = 'red')
            # TODO fix quantiles for predictions
            # axes[k].fill_between(pred.time, pred - pred.std(), pred + pred.std(), alpha=0.2, color='red')
            axes[k].plot(target.time, target, label='target', marker = 'x', color = 'blue')
            
            if k == 0:
                axes[k].legend()
            axes[k].set_ylim([target.min() - 100, 1.1 * target.max()])
            # rotate x axis labels, and include only 3 ticks on the x axis
            axes[k].tick_params(axis='x', labelrotation=30, labelsize=8)
            axes[k].tick_params(axis='y', labelrotation=30, labelsize=8)
            axes[k].set_title(f"{tickers[k]}", fontweight='bold')
    plt.show()

def plot_residuals(df, metric):
    residual_columns = ['res_nbeats_train', 'res_rf_train', 'res_xgb_train',
                        'res_nbeats_val', 'res_rf_val', 'res_xgb_val',
                        'res_nbeats_test', 'res_rf_test', 'res_xgb_test',
                        'res_lin_reg_train', 'res_lin_reg_val', 'res_lin_reg_test']
    assert all(col in df.columns for col in residual_columns), "residuals not found in dataframe"
    
    colors = ['red', 'blue', 'green', 'orange']
    
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=False, figsize=(20, 5))
    
    for i, subset in enumerate(['train', 'val', 'test']):
        cols = [f"res_{model}_{subset}" for model in ['nbeats', 'rf', 'xgb', 'lin_reg']]
        for j, col in enumerate(cols):
            label = col.split('_')[1]
            sns.histplot(data=df, x=col, color=colors[j], alpha=0.3, kde=True, ax=ax[i], bins=8, label=f'{label.upper()}')
        ax[i].set_title(f"{subset.capitalize()} Residuals", fontweight='bold')
        ax[i].set_xlabel(f"{metric.upper()}")
    plt.legend()
    plt.show()

def print_table(output_length, targets, preds_nbeats, preds_rf, preds_xgb, preds_lin_reg, preds_analysts, subset):
    print(f'\t\t\t\t {subset.upper()} TABLE')
    print('-'*77)
    print('| Metric | N-BEATS \t | RF \t  | XGB    | Lin Reg \t | Analysts')
    print('-'*77)
    for metric in [rmse, mse, smape]:

      nbeats = eval(targets=targets, preds=preds_nbeats, n_preds=output_length, err_func=metric)
      rf = eval(targets=targets, preds=preds_rf, n_preds=output_length, err_func=metric)
      xgb = eval(targets=targets, preds=preds_xgb, n_preds=output_length, err_func=metric)
      lin_reg = eval(targets=targets, preds=preds_lin_reg, n_preds=output_length, err_func=metric)
      analysts = eval(targets=targets, preds=preds_analysts, n_preds=1, err_func=metric)
      
      metric_label = str(metric).split(' ')[1]
      
      if metric_label != 'smape':
        print(f"| {metric_label} \t | {np.mean(nbeats):.4f} \t | {np.mean(rf):.4f} | {np.mean(xgb):.4f} | {np.mean(lin_reg):.4f} \t | {np.mean(analysts):.4f}")
      else:
        print(f"| {metric_label}  | {np.mean(nbeats):.3f} \t | {np.mean(rf):.3f} | {np.mean(xgb):.3f}  | {np.mean(lin_reg):.3f} \t | {np.mean(analysts):.3f}")

      print('-'*77)
