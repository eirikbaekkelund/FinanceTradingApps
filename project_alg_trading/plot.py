from matplotlib import pyplot as plt
import numpy as np

def plot_excess_return(df):
    """ 
    Plot the excess return of S&P500 ETF 
    as compared to the risk-free rate.

    Args:
        df (pd.DataFrame): dataframe including excess return
    """
    plt.figure(figsize=(15, 5))
    
    plt.plot(df['Excess Return'])
    
    plt.title('Excess Return of S&P500 ETF', fontweight='bold')
    plt.xlabel('Date', fontweight='bold')
    plt.ylabel('Excess Return', fontweight='bold')

    # rotate x-axis labels
    # include 10 ticks 
    
    interval = len(df.index)//10
    plt.xticks(df.index[::interval], rotation=30)
    
    plt.show()

def plot_time_series_annotated(df, ylabel, x_col='Date', y_cols=['Close Price'], title='Apple', facecolor='whitesmoke', figsize=(14,4)):
    
    cmap = plt.cm.get_cmap('Set1', len(y_cols))
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.set_facecolor(facecolor)
    
    for i, col in enumerate(y_cols):
        line_color = cmap(i)
        df.plot(x=x_col, y=col, grid=True, ax=ax, label=None, color=line_color)
        max_val = df[col].max()
        min_val = df[col].min()
        
        # get index of max and min values
        max_index = df.index[df[col] == max_val]
        min_index = df.index[df[col] == min_val]

        # find position of max and min values in the dataframe
        max_index = df.index.get_loc(max_index[0])
        min_index = df.index.get_loc(min_index[0])
        ax.plot(max_index, max_val, marker='^', markersize=8, color=line_color)
        ax.annotate(f"{max_val:.2f}\nhigh", xy=(max_index, max_val), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontsize=8, fontweight='light')
        
        ax.plot(min_index, min_val, marker='v', markersize=8, color=line_color)
        ax.annotate(f"{min_val:.2f}\nlow", xy=(min_index, min_val), xytext=(-15, -5), 
                    textcoords='offset points', ha='center', fontsize=8, fontweight='light')
    
    plt.title(title, loc='center', fontsize=12, fontweight='bold', pad=20)
    plt.xlabel(x_col, fontsize=10, fontweight='bold')
    plt.xticks(fontsize=8)
    plt.ylabel(ylabel, fontsize=10, fontweight='bold')
    plt.yticks(fontsize=8)
    ax.tick_params(axis='both', which='both', length=0, labelrotation=25)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()

def plot_sampled_functions(df, f, title='Sampled Functions'):
    """
    Plot sampled functions.

    Args:
        X (np.array): array of x values
        f (np.array): array of sampled values
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df['Date'][-len(f[-1]):], f.T, alpha=0.5)
    n_ticks = 15
    ax.set_xticks(df['Date'][-len(f[-1]):][::len(f[-1]) // n_ticks])
    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(title)
    plt.show()

def plot_distribution(X, f, title='Distribution'):
    """
    Plot distribution.

    Args:
        X (np.array): array of x values
        f (np.array): array of sampled values
    """
    _ , ax = plt.subplots(figsize=(15, 5))
    
    ax.plot(np.arange(f.shape[-1]), f.mean(axis=0), label='mean')
    ax.fill_between(X[:, 0], f.mean(axis=0) - f.std(axis=0), f.mean(axis=0) + f.std(axis=0), alpha=0.5, label='std')
    
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(title)
    
    ax.legend()
    
    plt.show()

def plot_strategy(df):
    plt.rcParams["figure.figsize"] = (20,5)
    plt.plot(df['Account'], label='Account')
    plt.plot(df['Risk Free Account'], label='Risk Free Account')
    plt.plot(df['Buy and Hold Account'], label='Buy and Hold Account')
    
    mask_buy = df['Signal'] == 1
    mask_sell = df['Signal'] == -1

    plt.scatter(df[mask_buy].index, df[mask_buy]['Account'], marker='^', color='green', label='Buy')
    plt.scatter(df[mask_sell].index, df[mask_sell]['Account'], marker='v', color='red', label='Sell')

    plt.legend()
    plt.show()

def plot_margin(df, strategy = 'BB', leverage = 5):
    """ 
    Plot the margins account value * leverage

    Args:
        df (pd.DataFrame): data frame with account values
        strategy (str): specifies the strategy to plot
        leverage (int): specifies the leverage used in the strategy

    """
    # TODO make axes labels and title dynamic for different strategies
    
    account = f'Account {strategy}'
    signal = f'Signal {strategy}'
    theta = f'Theta {strategy}'
    
    plt.rcParams['figure.figsize'] = (20,5)
    plt.fill_between(x = df.index,
                     y1 = leverage * df[account],
                     y2 = - leverage * df[account],
                     color = 'purple',
                     alpha = 0.2,
                     label = '$[ - V_t \cdot L, V_t \cdot L]$')
    
    plt.plot(df[theta], label = '$\\theta_t$')
    try:
        mask_buy = df[signal] == 1
        mask_sell = df[signal] == -1

        plt.scatter(df[mask_buy].index, 
                    df[mask_buy][theta], 
                    marker='^',
                    color = 'green',
                    label = 'Buy / Short Close',
                    s=15,
                    alpha = 0.5)
        
        plt.scatter(df[mask_sell].index, 
                    df[mask_sell][theta], 
                    marker='v',
                    color = 'red',
                    label = 'Sell / Short',
                    s=15,
                    alpha = 0.5)
    except KeyError:
        print('Portfolio is diversified, contradicting signals are not plotted.')
        
    plt.legend()
    plt.show()

def plot_delta_V(pnl, delta_v, delta_vcap, strategy = 'BB'):
    # TODO make axes labels and title dynamic for different strategies
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(pnl, 
            linestyle = '-.',
            label = 'PnL')
    ax[0].plot(delta_v, 
            linestyle = '--',
            label = '$\Delta V_t$')
    ax[0].plot(delta_vcap, label = '$\Delta V_t^{cap}$')

    ax[0].legend()

    
    ax[1].plot(pnl.cumsum(), 
            linestyle = '-.',
            label = '$\sum$ PnL')
    ax[1].plot(delta_v.cumsum(), 
            linestyle = '--',
            label = '$\sum \Delta V_t$')
    ax[1].plot(delta_vcap.cumsum(), label = '$\sum \Delta V_t^{cap}$')

    ax[1].legend()

    plt.suptitle(f'PnL and $\Delta V$ for {strategy} Strategy')
    plt.show()

# TODO make it so the three strategies are plotted together for drawdown and sharpe ratios
def plot_drawdown(df):
    pass

def plot_sharpe_ratio(df):
    """
    
    """
    
    strategies = [col[7:] if col[:7] == 'Account' else '' for col in df.columns]
    strategies = strategies[strategies != '']

    for strategy in strategies:
        plt.scatter(df[f'PnL {strategy}'].mean(), df[f'PnL {strategy}'].std(), label = strategy)
    
    plt.scatter(df['Excess Return'].mean(), df['Excess Return'].std(), label='SPDR')
    plt.title('Sharpe Ratio')
    plt.xlabel('$\mu$')
    plt.ylabel('$\sigma$')


    
    
    
    

