import numpy as np

def sharpe_ratio(pnl):
    """
    Calculates the sharpe ratio of the portfolio.
    """
    return pnl.mean() / pnl.std()

def sortino_ratio(pnl):
    """
    Calculates the sortino ratio of the portfolio.
    """
    return pnl.mean() / pnl[ pnl < 0 ].std()

# TODO verify max drawdown
def max_drawdown_percent(df, strategy = 'BB'):
    """
    Calculates the drawdown of the account.
    """
    account = f'Account {strategy}'
    
    drawdown =  df[account].div(df[account].cummax()).sub(1) * 100
    return abs(drawdown.min())

def max_drawdown(df, strategy = 'BB'):
    """
    Calculates the max drawdown of the portfolio
    """
    return max_drawdown_percent(df, strategy) / 100

def drawdown_value(df, strategy='BB'):
    """
    Calculates the drawdown of the account.
    """
    account = f'Account {strategy}'
    
    return df[account].sub(df[account].cummax())

def drawdown_percent(df, strategy='BB'):
    """
    Calculates the drawdown of the account.
    """
    account = f'Account {strategy}'
    
    return df[account].div(df[account].cummax()).sub(1) * 100

def compunded_rate(df, strategy='BB'):
    """
    Calculates the Calmar ratio of the account.
    """

    compounded_return = (1 + df[f'Pct Change {strategy}']).cumprod()
    cr = compounded_return.iloc[-1]
    
    return cr 

def calmar_ratio(df, strategy='BB'):
    """
    Calculates the Calmar ratio of the account.
    """

    compounded_return = (1 + df[f'Pct Change {strategy}']).cumprod()
    cr = compounded_return.iloc[-1]
    dd = drawdown_value(df, strategy=strategy)
    
    max_dd = dd.min()
    print(max_dd)
    calmar = cr / abs(max_dd)
    
    return calmar

def mean_return(pnl):
    """
    Calculates the mean return of the portfolio trades
    """
    return pnl.mean()

def median_return(pnl):
    """
    Calculates the median return of the portfolio of trades
    """
    return pnl.median()

def tracking_error(asset_return, portfolio_return, benchmark_return, risk_free_return):
    """
    Calculates the tracking error of the portfolio.
    """
# calculate excess returns of the asset over the risk-free rate
    asset_excess_return = asset_return - risk_free_return

    # calculate the excess return of the portfolio over the benchmark
    portfolio_excess_return = portfolio_return - benchmark_return
    return np.sqrt(np.sum((asset_excess_return - portfolio_excess_return) ** 2) / (len(asset_return) - 1))
