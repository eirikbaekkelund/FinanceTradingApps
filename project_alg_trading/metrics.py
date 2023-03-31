# TODO implement column names as parameters

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

def calmar_ratio(df, strategy = 'BB'):
    """
    Calculates the calmar ratio of the portfolio
    """
    excess_return = df[f'Pct Change {strategy}'].mean()
    return excess_return / max_drawdown(df, strategy)

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