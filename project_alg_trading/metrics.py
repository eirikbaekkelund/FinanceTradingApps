import numpy as np
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
def max_drawdown(pnl):
    """
    Calculates the maximum drawdown of the portfolio.
    """
    return (pnl / (pnl.cummax() - 1) ) .min()

def calmar_ratio(pnl):
    """
    Calculates the calmar ratio of the portfolio
    """
    return sharpe_ratio(pnl) / max_drawdown(pnl)

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