import numpy as np
# TODO implement column names as parameters

def turnover_dollars(df):
    """
    Calculates the turnover in dollars.
    """
    df = df[(df['Position'] == 1) | (df['Position'] == -1)]
    turnover = df['Theta'].diff().sum()
    return turnover

def turnover_units(df):
    """
    Calculates the turnover in units.
    """
    df = df[df['Theta'] != 0]
    df = df.reset_index(drop=True)
    turnover_units = np.sum([ abs( df['Theta'][i+1] / df['SPY'][i+1] - df['Theta'][i] / df['SPY'][i] ) for i in range(len(df)-1)])
    return turnover_units


def delta_value(df):
    """
    Calculates the change in value of the portfolio.
    """
    delta_v = df['Excess Return'] * df['Theta']
    return delta_v

def delta_cap(df, leverage_limit=5):
    """
    Calculates the capped change in value of the portfolio by 
    the leverage limit.
    """
    delta_v_total = df['Account']
    M = df['Theta'].abs() / leverage_limit
    delta_v_cap = (delta_v_total - M) * df['EFFR']

    return delta_v_cap

def delta_total_value(df):
    """
    Calculates the total change in value of the portfolio.
    """
    delta_v = delta_value(df)
    delta_v_cap = delta_cap(df)
    delta_v_total =  delta_v + delta_v_cap
    return delta_v_total

def sharpe_ration(df):
    """
    Calculates the sharpe ratio of the portfolio.
    """
    delta_v_total = delta_total_value(df)
    sharpe_ratio = delta_v_total.mean() / delta_v_total.std()
    return sharpe_ratio

def sortino_ratio(df):
    """
    Calculates the sortino ratio of the portfolio.
    """
    delta_v_total = delta_total_value(df)
    sortino_ratio = delta_v_total.mean() / delta_v_total[delta_v_total < 0].std()
    return sortino_ratio

def max_drawdown(df):
    """
    Calculates the maximum drawdown of the portfolio.
    """
    max_drawdown = (df['Account'] / df['Account'].cummax() - 1).min()
    return max_drawdown

def calmar_ratio(df):
    """
    Calculates the calmar ratio of the portfolio.
    """
    calmar_ratio = sharpe_ration(df) / max_drawdown(df)
    return calmar_ratio

def excess_return_account(df):
    """
    Calculates the excess return of the portfolio.
    """
    excess_return = df['Excess Return'] * df['Theta']
    return excess_return