import numpy as np
from scipy.stats import norm
def transaction_cost(position_size, price, transaction_cost=0.5):
    """
    Computes the transaction cost for a given position size.

    Args:
        position_size (float): position size
        price (float): price of the stock
        transaction_cost (float): transaction cost in percent
    
    Returns:
        transaction_cost (float): transaction cost
    """
    return position_size * price * transaction_cost

def kelly_return(kelly_fraction, account_val, price, leverage):
    """
    Computes the optimal fraction of the account value to invest in the stock.
    The optimal fraction is computed based on the Kelly Criterion.
    Also consideres the leverage available based on account value.

    Args:
        df (pd.DataFrame): dataframe with data
    """
    # assume kelly fraction is N(0,1)
    kelly_proportion = norm(loc=0, scale=1).cdf(kelly_fraction)
    
    if kelly_proportion * account_val <= (account_val * leverage) / price:
        return kelly_proportion * account_val
    
    return (account_val * leverage) / price

def trade_return_sell(account_val, kelly_fraction, price_acquired, price_sold, effr, leverage, iter=None):
    """
    Computes the return of a trade based on the Kelly fraction and the price acquired and sold.

    Args:
        kelly_fraction (float): fraction of the account value to invest in the stock
        price_acquired (float): price at which the stock was acquired
        price_sold (float): price at which the stock was sold
        effr (float): effective risk free rate
    
    Returns:
        return (float): return of the trade
    """
    
    kelly_fraction = kelly_return(kelly_fraction=kelly_fraction,
                                account_val=account_val,
                                price=price_acquired,
                                leverage=leverage)
   
    print('kelly fraction: ', kelly_fraction)
    
    return price_sold * kelly_fraction * ( (price_sold - price_acquired) / price_acquired - effr )
    
    
def trade(df, leverage=5, start_val=2*(10**5)):
    """
    Trading strategy based on the GP predictions.

    Args:
        df (pd.DataFrame): dataframe with data
        leverage (int): leverage
        start_val (int): starting value of the account
        transaction_cost (float): transaction cost in percent
    
    Returns:
        df (pd.DataFrame): dataframe with GP trading strategy
    """

    account = np.zeros(len(df))
    benchmark_account = np.zeros(len(df))
    buy_and_hold_account = np.zeros(len(df))
    gp_signal = np.zeros(len(df))

    # add starting value to the accounts for all trade strategies
    account[0] = start_val
    benchmark_account[0] = start_val
    buy_and_hold_account[0] = start_val
    
    # TODO implement leverage
    hold_stock = False

    for i in range(1,len(df)):

        # buy signal
        if -1 < df['Excess Return Standardized'][i-1] and not hold_stock: # df['GP'][i-1] + df['GP Std Dev'][i-1]
            gp_signal[i] = 1
            hold_stock = True
        
        # sell signal
        elif 1 > df['Excess Return Standardized'][i-1] and hold_stock: # df['GP'][i-1]- df['GP Std Dev'][i-1] 
            gp_signal[i] = -1
            hold_stock = False

        if  hold_stock:
            account[i] = account[i-1] * ( 1 + df['Excess Return'][i] )
        else:
            account[i] = account[i-1] * ( 1 + df['EFFR'][i])
        
        
        
        benchmark_account[i] = benchmark_account[i-1] * ( 1 + df['EFFR'][i] )
        buy_and_hold_account[i] = buy_and_hold_account[i-1] * ( 1 + df['Excess Return'][i] )
    
    df['Account'] = account
    df['Risk Free Account'] = benchmark_account
    df['Buy and Hold Account'] = buy_and_hold_account
    df['Signal'] = gp_signal

    return df  

