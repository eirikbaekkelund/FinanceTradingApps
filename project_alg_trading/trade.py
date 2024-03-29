import numpy as np
import data

def risk_free_buy_and_hold(df, leverage=5, start_val = 2*(10**5)):
    """
    Buys and holds SPY stock and uses the risk free rate to calculate the account value.
    """
    risk_free_account = np.zeros(len(df))
    buy_and_hold_account = np.zeros(len(df))
    theta_buy = np.zeros(len(df))
    theta_risk_free = np.zeros(len(df))


    buy_and_hold_account[0] = start_val 
    risk_free_account[0] = start_val 

    theta_buy[0] = start_val * leverage
    theta_risk_free[0] = start_val * leverage

    for i in range(1,len(df)):

        theta_buy[i] = theta_buy[i-1] * (1 + df['Excess Return'][i])
        theta_risk_free[i] = theta_risk_free[i-1] * (1 + df['EFFR'][i])
        risk_free_account[i] = risk_free_account[i-1] + (theta_risk_free[i] - theta_risk_free[i-1])
        buy_and_hold_account[i] = buy_and_hold_account[i-1] + (theta_buy[i] - theta_buy[i-1])
    
    df['Account Risk Free'] = risk_free_account 
    df['Account Buy and Hold'] = buy_and_hold_account 
    
    df['Theta Risk Free'] = theta_risk_free
    df['Theta Buy and Hold'] = theta_buy
    
    return df


def bollinger_band_strategy(df, col='SPY', start_val = 2*(10**5), leverage=5, drop_lim = -0.02, window=20, sigma=2):
    """
    Buys and sells SPY stock based on the bollinger band strategy.
    If the price is below the lower band, buy.
    If the price is above the upper band, sell.
    """
    
    df = data.bollinger_bands(df, col=col, window=window, sigma=sigma)

    hold_stock = False

    signal = np.zeros(len(df))
    position = np.zeros(len(df))
    account = np.zeros(len(df))
    theta = np.zeros(len(df))

    account[0] = start_val
  
    for i in range(1,len(df)):
        
        # buy signal 
        if df[col][i-1] < df['BB Lower'][i-1] and not hold_stock:
            signal[i] = 1
            hold_stock = True
            theta[i-1] = account[i-1] * leverage

        # sell signal (stop loss or bb upper)
        elif ( df[col][i-1] > df['BB Upper'][i-1] or df['Price Change'][i] < drop_lim ) and hold_stock:
            signal[i] = -1
            hold_stock = False
        
        if hold_stock:
            position[i] = 1
            theta[i] = theta[i-1] * (1 +  df['Excess Return'][i] )
            account[i] = account[i-1]  + (theta[i] - theta[i-1])

            # refactor theta based on earnings
            theta[i] = account[i] * leverage


        else:
            account[i] = account[i-1] * ( 1 + df['EFFR'][i])
    
    
    df['Account BB'] = account 
    df['Signal BB'] = signal
    df['Position BB'] = position
    df['Theta BB'] = np.array([theta[i] * position[i] for i in range(len(df))])

    return df


def momentum_strategy(df, col='SPY', start_val=2*(10**5), leverage=5, window='20', sigma=2):
    """
    A momentum trading strategy that incorporates leverage.
    
    Parameters:
    df (pandas.DataFrame): A DataFrame containing the price data for the stock
    col (str): The column containing the price data
    start_val (float): The amount of unlevered capital to start with
    leverage (float): The amount of leverage to use in the strategy
    window (str): The window for the moving average calculation
    sigma (float): The standard deviation multiplier for the moving average
    
    Returns:
    pandas.DataFrame: A DataFrame containing the input data as well as the
                      signal, position, account value, leverage, and theta columns.
    """
    
    df = data.moving_averages(df, col)
    
    if window == '10':
        ma_col = f'10 MA {col}'
    elif window == '20':
        ma_col = f'20 MA {col}'
    else:
        ma_col = f'30 MA {col}' 

    hold_stock = False
    
    signal = np.zeros(len(df))
    position = np.zeros(len(df))
    account = np.zeros(len(df))
    theta = np.zeros(len(df))

    account[0] = start_val 

    short_val = 0
    short_val_start = 0  # Keep track of the short value when signal is -1
    
    for i in range(1, len(df)):
        # buy signal / close short position
        if df[col][i] > df[ma_col][i] - sigma and hold_stock == False: # maybe change to minus sigma
            signal[i] = 1
            hold_stock = True

            # if we were in short position, recover the short
            if short_val > 0:
                account[i] = account[i-1] + (short_val - short_val_start)
                short_val = 0
                short_val_start = 0
            
            theta[i-1] = account[i-1] * leverage

        # short signal
        elif (df[col][i] < df[ma_col][i] + sigma ) and hold_stock == True: # maybe change to plus sigma
            signal[i] = -1
            position[i] = -1
            hold_stock = False
            theta[i] = account[i-1] * leverage

        if hold_stock:
            theta[i] = theta[i-1] * ( 1  +  df['Excess Return'][i] )               
            account[i] = account[i-1] + ( theta[i] - theta[i-1] )
            position[i] = 1

            # refactor theta based on earnings at t = i
            if  abs(theta[i]) > abs(account[i] * leverage):
                theta[i] = account[i] * leverage
        
        else:
            if short_val_start == 0:
                short_val_start = theta[i]
                short_val = short_val_start
            else:
                short_val = short_val * (1 + df['Excess Return'][i])   
            
            account[i] = account[i-1] * (1 + df['EFFR'][i])

    df['Signal Momentum'] = signal
    df['Position Momentum'] = position
    df['Account Momentum'] = account
    df['Theta Momentum'] = np.array([theta[i] * position[i] for i in range(len(df))])

    return df


def diversified_strategy(df, window_bb, window_mom, sigma_bb, sigma_mom, drop_lim_bb, start_val=2*(10**5), leverage=5):
    """
    A diversified trading strategy that incorporates leverage.

    
    Parameters:
    df (pandas.DataFrame): A DataFrame containing the price data for the stock
    window_bb (str): The window for the bollinger band strategy
    window_mom (str): The window for the momentum strategy
    sigma_bb (float): The standard deviation multiplier for the bollinger band strategy
    sigma_mom (float): The standard deviation multiplier for the momentum strategy
    start_val (float): The amount of unlevered capital to start with
    leverage (float): The amount of leverage to use in the strategy
    
    Returns:
    pandas.DataFrame: A DataFrame containing the input data as well as the
                      signal, position, account value, leverage, and theta columns.
    """
    start_val_bb = start_val * 0.3
    start_val_mom = start_val * 0.3
    start_val_hold = start_val * 0.3
    start_val_risk_free = start_val * 0.1

    
    
    df_bb = bollinger_band_strategy(df, col='SPY', start_val=start_val_bb, leverage=leverage, window=window_bb, sigma=sigma_bb, drop_lim=drop_lim_bb)
    df_mom = momentum_strategy(df, col='SPY', start_val=start_val_mom, leverage=leverage, window=window_mom, sigma=sigma_mom)
    df_hold = risk_free_buy_and_hold(df, start_val=start_val_hold, leverage=leverage)
    df_risk_free = risk_free_buy_and_hold(df, start_val=start_val_risk_free, leverage=leverage)

    df['Account Diversified'] = df_bb['Account BB'] + df_mom['Account Momentum'] + df_hold['Account Buy and Hold'] + df_risk_free['Account Risk Free']
    df['Theta Diversified'] = df_bb['Theta BB'] + df_mom['Theta Momentum'] + df_hold['Theta Buy and Hold'] + df_risk_free['Theta Risk Free']

    return df
   