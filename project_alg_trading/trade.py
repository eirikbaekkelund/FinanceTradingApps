import numpy as np
import data

def risk_free_buy_and_hold(df, leverage=5, start_val = 2*(10**5)):
    """
    Buys and holds SPY stock and uses the risk free rate to calculate the account value.
    """
    risk_free_account = np.zeros(len(df))
    buy_and_hold_account = np.zeros(len(df))

    buy_and_hold_account[0] = start_val * leverage
    risk_free_account[0] = start_val * leverage

    for i in range(1,len(df)):
        risk_free_account[i] = risk_free_account[i-1] * (1 + df['EFFR'][i])
        buy_and_hold_account[i] = buy_and_hold_account[i-1] * (1 + df['Excess Return'][i])
    
    df['Risk Free Account'] = risk_free_account - start_val * (leverage - 1)
    df['Buy and Hold Account'] = buy_and_hold_account - start_val * (leverage - 1)
    
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

    account[0] = start_val
  
    money_owed = 0
    trade_amount_list = [0]

    for i in range(1,len(df)):
        
        # buy signal 
        if df[col][i-1] < df['BB Lower'][i-1] and not hold_stock:
            signal[i] = 1
            hold_stock = True
            trade_amount = account[i-1] * leverage
            money_owed = account[i-1] * (leverage - 1)

        # sell signal (stop loss or bb upper)
        elif ( df[col][i-1] > df['BB Upper'][i-1] or df['Price Change'][i] < drop_lim ) and hold_stock:
            signal[i] = -1
            hold_stock = False
            money_owed = 0
        
        if hold_stock:
            position[i] = 1
            money_owed = money_owed * (1 + df['EFFR'][i])
            trade_amount = trade_amount * (1 +  df['Excess Return'][i] )
            account[i] = trade_amount  - money_owed

        else:
            account[i] = account[i-1] * ( 1 + df['EFFR'][i])
        
        try:
            trade_amount_list.append(trade_amount)
        
        except UnboundLocalError:
            trade_amount_list.append(0)
        

    df['Account BB'] = account - money_owed
    df['Signal BB'] = signal
    df['Position BB'] = position
    df['Leverage BB'] = np.array(trade_amount_list / account) - 1
    df['Trade Amount BB'] = np.array(trade_amount_list)
    df['Theta BB'] = np.array([df['Account'][i] if df['Position'][i] == 1 else 0 for i in range(len(df))])

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
        # buy signal
        if (df[col][i-1] > df[ma_col][i-1] + sigma) and hold_stock == False:
            signal[i] = 1
            hold_stock = True

            # if we were in short position, recover the short
            if short_val > 0:
                account[i] = account[i-1] + (short_val - short_val_start)
                short_val = 0
                short_val_start = 0
            
            trade_amount =  account[i-1] * leverage
            theta[i-1] = trade_amount

        # short signal
        elif (df[col][i] < df[ma_col][i] - sigma ) and hold_stock == True:
            signal[i] = -1
            position[i] = -1
            hold_stock = False
            theta[i] = account[i-1] * leverage

        if hold_stock:
            theta[i] = theta[i-1] *( 1  +  df['Excess Return'][i] )               
            account[i] = theta[i] - theta[i-1] + account[i-1]
            position[i] = 1
        
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
