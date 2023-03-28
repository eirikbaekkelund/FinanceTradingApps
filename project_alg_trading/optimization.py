from trade import bollinger_band_strategy, momentum_strategy

def cross_val_bollinger_band_strategy(df):
    """
    Cross validates the bollinger band strategy.
    """
    best_acc = 0
    for window in [5, 10, 20, 30]:
        for sigma in [1,2,3,4]:
            for drop_lim in [-0.01, -0.02, -0.03, -0.04, -0.05]:
                df = bollinger_band_strategy(df, window=window, sigma=sigma, drop_lim=drop_lim)
                if df['Account BB'].iloc[-1] > best_acc:
                    best_acc = df['Account BB'].iloc[-1]
                    best_window = window
                    best_sigma = sigma
                    best_drop_lim = drop_lim
    
    print('Best window: ', best_window)
    print('Best sigma: ', best_sigma)
    print('Best drop limit: ', best_drop_lim)
    
    return best_acc, best_window, best_sigma, best_drop_lim

def cross_val_momentum_strategy(df):

    best_acc = 0
    best_window = 0
    best_sigma = 0
    for window in [10,20,30]:
        for sigma in [0, 0.05, 0.5, 1,2,3]:
            df_new = momentum_strategy(df, window=window, sigma=sigma)
            if df_new['Account Momentum'].iloc[-1] > best_acc:
                best_window = window
                best_sigma = sigma
                best_acc = df_new['Account Momentum'].iloc[-1]
    
    print(f'Best window: {best_window}')
    print(f'Best sigma: {best_sigma}')
    
    df = momentum_strategy(df, window=best_window, sigma=best_sigma)

    return df, best_window, best_sigma