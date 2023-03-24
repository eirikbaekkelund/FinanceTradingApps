import numpy as np
import data

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# import tensor board
import matplotlib.pyplot as plt
import plot as pt
import warnings
warnings.filterwarnings('ignore')

def bollinger_band_strategy(df, col='SPY'):
    """
    Buys and sells SPY stock based on the bollinger band strategy.
    If the price is below the lower band, buy.
    If the price is above the upper band, sell.
    """
    # TODO implement leverage
    hold_stock = False
    account = np.zeros(len(df))
    benchmark_account = np.zeros(len(df))
    buy_and_hold_account = np.zeros(len(df))
    signal = np.zeros(len(df))
    account[0] = 2*(10**5)
    benchmark_account[0] = 2*(10**5)
    buy_and_hold_account[0] = 2*(10**5)

    for i in range(1,len(df)):
        # buy signal 
        if df[col][i-1] < df['BB Lower'][i-1] and not hold_stock:
            signal[i] = 1
            hold_stock = True
        
        # sell signal
        elif ( df[col][i-1] > df['BB Upper'][i-1] or df['Price Change'][i]) < -0.02 and hold_stock:
            signal[i] = -1
            hold_stock = False

        if  hold_stock:
            account[i] = account[i-1] * ( 1 + df['Excess Return'][i] )
        else:
            account[i] = account[i-1] * ( 1 + df['EFFR'][i])

        benchmark_account[i] = benchmark_account[i-1] * ( 1 + df['EFFR'][i])
        buy_and_hold_account[i] = buy_and_hold_account[i-1] * ( 1 + df['Excess Return'][i])

        df['Account'] = account
        df['Benchmark Account'] = benchmark_account
        df['Buy and Hold Account'] = buy_and_hold_account
        df['Signal'] = signal
    
    return df

def mean_reversion_strategy(df, col='SPY'):
    """
    Buys and sells SPY stock based on the mean reversion strategy.
    historical average price and selling stocks that have risen above their historical average price. 
    In noisy stock series, we need to use a longer-term average to avoid getting caught in short-term fluctuations
    We base it on the 10 day moving average with bollinger bands.
    """
    # TODO implement leverage
    hold_stock = False
    account = np.zeros(len(df))
    benchmark_account = np.zeros(len(df))
    buy_and_hold_account = np.zeros(len(df))
    signal = np.zeros(len(df))
    account[0] = 2*(10**5)
    benchmark_account[0] = 2*(10**5)
    buy_and_hold_account[0] = 2*(10**5)

    for i in range(1,len(df)):
        # buy signal
        if df[col][i-1] < df['BB Lower'][i-1] and not hold_stock:
            signal[i] = 1
            hold_stock = True
        
        # sell signal
        elif df[col][i-1] > df['BB Upper'][i-1] and hold_stock:
            signal[i] = -1
            hold_stock = False
        
        if  hold_stock:
            account[i] = account[i-1] * ( 1 + df['Excess Return'][i] )
        else:
            account[i] = account[i-1] * ( 1 + df['EFFR'][i])

        
        benchmark_account[i] = benchmark_account[i-1] * ( 1 + df['EFFR'][i])
        buy_and_hold_account[i] = buy_and_hold_account[i-1] * ( 1 + df['Excess Return'][i])
    
    df['Account'] = account
    df['Benchmark Account'] = benchmark_account
    df['Buy and Hold Account'] = buy_and_hold_account
    df['Signal'] = signal
    
    return df

def k_fold_cross_val(X, y, model, k=5):
    """
    Performs k-fold cross validation on the data.
    """
    # Split the data into k folds
    X_split = np.array_split(X, k)
    y_split = np.array_split(y, k)
    accuracy = []

    # Perform k iterations of training and testing
    for i in range(k):
        # Split the data into training and testing sets
        X_train = np.concatenate(X_split[:i] + X_split[i+1:])
        X_test = X_split[i]
        y_train = np.concatenate(y_split[:i] + y_split[i+1:])
        y_test = y_split[i]

        # Train the model on the training set
        model.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = model.predict(X_test)

        # Calculate and print the accuracy
        accuracy.append(accuracy_score(y_test, predictions))
        print("Fold: {}, Accuracy: {}".format(i, accuracy_score(y_test, predictions)))
    
    print("Average Accuracy: {}".format(np.mean(accuracy)))

def cross_val_xgboost(X, y):
    """
    Performs cross validation on the data.
    """
    best_score = 0 
    for tree in [10, 50, 100, 200, 500]:
        for depth in [1, 3, 5, 7, 9]:
            for lr in [0.01, 0.05, 0.1, 0.2, 0.5]:
                model = xgb.XGBClassifier(n_estimators=tree, 
                                          max_depth=depth, 
                                          learning_rate=lr)
                scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                print("Tree: {}, Depth: {}, LR: {}, Accuracy: {}".format(tree, depth, lr, np.mean(scores)))

                if np.mean(scores) > best_score:
                    best_score = np.mean(scores)
                    best_tree = tree
                    best_depth = depth
                    best_lr = lr
    
    return best_tree, best_depth, best_lr

if __name__ == "__main__":


    target_col = 'SPY'
    
    df = data.get_data()
    df = data.excess_return_unit(df) 
    df = data.exponential_moving_averages(df, col=target_col)
    df = data.moving_averages(df, col=target_col)
    df = data.kelly_fraction(df, col=target_col)
    df = data.bollinger_bands(df, col=target_col)
    df = df.fillna(0)

    plt.plot(df['SPY'])
    plt.fill_between(df.index, df['BB Upper'], df['BB Lower'], color='blue', alpha=0.2)
    plt.show()
    
    df = bollinger_band_strategy(df, col='SPY')
    pt.plot_strategy(df)

    # get bollinger bands for 10 MA SPY
    df = data.bollinger_bands(df, col='30 MA SPY')
    df = mean_reversion_strategy(df, col='SPY')
    pt.plot_strategy(df)

X = df[['SPY', 'Excess Return', 'Price Change', 'Volume']]
# standardize the data
X = (X - X.mean()) / X.std()
y = np.where(df['SPY'].shift(-1) > df['SPY'], 1, 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# find best hyperparameters
best_tree, best_depth, best_lr = cross_val_xgboost(X_train, y_train)

# train on best hyperparameters
model = xgb.XGBClassifier(n_estimators=best_tree,
                            max_depth=best_depth,
                            learning_rate=best_lr)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate and print the accuracy
print("Accuracy: {}".format(accuracy_score(y_test, predictions)))
