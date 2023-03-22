import numpy as np
from l2distance import l2distance_numpy
import data
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import plot as pt

class GaussianProcess:
    def __init__(self, df, prop_train=0.7):
        self.df = df
        self.prop_train = int( len(df) * prop_train )
    @staticmethod
    def compute_kernel(X, Z, sigma=1, l=1, ktype='matern'):
        """
        Computes kernel function. Takes in two arrays of x values and z values and returns an array of kernel values.
        If input is (N, 1) and (M, 1), output is (N, M).

        Args:
            x (np.array): array of x values
            y (np.array): array of y values
            sigma (float): sigma value
            l (float): l value
        Returns:
            k (np.array): array of kernel values
        """
        assert ktype in ['matern', 'rbf', 'linear', 'poly']
        
        if ktype == 'linear': 
            return (np.matmul(X, Z.T))
        
        elif ktype == 'rbf': 
            return (np.exp(-l*np.power(l2distance_numpy(X, Z), 2)))
        
        elif ktype == 'matern':
            d = l2distance_numpy(X, Z)
            k = sigma**2 * (1 + np.sqrt(3)*d/l) * np.exp(-np.sqrt(3)*d/l)
        
        else: 
            return (np.power(np.matmul(X, Z.T) + 1, sigma))
        
        n = k.shape[0]
        if n == k.shape[1]:
            return k + 1e-6 * np.eye(n)
        else:
            return k
    
    def sample_prior(self, X, Z, mu=None, sigma=1, l=1, ktype='rbf', n_samples=10):
        """
        Sample from the prior distribution.

        Args:
            X (np.array): array of x values
            Z (np.array): array of z values
            sigma (float): sigma value
            l (float): l value
        Returns:
            f (np.array): array of sampled values
        """
        
        K = self.compute_kernel(X, Z, sigma, l, ktype)

        if mu is None:
            mu = np.zeros(K.shape[0])
        
        else:
            mu = mu.reshape(-1)
        
        f = np.random.multivariate_normal(mu, K, size=n_samples)
        return f
    
    def gp_posterior(self, X_old, X_new, mu_old, mu_new=None, n_samples=10, sigma=1, l=1, ktype='rbf'):
        """
        Compute the posterior distribution of the GP.
        It is assumed that the prior distribution is a multivariate normal distribution
        where the mean is the past price values and 

        Args:
            X (np.array): array of x values
            Z (np.array): array of z values
            y (np.array): array of y values
            sigma (float): sigma value
            l (float): l value
        Returns:
            f (np.array): array of sampled values
        """
    
        k_11 = self.compute_kernel(X_old, X_old, sigma, l, ktype)
        k_12 = self.compute_kernel(X_old, X_new, sigma, l, ktype)
        k_22 = self.compute_kernel(X_new, X_new, sigma, l, ktype)

        if mu_old is None:
            mu_old = np.zeros(k_11.shape[0])
        else:
            mu_old = mu_old.reshape(-1)
        
        if mu_new is None:
            mu_new = np.zeros(k_22.shape[0])
        else:
            mu_new = mu_new.reshape(-1)
        
        # compute mean of GP prior
        # TODO may need to be altered and additive noise to computation of posterior mean and cov 
        f_old = np.mean( np.random.multivariate_normal(mu_old, k_11, size=20), axis=0)
        
        mu_conditional = mu_new + np.matmul(np.matmul(k_12.T, np.linalg.inv(k_11)), f_old - mu_old) 
        k_conditional =  k_22 - np.matmul(np.matmul(k_12.T, np.linalg.inv(k_11)), k_12)  

        f = np.random.multivariate_normal(mu_conditional, k_conditional, size=n_samples)

        return f
    
    def fit(self, col='SPY', kernel_cols=['SPY', 'Excess Return', 'EFFR'], sigma=1, l=1, ktype='matern', n_samples=20, n_preds=None, run_type='train'):
        """ 
        Computes the posterior prediction of the GP iteratively.
        Starts with a prior and then updates the posterior with new data for each time step.
        The posterior is used to predict the next time step. Then the posterior is updated with the new data.

        At time 0 we do not have any data, so we set the posterior to the prior.
        We do not know the mean of the posterior, so we set it to the mean of the last observation of the prior.

        Args:
            df (pd.DataFrame): dataframe with data
            sigma (float): sigma value
            l (float): l value
            n_samples (int): number of samples to draw from prior and posterior
        
        Returns:
            df (pd.DataFrame): dataframe with predictions
        
        """
        assert col in self.df.columns
        assert run_type in ['train', 'test']
        assert ktype in ['matern', 'rbf', 'linear', 'poly']
        assert n_preds is None or n_preds > 0
        assert n_samples > 0
        assert sigma > 0
        assert l > 0
        assert [col in self.df.columns for col in kernel_cols]
        
        df_new = self.df.copy()
        
        if run_type == 'train':
            df_new = df_new[:self.prop_train]
        else:
            df_new = df_new[self.prop_train:]

        gp_preds = np.zeros(df_new.shape[0])
        gp_std_devs = np.zeros(df_new.shape[0])
        gp_uppers = np.zeros(df_new.shape[0])
        gp_lowers = np.zeros(df_new.shape[0])
        
        X = df_new[kernel_cols].values
        mu = df_new[col].values
    
       # TODO change forecast horizon and length scale parameter to the kernel
        for i in range(3, len(df_new)):
            X_old = X[:i]
            mu_old = mu[:i]
            if col == 'Excess Return':
                eps_mu = 0.0001*np.random.randn(1)
                eps_X = 0.1*np.random.randn(1, X.shape[1])
            else:
                eps_mu = 0.0001*np.random.randn(1)
                eps_X = np.random.randn(1, X.shape[1])
            
            posterior = self.gp_posterior(
                                    X_old=X_old, 
                                    X_new=X_old[-1].reshape(1, -1) + eps_X,
                                    mu_old=mu_old, 
                                    mu_new=mu_old[-1].reshape(1, -1) + eps_mu,
                                    n_samples=n_samples, 
                                    sigma=sigma, l=l, ktype=ktype
                                    )
            
            gp_preds[i] = np.mean(posterior, axis=0)[-1]
            gp_std_devs[i] = np.std(posterior, axis=0)[-1]
            gp_uppers[i] = gp_preds[i] + gp_std_devs[i]
            gp_lowers[i] = gp_preds[i] - gp_std_devs[i]
            
            print(f"Epoch: {i-3} | \t Abs Error: {abs(gp_preds[i] - df_new[col][i])} ")


            if i == n_preds and n_preds is not None:
                gp_preds = gp_preds[:n_preds]
                gp_std_devs = gp_std_devs[:n_preds]
                gp_uppers = gp_uppers[:n_preds]
                gp_lowers = gp_lowers[:n_preds]

                df_new = df_new.iloc[:n_preds]
                df_new['GP'] = gp_preds
                df_new['GP'].iloc[:3] = self.df[col].iloc[:3]
                df_new['GP Std Dev'] = gp_std_devs
                df_new['GP Upper'] = gp_uppers
                df_new['GP Lower'] = gp_lowers

                return df_new

        
        df_new = df_new.iloc[:n_preds]

        df_new['GP'] = gp_preds
        df_new['GP'].iloc[:3] = self.df[col].iloc[:3]
        df_new['GP Std Dev'] = gp_std_devs
        df_new['GP Upper'] = gp_uppers
        df_new['GP Lower'] = gp_lowers

        df_new['GP Error'] = df_new['GP'] - df_new[col]

        return df_new
    
def trade_gp(df, leverage=5, start_val=2*(10**5), transaction_cost=0.5):
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

    gp_account = np.zeros(len(df))
    benchmark_account = np.zeros(len(df))

    # add starting value to the accounts for all trade strategies
    gp_account[0] = start_val
    benchmark_account[0] = start_val
    
    # TODO buy with benchmark at time 0, add transaction cost to benchmark
    # TODO change leverage and transaction cost dynamic
    gp_signal = np.zeros(len(df))
    
    for i in range(1,len(df)):
        # sell
        if df['GP'][i] + df['GP Std Dev'][i] > df['Excess Return'][i]:
            
            gp_signal[i] = -1
            gp_account[i] = gp_account[i-1] + leverage * df['SPY'][i] - transaction_cost
        
        # buy
        elif df['GP'][i] - df['GP Std Dev'][i] < df['Excess Return'][i]:
            
            gp_signal[i] = 1
            gp_account[i] = gp_account[i-1] - leverage * df['SPY'][i] - transaction_cost
        
        else:
            gp_account[i] = gp_account[i-1]
            
        benchmark_account[i] = benchmark_account[i-1] * ( 1 + df['EFFR'][i] )
    
    df['GP Account'] = gp_account
    df['Risk Free Account'] = benchmark_account
    df['GP Signal'] = gp_signal

    return df
    

if __name__ == "__main__":
    
    print('\n')
    df = data.get_data()
    df = data.excess_return_unit(df)
    target_col = 'Excess Return'
    
    if target_col == 'Excess Return':
        df[target_col] =  (df[target_col] - df[target_col].mean() ) / df[target_col].std()


    df = data.exponential_moving_averages(df, col=target_col)
    df = data.moving_averages(df, col=target_col)
    
    kernel_cols = ['SPY', 'Excess Return', 'EFFR']
    target_col = 'Excess Return'
        
    gaussian_process = GaussianProcess(df)
   
    df_posterior_train = gaussian_process.fit(col=target_col, 
                                              kernel_cols=kernel_cols, 
                                              ktype='matern', 
                                              l=1, 
                                              n_samples=50, 
                                              run_type='train')
    df_posterior_test = gaussian_process.fit(col=target_col,
                                             kernel_cols=kernel_cols,
                                             ktype='matern', 
                                             l=1, 
                                             n_samples=50, 
                                             run_type='test')

    df_posterior_train = trade_gp(df_posterior_train, leverage=5, start_val=2*(10**5), transaction_cost=0.5)
    df_posterior_test = trade_gp(df_posterior_test, leverage=5, start_val=2*(10**5), transaction_cost=0.5)

    pt.plot_gp(df_posterior=df_posterior_train)
    pt.plot_gp(df_posterior=df_posterior_test)
    
    pt.plot_time_series_annotated(df=df_posterior_train,
                                  x_col='Date',
                                  y_cols=['GP Account', 'Risk Free Account'],
                                  ylabel='Account Value ($)',
                                  title='GP vs Buy and Hold (TRAIN)',
                                  )
    
    pt.plot_time_series_annotated(df=df_posterior_test,
                                    x_col='Date',
                                    y_cols=['GP Account', 'Risk Free Account'],
                                    ylabel='Account Value ($)',
                                    title='GP vs Buy and Hold (TEST)',
                                    )
    
    # # check if the next value is higher or lower than the past value
    # gp_diff = np.array([1 if df_posterior['GP'].iloc[i] > df_posterior['GP'].iloc[i-1] else -1 for i in range(1, len(df_posterior))])
    # return_diff = np.array([1 if df_posterior[target_col].iloc[i] > df_posterior[target_col].iloc[i-1] else -1 for i in range(1, len(df_posterior))])

    # # check if the prediction was correct
    # correct = np.sum(gp_diff == return_diff)
    
    # print(f'Correct: {correct} | Total: {len(gp_diff)} | Accuracy: {correct / len(gp_diff)}')

