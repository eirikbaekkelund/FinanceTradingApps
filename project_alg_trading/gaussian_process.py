import numpy as np
import pandas as pd
from l2distance import l2distance_numpy

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
    
    def gp_posterior(self, X_old, X_new, mu_old, n_samples=10, sigma=1, l=1, ktype='rbf'):
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

        # add noise to k_11 and mu_old
        k_11 += np.random.randn(k_11.shape[0], k_11.shape[1]) #*0.005
        mu_old += np.random.randn(mu_old.shape[0]) * 0.1
    
        mu_conditional =  np.matmul(  np.matmul(k_12.T, np.linalg.inv(k_11)) , mu_old ) 
        k_conditional =  k_22 - np.matmul(  np.matmul(k_12.T, np.linalg.inv(k_11)) , k_12)  
    
        f = np.random.multivariate_normal(mu_conditional, k_conditional, size=n_samples)

        return f
    
    def fit(self, col, kernel_cols, sigma=1, l=1, ktype='matern', n_samples=20, run_type='train'):
        """ 
        Computes the posterior prediction of the GP iteratively.
        Starts with a prior and then updates the posterior with new data for each time step.
        The posterior is used to predict the next time step. Then the posterior is updated with the new data.

        At time 0 we do not have any data, so we set the posterior to the prior.
        We do not know the mean of the posterior, so we set it to the mean of the last observation of the prior.

        Args:
            col (str): column to predict
            kernel_cols (list): columns to use as kernel
            sigma (float): sigma value
            l (float): l value
            ktype (str): kernel type
            n_samples (int): number of samples to draw from prior and posterior
            run_type (str): 'train' or 'test'
        Returns:
            df (pd.DataFrame): dataframe with predictions
        
        """
        assert col in self.df.columns
        assert run_type in ['train', 'test', 'all']
        assert ktype in ['matern', 'rbf', 'linear', 'poly']
        assert n_samples > 0
        assert sigma > 0
        assert l > 0
        assert [colmn in self.df.columns for colmn in kernel_cols]
        
        df_new = self.df.copy()
        
        if run_type == 'train':
            df_new = df_new[:self.prop_train]
        elif run_type == 'test':
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
       
            
            posterior = self.gp_posterior(
                                    X_old=X_old, 
                                    X_new=X[i].reshape(1, -1),
                                    mu_old=mu_old, 
                                    n_samples=n_samples, 
                                    sigma=sigma, l=l, ktype=ktype
                                    )
            
            gp_preds[i] = np.mean(posterior, axis=0)[-1]
            gp_std_devs[i] = np.std(posterior, axis=0)[-1]
            gp_uppers[i] = gp_preds[i] + gp_std_devs[i]
            gp_lowers[i] = gp_preds[i] - gp_std_devs[i]
            
            print(f"Epoch: {i-3} | Abs Error: {abs(gp_preds[i] - df_new[col][i])}")
        

        df_new['GP'] = gp_preds
        df_new['GP'].iloc[:3] = self.df[col].iloc[:3]
        df_new['GP Std Dev'] = gp_std_devs
        df_new['GP Upper'] = gp_uppers
        df_new['GP Lower'] = gp_lowers

        df_new['GP Error'] = df_new['GP'] - df_new[col]

        idx = df_new[df_new['GP Error'].abs() > 3].index
        # set to 3 * sign of error
        df_new['GP'].loc[idx] = 3 * np.sign(df_new['GP Error'].loc[idx])

        return df_new