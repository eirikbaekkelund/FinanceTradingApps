import numpy as np
from scipy.stats import chi2

def giacomini_white_test_equal(res1, res2, num_lags, alpha=0.05):
    """
    Performs the Giacomini-White test of equal predictive ability for two regression models.

    Args:
        res1: numpy array of residuals from model 1
        res2: numpy array of residuals from model 2
        num_lags: integer number of lags to include in the test
        alpha: significance level of the test

    Returns:
        p_value: float p-value of the test
    """
    # residual difference between the two models
    diff_res = res1 - res2

    # sample variance of the difference in residuals
    sigma2 = np.var(diff_res, ddof=1)

    # squared residuals at each lag up to num_lags
    squared_residuals = np.square(diff_res)
    autocorr = np.correlate(squared_residuals, squared_residuals, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    autocorr = autocorr[:num_lags]

    # Calculate the test statistic
    n = len(diff_res)
    k = num_lags
    Q = (n - k) * np.sum(autocorr) / sigma2

    # Calculate the p-value of the test
    p_value = 1 - chi2.cdf(Q, k)

    if p_value < alpha:
        print("The null hypothesis that the two models have equal predictive ability is rejected at significance level {}.".format(alpha))
    else:
        print("The null hypothesis that the two models have equal predictive ability is not rejected at significance level {}.".format(alpha))

    print("The p-value of the test is {}.".format(p_value))

    return p_value