import numpy as np
import pandas as pd
from sklearn import decomposition


def add_random(df, n=2, seed=149, low=0, high=5):
    """ Add random vectors to the raw feature sets
    
    Args:
    df: pandas dataframe. the raw feature sets
    n: int. dimension of random vectors added
    seed: int. random seed
    low: int. low limit of the integer range
    high: int. high limit of the integer range

    Returns:
    df_rnd: pandas dataframe. random vectors added dataframe

    """
    df_rnd = df.copy()
    for i in range(n):
        np.random.seed(seed=seed)
        rnd = np.random.randint(low, high, size=len(df))
        df_rnd['rnd_'+str(i+1)] = rnd
        seed += i+1

    return df_rnd


def choose_pca_n(X, exp_var_ratio):
    pca = decomposition.PCA()
    pca.fit(X)
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    n_comp = next(x for x, val in enumerate(exp_var_cumul) if val > exp_var_ratio)
    n = n_comp + 1
    return n
