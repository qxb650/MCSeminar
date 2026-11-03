import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from types import SimpleNamespace

class BlackScholes:

    def __init__(self, path='SP500.csv', priors=None):
        
        self.load_and_transform_data(path=path)
        self.setup_priors(priors)

    def load_and_transform_data(self, path):

        Y = pd.read_csv('SP500.csv')['SP500'].apply(lambda x: np.log(x)).diff().dropna().to_numpy()
        T = len(Y)

        self.Y = Y
        self.T = T

    def setup_priors(self, priors):

        priors = {}

        priors['mu'] = {'mu_p' : 0, 'sigma_sq_p' : 100}
        priors['sigma_sq'] = {'alpha_p' : 2.0, 'beta_p' : 0.001}

        if priors is not None:
            for var, var_dict in priors.items():
                for k, v in var_dict.items():
                    priors[var][k] = v

        self.priors = priors

    def plot_Y(self):
        plt.hist(self.Y, bins=100, density=True);

    def plot_priors(self, N_=1000):

        priors = self.priors

        mu_grid = np.linspace(-5*np.sqrt(priors['mu']['sigma_sq_p']), 5*np.sqrt(priors['mu']['sigma_sq_p']), N_)
        sigma_grid = np.linspace(0, 0.05, N_)

        f, ax = plt.subplots(1,2,figsize=(12,5))

        ax[0].plot(mu_grid, st.norm.pdf(mu_grid, scale=np.sqrt(priors['mu']['sigma_sq_p'])))
        ax[1].plot(sigma_grid, st.invgamma.pdf(sigma_grid, a=priors['sigma_sq']['alpha_p'], scale=priors['sigma_sq']['beta_p']))

        ax[0].set_title(r'Prior for $\mu$')
        ax[1].set_title(r'Prior for $\sigma^2$');

    # functions for updating parameters of distributions
    def sigma_sq_N(self, T, sigma_sq_k):

        priors = self.priors

        return (T/(sigma_sq_k) + 1/(priors['mu']['sigma_sq_p']))**(-1)

    def mu_N(self, Y, T, sigma_sq_k, mu_prior, sigma_sq_prior):

        priors = self.priors

        sigma_sq_N_ = self.sigma_sq_N(T, sigma_sq_k)

        return sigma_sq_N_*(np.sum(Y)/(sigma_sq_k) + priors['mu']['mu_p']/(priors['mu']['sigma_sq_p']))

    def alpha_N(self, T):

        priors = self.priors

        return priors['sigma_sq']['alpha_p'] + T/2

    def beta_N(self, Y, mu_k):

        priors = self.priors

        return priors['sigma_sq']['beta_p'] + 0.5*(np.sum((Y-mu_k)**2))