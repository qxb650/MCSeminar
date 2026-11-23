import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from types import SimpleNamespace

from scipy.stats import skew, kurtosis

def Rhat_ess(x):
    
    # x is (m,n): m chains, each with n draws

    # unpack shape
    m, n = x.shape

    # compute between-chain-variance: 1) take mean over n-dimension, 2) compute variance between means over m-dimension
    # variance of the m chains means
    B = n * x.mean(axis=1).var(ddof=1)

    # compute within-chain-variance: 1) take variance over n-dimension, 2) compute mean over variances
    # mean of the m chains variances
    W = x.var(axis=1, ddof=1).mean()

    # estimate marginal posterior variance
    var = (n-1)/n*W + B/n

    # estimate R^2: potential reduction scale
    Rhat = np.sqrt(var/W)

    # estimate 
    def rho_hat(t):

        # slice: x_(m,t), x_(m,t+1), ..., x_(m,n), shape: (m,n-t)
        x_fwd = x[:, t:]

        # slice: x_(m,1), x_(m,2), ..., x_(m,t-1), shape (m,t)
        x_bwd = x[:, :-t]

        # x_(m,t) - x_(m,1), x_(m,t+1) - x_(m,2)
        x_tdiff = x_fwd - x_bwd

        V_t = np.mean(x_tdiff**2) # variogram at lag t

        return 1 - V_t/(2*var)

    # allocate list for rhos
    rho = [None, rho_hat(1)] 

    t = 1
    # t = 1,2,...,n-1
    while t + 2 <= n - 1:

        while len(rho) <= t + 2: rho.append(rho_hat(len(rho)))

        if rho[t+1] + rho[t+2] < 0: break 

        t += 2

    n_eff = m*n / (1 + 2*sum(rho[1:t+1]))

    return Rhat, n_eff, var, B, W

def print_conv_table(draws, burnin=0):

    names = [k for k in draws.__dict__.keys() if ('_accepts' not in k) and (k != 'V')]
    w = max(len('Parameter'), max(len(k) for k in names))
    w1, w2, w3, w4, w5, w6, w7, w8, w9, w10 = 14, 10, 10, 10, 10, 10, 8, 10, 12, 12

    print(f"{'Parameter':<{w}}  {'Posterior Mean':>{w1}}  {'s.e.':>{w2}}  {'2.5%':>{w3}}  {'97.5%':>{w4}}  {'Skew':>{w5}}  {'Kurt':>{w6}}  {'Rhat':>{w7}}  {'n_eff':>{w8}}  {'B':>{w9}}  {'W':>{w10}}")
    print('-' * (w + w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9 + w10 + 20))

    for k in names:
        
        # handle theta seperately
        if k == 'kappatheta':
            k = 'theta'
            draws_k = draws.kappatheta[...,burnin:] / draws.kappa[...,burnin:]
        else:
            draws_k = getattr(draws, k)[...,burnin:]

        Rhat, n_eff, var, B, W = Rhat_ess(draws_k)
        lo, hi = np.percentile(draws_k, [2.5, 97.5])
        sk = skew(draws_k, axis=None)
        ku = kurtosis(draws_k, axis=None)
        print(f"{k:<{w}}  {draws_k.mean():>{w1}.5f}  {np.sqrt(var):>{w2}.5f}  {lo:>{w3}.5f}  {hi:>{w4}.5f}  {sk:>{w5}.4f}  {ku:>{w6}.4f}  {Rhat:>{w7}.4f}  {n_eff:>{w8}.0f}  {B:>{w9}.5f}  {W:>{w10}.5f}")