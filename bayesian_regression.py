import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

from ll import eps_V_t, eps_Y_t

def mu_regression(priors, kappa, kappatheta, sigma_sq_V, rho, Y, V, Delta):

    # shape Y and V
    Y_noT = Y[:-1] # T-1: Y1,Y2,...,YT-1
    Y_leaded = Y[1:] # T-1: Y2,Y3,...,YT
    V_noT = V[:-1] # T-1: V1,V2,...,VT-1
    V_leaded = V[1:] # T-1: V2,V3,...,VT

    # compute V-error
    eps_Vt = eps_V_t(kappa, kappatheta, sigma_sq_V, V_noT, V_leaded, Delta) # (T-1,)

    # construct regressor x and regressand y
    y = ((Y_leaded - Y_noT + 0.5*V_noT*Delta)/(np.sqrt(V_noT*Delta)) - rho*eps_Vt)/(np.sqrt(1-rho**2)) # (T-1,)
    x = Delta/(np.sqrt(V_noT*Delta*(1-rho**2))) # (T-1,)

    # unpack prior
    mu_prior = priors['mu_mu']
    sigma_sq_prior = priors['sigma_sq_mu']

    # update prior into posterior given formulas from appendix
    sigma_sq_posterior = 1/((1/sigma_sq_prior) + np.sum(x**2))
    mu_posterior = sigma_sq_posterior*((mu_prior/sigma_sq_prior)+np.sum(x*y))

    return mu_posterior, sigma_sq_posterior

def kappa_kappatheta_regression(priors, mu, sigma_sq_V, rho, Y, V, Delta):

    # shape Y and V
    Y_noT = Y[:-1] # T-1: Y1,Y2,...,YT-1
    Y_leaded = Y[1:] # T-1: Y2,Y3,...,YT
    V_noT = V[:-1] # T-1: V1,V2,...,VT-1
    V_leaded = V[1:] # T-1: V2,V3,...,VT

    # compute Y-error
    eps_Yt = eps_Y_t(mu, Y_noT, Y_leaded, V_noT, Delta) # (T-1,)

    # construct regressor X and regressand y
    y = (((V_leaded - V_noT)/(np.sqrt(sigma_sq_V*V_noT*Delta))) - rho* eps_Yt)/(np.sqrt(1-rho**2)) # (T-1,)
    y = y[:,None] # (T-1,1)
    x1 = (np.sqrt(Delta))/(np.sqrt(sigma_sq_V*V_noT*(1-rho**2))) # (T-1,)
    x2 = - (np.sqrt(V_noT*Delta))/(np.sqrt(sigma_sq_V*(1-rho**2))) # (T-1,)
    X = np.concatenate([x1[:, None], x2[:, None]], axis=1) # (T-1,2)

    # unpack prior 
    mu_kappa = priors['mu_kappa']
    sigma_sq_kappa = priors['sigma_sq_kappa']
    mu_kappatheta = priors['mu_kappatheta']
    sigma_sq_kappatheta = priors['sigma_sq_kappatheta']

    # priors on bivariate matrix form
    Omega_prior = np.diag([sigma_sq_kappatheta, sigma_sq_kappa]) # (2,2)

    Mu_prior = np.array([
        [mu_kappatheta],
        [mu_kappa]
    ]) # (2,1)

    # update prior into posterior given formulas from appendix
    Omega_post = np.linalg.inv(np.linalg.inv(Omega_prior)+X.T @ X) # inv(Omega_prior): (2,2), X.T @ X: (2,2) -> Omega_post: (2,2)
    Mu_post = Omega_post @ (np.linalg.inv(Omega_prior) @ Mu_prior + X.T @ y) # inv(Omega_prior) @ Mu: (2,1), X.T @ y: (2,1) -> Mu_post: (2,1)

    return Mu_post, Omega_post

def sigma_sq_V_regression(priors, kappa, kappatheta, V, Delta):

    # shape V
    V_noT = V[:-1] # T-1: V1,V2,...,VT-1
    V_leaded = V[1:] # T-1: V2,V3,...,VT
    T = len(V_noT) + 1

    # construct regressand y
    y = (V_leaded - V_noT - kappatheta*Delta + kappa*V_noT*Delta)/(np.sqrt(V_noT*Delta)) # (T-1)

    # unpack prior
    alpha_sigma_sq_V_prior = priors['alpha_sigma_sq_V']
    beta_sigma_sq_V_prior = priors['beta_sigma_sq_V']

    # update prior into posterior given formulas from appendix
    alpha_sigma_sq_V_post = alpha_sigma_sq_V_prior + (T-1)/2
    beta_sigma_sq_V_post = beta_sigma_sq_V_prior + (np.sum(y**2))/2

    return alpha_sigma_sq_V_post, beta_sigma_sq_V_post