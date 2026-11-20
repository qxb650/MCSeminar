import numpy as np

from ll import eps_V_t, eps_Y_t

def mu_regression(priors, kappa, kappatheta, sigma_sq_V, rho, Y, V, Delta):

    # V shape is (N,T)
    # par shape is (N,)
    # Y shape is (T,)

    # shape Y and V
    Y_noT = Y[None,:-1] # T-1: Y1,Y2,...,YT-1, shape (1,T-1)
    Y_leaded = Y[None,1:] # T-1: Y2,Y3,...,YT, shape (1,T-1)
    V_noT = V[:, :-1] # T-1: V1,V2,...,VT-1, shape (N,T-1)
    V_leaded = V[:, 1:] # T-1: V2,V3,...,VT, shape (N,T-1)

    # compute V-error
    eps_Vt = eps_V_t(kappa[:,None], kappatheta[:,None], sigma_sq_V[:,None], V_noT, V_leaded, Delta) # (N,T-1)

    # construct regressor x and regressand y
    y = ((Y_leaded - Y_noT)/(np.sqrt(V_noT*Delta)) - rho[:,None]*eps_Vt)/(np.sqrt(1-rho[:,None]**2)) # (N,T-1)
    x = Delta/(np.sqrt(V_noT*Delta*(1-rho[:,None]**2))) # (N,T-1)

    # unpack prior
    mu_prior = priors['mu_mu']
    sigma_sq_prior = priors['sigma_sq_mu']

    # update prior into posterior given formulas from appendix
    sigma_sq_posterior = 1/((1/sigma_sq_prior) + np.sum(x**2, axis=1)) # (N,)
    mu_posterior = sigma_sq_posterior*((mu_prior/sigma_sq_prior)+np.sum(x*y, axis=1)) # (N,)

    return mu_posterior, sigma_sq_posterior

def kappa_kappatheta_regression(priors, mu, sigma_sq_V, rho, Y, V, Delta):

    # V shape is (N,T)
    # par shape is (N,)
    # Y shape is (T,)

    # shape Y and V
    Y_noT = Y[None,:-1] # T-1: Y1,Y2,...,YT-1, shape (1,T-1)
    Y_leaded = Y[None,1:] # T-1: Y2,Y3,...,YT, shape (1,T-1)
    V_noT = V[:, :-1] # T-1: V1,V2,...,VT-1, shape (N,T-1)
    V_leaded = V[:, 1:] # T-1: V2,V3,...,VT, shape (N,T-1)

    # compute Y-error
    eps_Yt = eps_Y_t(mu[:,None], Y_noT, Y_leaded, V_noT, Delta) # (N,T-1)

    # construct regressor X and regressand y
    y = (((V_leaded - V_noT)/(np.sqrt(sigma_sq_V[:,None]*V_noT*Delta))) - rho[:,None]* eps_Yt)/(np.sqrt(1-rho[:,None]**2)) # (N,T-1)
    x1 = (np.sqrt(Delta))/(np.sqrt(sigma_sq_V[:,None]*V_noT*(1-rho[:,None]**2))) # (N,T-1)
    x2 = - (np.sqrt(V_noT*Delta))/(np.sqrt(sigma_sq_V[:,None]*(1-rho[:,None]**2))) # (N,T-1)
    X = np.concatenate([x1[:, :, None], x2[:, :, None]], axis=2) # (N,T-1,2)

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

    # construct X'X and X'y
    XtX = np.einsum('nti,ntj->nij', X, X) # (N,T-1,2), (N,T-1,2) -> (N,2,2)
    Xty = np.einsum('nti,nt->ni',  X, y)  # (N,T-1,2), (N,T-1) -> (N,2)

    # update prior into posterior given formulas from appendix
    Omega_post = np.linalg.inv(np.linalg.inv(Omega_prior)[None,:,:]+XtX) # inv(Omega_prior): (1,2,2), X.T @ X: (N,2,2) -> Omega_post: (N,2,2)
    Mu_post = Omega_post @ (np.linalg.inv(Omega_prior)[None,:,:] @ Mu_prior + Xty[:,:,None]) # inv(Omega_prior) @ Mu: (1,2,1), X.T @ y: (N,2,1) -> Mu_post: (N,2,1)

    return Mu_post[:,:,0], Omega_post # (N,2), (N,2,2)

def sigma_sq_V_regression(priors, kappa, kappatheta, V, Delta):

    # shape V
    V_noT = V[:, :-1] # T-1: V1,V2,...,VT-1, shape (N,T-1)
    V_leaded = V[:, 1:] # T-1: V2,V3,...,VT, shape (N,T-1)
    N = V_noT.shape[0]
    T = V_noT.shape[1] + 1

    # construct regressand y
    y = (V_leaded - V_noT - kappatheta[:,None]*Delta + kappa[:,None]*V_noT*Delta)/(np.sqrt(V_noT*Delta)) # (N,T-1)

    # unpack prior
    alpha_sigma_sq_V_prior = priors['alpha_sigma_sq_V']
    beta_sigma_sq_V_prior = priors['beta_sigma_sq_V']

    # update prior into posterior given formulas from appendix
    alpha_sigma_sq_V_post = alpha_sigma_sq_V_prior + (T-1)/2 # (1,)
    alpha_sigma_sq_V_post = np.ones(N)*alpha_sigma_sq_V_post # (N,)
    beta_sigma_sq_V_post = beta_sigma_sq_V_prior + (np.sum(y**2, axis=1))/2 # (N,)

    return alpha_sigma_sq_V_post, beta_sigma_sq_V_post