import numpy as np

from bayesian_regression import mu_regression, kappa_kappatheta_regression

def mu_step(priors, Theta_fill, Y, V, Delta):

    # unpack other par
    kappa = Theta_fill['kappa'] # (N,)
    kappatheta = Theta_fill['kappatheta'] # (N,)
    sigma_sq_V = Theta_fill['sigma_sq_V'] # (N,)
    rho = Theta_fill['rho'] # (N,)

    # do bayesian regression to obtain posterior density
    mu_post, sigma_sq_post = mu_regression(priors, kappa, kappatheta, sigma_sq_V, rho, Y, V, Delta) # (N,), (N,)

    # draw normal by broadcasting
    mu_new = np.random.normal(loc=mu_post, scale=np.sqrt(sigma_sq_post)) # (N,)

    return mu_new # (N,)

def kappatheta_kappa_step(priors, Theta_fill, Y, V, Delta):

    # unpack other par
    mu = Theta_fill['mu'] # (N,)
    sigma_sq_V = Theta_fill['sigma_sq_V'] # (N,)
    rho = Theta_fill['rho'] # (N,)

    # do bayesian regression to obtain posterior density
    Mu_post, Omega_post = kappa_kappatheta_regression(priors, mu, sigma_sq_V, rho, Y, V, Delta) # (N,2), (N,2,2)

    # do cholesky decomposition of Omega_post by broadcasting
    L = np.linalg.cholesky(Omega_post) # (N,2,2)

    # draw
    standard_draws = np.random.standard_normal(Mu_post.shape,) # (N,2)

    # mean shift + variance
    draws = Mu_post + np.einsum('nij,nj->ni', L, standard_draws) # (N,2,2), (N,2) -> (N,2)

    while np.any(draws <= 0):
        standard_draws = np.random.standard_normal(Mu_post.shape,) # (N,2)
        draws = Mu_post + np.einsum('nij,nj->ni', L, standard_draws) # (N,2,2), (N,2) -> (N,2)

    kappatheta = draws[:,0] # (N,)
    kappa = draws[:,1] # (N,)

    return kappatheta, kappa