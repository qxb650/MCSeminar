import numpy as np

from bayesian_regression import mu_regression, kappa_kappatheta_regression

def mu_step(priors, Theta_fill, Y, V, Delta):

    # unpack other par
    kappa = Theta_fill['kappa']
    kappatheta = Theta_fill['kappatheta']
    sigma_sq_V = Theta_fill['sigma_sq_V']
    rho = Theta_fill['rho']

    # do bayesian regression to obtain posterior density
    mu_post, sigma_sq_post = mu_regression(priors, kappa, kappatheta, sigma_sq_V, rho, Y, V, Delta)

    # draw normal
    mu_new = np.random.normal(loc=mu_post, scale=np.sqrt(sigma_sq_post))

    return mu_new

def kappatheta_kappa_step(priors, Theta_fill, Y, V, Delta):

    # unpack other par
    mu = Theta_fill['mu']
    sigma_sq_V = Theta_fill['sigma_sq_V']
    rho = Theta_fill['rho']

    # do bayesian regression to obtain posterior density
    Mu_post, Omega_post = kappa_kappatheta_regression(priors, mu, sigma_sq_V, rho, Y, V, Delta)

    # draw normal
    draws = np.random.multivariate_normal(mean=Mu_post.flatten(), cov=Omega_post)
    kappatheta, kappa = draws[0].item(), draws[1].item()

    return kappatheta, kappa