import numpy as np
import jax
import jax.numpy as jnp

from bayesian_regression_jax import mu_regression_jax, kappa_kappatheta_regression_jax

def mu_step_jax(priors, Theta_fill, Y, V, Delta, key):

    # unpack other par
    kappa = Theta_fill['kappa']
    kappatheta = Theta_fill['kappatheta']
    sigma_sq_V = Theta_fill['sigma_sq_V']
    rho = Theta_fill['rho']

    # do bayesian regression to obtain posterior density
    mu_post, sigma_sq_post = mu_regression_jax(priors, kappa, kappatheta, sigma_sq_V, rho, Y, V, Delta)

    # draw normal
    mu_new = mu_post + jnp.sqrt(sigma_sq_post)*jax.random.normal(key)

    return mu_new

def kappatheta_kappa_step_jax(priors, Theta_fill, Y, V, Delta, key):

    # unpack other par
    mu = Theta_fill['mu']
    sigma_sq_V = Theta_fill['sigma_sq_V']
    rho = Theta_fill['rho']

    # do bayesian regression to obtain posterior density
    Mu_post, Omega_post = kappa_kappatheta_regression_jax(priors, mu, sigma_sq_V, rho, Y, V, Delta)

    # draw normal
    draws = jax.random.multivariate_normal(key, Mu_post.flatten(), Omega_post)
    kappatheta, kappa = draws[0], draws[1]

    return kappatheta, kappa