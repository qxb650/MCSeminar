import numpy as np
import scipy.stats as stats
import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
from jax.scipy.special import gammaln

from bayesian_regression import sigma_sq_V_regression, sigma_sq_V_regression_jax
from ll import eps_V_t, eps_Y_t, obs_ll_t, obs_ll, latent_ll_t, latent_ll, eps_V_t_jax, eps_Y_t_jax, obs_ll_t_jax, obs_ll_jax, latent_ll_t_jax, latent_ll_jax

########################################
# GENERAL METROPOLIS-HASTINGS FUNCTION #
########################################

def Metropolis_Hastings_jax(x_g, draw_x_prop, prop_logdensity, post_logdensity, key, check_support_fn=None):

    key_draw, key_u = jax.random.split(key)

    # 1. draw proposal
    x_prop = draw_x_prop(key_draw)

    # 2. check that proposal has support, and let it be old draw if it has not
    in_support = jnp.asarray(True) if check_support_fn is None else check_support_fn(x_prop)
    x_eval = jnp.where(in_support, x_prop, x_g)

    # 3. log alpha
    log_alpha = (post_logdensity(x_eval) - post_logdensity(x_g)) \
              + (prop_logdensity(x_g)    - prop_logdensity(x_eval))

    # 4. accept/reject
    logu = jax.random.uniform(key_u)
    accept = in_support & jnp.isfinite(log_alpha) & (jnp.log(logu) < log_alpha)
    x_new = jnp.where(accept, x_prop, x_g)
    return x_new, accept # accept is array of boolean value

#########################################
# STEP FUNCTIONS FOR SIGMA_V, RHO AND V #
#########################################


def invgamma_logpdf_jax(x, a, b):
    return a*jnp.log(b) - gammaln(a) - (a+1)*jnp.log(x) - b/x

def sigma_V_step_jax(priors, Theta, Theta_fill, Y, V, Delta, key):

    # unpack prior and create prior density function
    alpha_sigma_sq_V_prior = priors['alpha_sigma_sq_V']
    beta_sigma_sq_V_prior = priors['beta_sigma_sq_V']
    prior_density = lambda x: invgamma_logpdf_jax(x, alpha_sigma_sq_V_prior, beta_sigma_sq_V_prior)

    # unpack other par
    mu = Theta_fill['mu']
    kappa = Theta_fill['kappa']
    kappatheta = Theta_fill['kappatheta']
    rho = Theta_fill['rho']

    # unpack old MH-draw
    sigma_sq_V_g = Theta['sigma_sq_V']

    # do bayesian regression to obtain proposal draw and define proposal density
    alpha_sigma_sq_V_prop, beta_sigma_sq_V_prop = sigma_sq_V_regression_jax(priors, kappa, kappatheta, V, Delta)
    draw_sigma_sq_V_prop = lambda key_: beta_sigma_sq_V_prop / jax.random.gamma(key_, alpha_sigma_sq_V_prop)
    prop_density = lambda x: invgamma_logpdf_jax(x, alpha_sigma_sq_V_prop, beta_sigma_sq_V_prop)

    # define posterior density function
    post_density = lambda x: obs_ll_jax(mu, kappa, kappatheta, x, rho, Y, V, Delta) + latent_ll_jax(kappa, kappatheta, x, V, Delta) + prior_density(x)

    # do MH-step
    sigma_sq_V_gplus, accept = Metropolis_Hastings_jax(sigma_sq_V_g, draw_sigma_sq_V_prop, prop_density, post_density, key)

    return sigma_sq_V_gplus, accept

def rho_step_jax(priors, Theta, Theta_fill, Y, V, Delta, key):

    # unpack other par
    mu = Theta_fill['mu']
    kappa = Theta_fill['kappa']
    kappatheta = Theta_fill['kappatheta']
    sigma_sq_V = Theta_fill['sigma_sq_V']

    # unpack old MH-draw
    rho_g = Theta['rho']

    # shape V and Y
    T = len(Y)
    Yt = Y[:-1] # T-1: Y1,Y2,...,YT-1
    Ytplus = Y[1:] # T-1: Y2,Y3,...,YT
    Vt = V[:-1] # T-1: V1,V2,...,VT-1
    Vtplus = V[1:] # T-1: V2,V3,...,VT

    # compute innovations
    eps_Vt = eps_V_t_jax(kappa, kappatheta, sigma_sq_V, Vt, Vtplus, Delta)
    eps_Yt = eps_Y_t_jax(mu, Yt, Ytplus, Vt, Delta)

    # rho proposal
    rho_mean = jnp.corrcoef(eps_Vt, eps_Yt)[0,1]
    rho_std = ((1-rho_mean**2)/jnp.sqrt(T-1))
    draw_rho_prop = lambda key_: rho_mean + rho_std*jax.random.normal(key_)

    # function that checks if rho is in [-1,1]
    check_support_fn = lambda x: jnp.abs(x) <= 1.0

    # posterior log density
    post_logdensity = lambda x: obs_ll_jax(mu, kappa, kappatheta, sigma_sq_V, x, Y, V, Delta)

    # proposal density
    prop_logdensity = lambda x: jstats.norm.logpdf(x, loc=rho_mean, scale=rho_std)

    # do MH-step
    rho_gplus, accept = Metropolis_Hastings_jax(rho_g, draw_rho_prop, prop_logdensity, post_logdensity, key, check_support_fn=check_support_fn)

    return rho_gplus, accept

def V_step_jax(priors, Theta, Theta_fill, Y, V, Delta, key, c=2.4):
    
    # par
    mu = Theta_fill['mu']
    kappa = Theta_fill['kappa']
    kappatheta = Theta_fill['kappatheta']
    sigma_sq_V = Theta_fill['sigma_sq_V']
    rho = Theta_fill['rho']
    T = V.shape[0]

    # t = 1: p(Vt|Vt,Vt+1,Yt,Yt+1,Theta) = p(Yt+1-Yt|Vt,Vt+1,Theta) * p(Vt+1-Vt|Theta)
    # t = 2,3,...,T-1: p(Vt|Vt-1,Vt+1,Yt-1,Yt,Yt+1,Theta) = p(Yt+1-Y_t|Vt,Vt+1,Theta) * p(Yt-Yt-1|Vt-1,Vt,Theta) * p(Vt+1-Vt|Theta) * p(Vt-Vt-1|Theta)
    # t = T: p(Vt|Vt-1,Yt-1,Yt,Theta) = p(Yt-Yt-1|Vt-1,Vt,Theta) * p(Vt-Vt-1|Theta)

    Ypad = jnp.concatenate([Y[:1], Y, Y[-1:]]) # Y0,Y0,Y1,Y2,...,YT-1,YT,YT
    Vpad = jnp.concatenate([V[:1], V, V[-1:]]) # same for V
    
    # a key for each iteration
    keys = jax.random.split(key, T) 

    # define body for scan function
    def body(Vpad, carry):
        t, key_t = carry
        tpad = t + 1

        # unpack Vt-1,Vt,Vt+1,Yt-1,Yt,Yt+1
        Vt_g, Vminus1, Vplus = Vpad[tpad], Vpad[tpad-1], Vpad[tpad+1] # remember Vpad is shifted one to the right
        Yminus1, Yt, Yn = Ypad[tpad-1], Ypad[tpad], Ypad[tpad+1]

        # boolean for whether both neighbors are avaliable
        has_prev = t > 0
        has_next = t < T-1

        # construct Vbar
        n_neighbors = has_prev.astype(jnp.float64) + has_next.astype(jnp.float64)
        Vbar = (jnp.where(has_prev, Vminus1, 0.0) + jnp.where(has_next, Vplus, 0.0))/n_neighbors
        prop_std = c*jnp.sqrt(sigma_sq_V*Vbar*Delta/n_neighbors)

        def post_logdensity(x):

            # unpack neighbours 
            Vplus_s = jnp.where(has_next, Vplus, x)
            Vminus1_s = jnp.where(has_prev, Vminus1, x)
            fwd = obs_ll_t_jax(mu, kappa, kappatheta, sigma_sq_V, rho, Yt, Yn, x, Vplus_s, Delta) \
                + latent_ll_t_jax(kappa, kappatheta, sigma_sq_V, x, Vplus_s, Delta)
            bwd = obs_ll_t_jax(mu, kappa, kappatheta, sigma_sq_V, rho, Yminus1, Yt, Vminus1_s, x, Delta) \
                + latent_ll_t_jax(kappa, kappatheta, sigma_sq_V, Vminus1_s, x, Delta)
            return jnp.where(has_next, fwd, 0.0) + jnp.where(has_prev, bwd, 0.0)

        Vt_new, acc = Metropolis_Hastings_jax(
            Vt_g,
            lambda key_: Vt_g + prop_std*jax.random.normal(key_), # draw proposal function
            lambda x: 0.0, # ratio function
            post_logdensity, # post logdensity
            key_t, # key
            check_support_fn=lambda x: x > 0.0) # support
        
        return Vpad.at[tpad].set(Vt_new), acc

    Vpad, accs = jax.lax.scan(body, Vpad, (jnp.arange(T), keys))
    return Vpad[1:-1], accs                        # nyt V (T,) og accept pr. periode (T,)