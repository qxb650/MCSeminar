import numpy as np
import scipy.stats as stats

from bayesian_regression import sigma_sq_V_regression
from ll import eps_V_t, eps_Y_t, obs_ll_t, obs_ll, latent_ll_t, latent_ll

########################################
# GENERAL METROPOLIS-HASTINGS FUNCTION #
########################################

def Metropolis_Hastings(x_g, draw_x_prop, prop_logdensity, post_logdensity, check_support_fn=None):

    # 1. draw proposal
    # 1.1 draw
    x_prop = draw_x_prop()

    # 1.2 check proposal within support of posterior
    if not check_support_fn is None:
        if not check_support_fn(x_prop):
            return x_g, False

    # 2. compute log_alpha
    # 2.a evaluate log proposal ratio: log(q(x_g)/q(x_prop)) = log(q(x_g)) - log(q(x_prop))
    prop_logdensity_x_g = prop_logdensity(x_g)
    prop_logdensity_x_prop = prop_logdensity(x_prop)
    prop_logratio = prop_logdensity_x_g - prop_logdensity_x_prop

    # 2.b evaluate posterior log density: log(p(x_prop)/p(x_g)) = log(p(x_prop)) - log(p(x_g))
    post_logdensity_x_g = post_logdensity(x_g)
    post_logdensity_x_prop = post_logdensity(x_prop)
    post_logratio = post_logdensity_x_prop - post_logdensity_x_g

    # 2.c compute log_alpha
    log_alpha = post_logratio + prop_logratio

    # 3. accept/reject
    # 3.1 draw uniform in [-1,1] and take log
    u = np.random.uniform()
    logu = np.log(u)

    if log_alpha > logu:
        x_gplus = x_prop
        accept = True

    else:
        x_gplus = x_g
        accept = False

    return x_gplus, accept

#########################################
# STEP FUNCTIONS FOR SIGMA_V, RHO AND V #
#########################################

def sigma_V_step(priors, Theta, Theta_fill, Y, V, Delta):

    # unpack prior and create prior density function
    alpha_sigma_sq_V_prior = priors['alpha_sigma_sq_V']
    beta_sigma_sq_V_prior = priors['beta_sigma_sq_V']
    prior_density = lambda x: stats.invgamma.logpdf(x, alpha_sigma_sq_V_prior, scale=beta_sigma_sq_V_prior)

    # unpack other par
    mu = Theta_fill['mu']
    kappa = Theta_fill['kappa']
    kappatheta = Theta_fill['kappatheta']
    rho = Theta_fill['rho']

    # unpack old MH-draw
    sigma_sq_V_g = Theta['sigma_sq_V']

    # do bayesian regression to obtain proposal draw and define proposal density
    alpha_sigma_sq_V_prop, beta_sigma_sq_V_prop = sigma_sq_V_regression(priors, kappa, kappatheta, V, Delta)
    draw_sigma_sq_V_prop = lambda: stats.invgamma.rvs(alpha_sigma_sq_V_prop, scale=beta_sigma_sq_V_prop)
    prop_density = lambda x: stats.invgamma.logpdf(x, alpha_sigma_sq_V_prop, scale=beta_sigma_sq_V_prop)

    # define posterior density function
    post_density = lambda x: obs_ll(mu, kappa, kappatheta, x, rho, Y, V, Delta) + latent_ll(kappa, kappatheta, x, V, Delta) + prior_density(x)

    # do MH-step
    sigma_sq_V_gplus, accept = Metropolis_Hastings(sigma_sq_V_g, draw_sigma_sq_V_prop, prop_density, post_density)

    return sigma_sq_V_gplus, accept

def rho_step(priors, Theta, Theta_fill, Y, V, Delta):

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
    eps_Vt = eps_V_t(kappa, kappatheta, sigma_sq_V, Vt, Vtplus, Delta)
    eps_Yt = eps_Y_t(mu, Yt, Ytplus, Vt, Delta)

    # rho proposal
    rho_mean = np.corrcoef(eps_Vt, eps_Yt)[0,1].item()
    rho_std = ((1-rho_mean**2)/np.sqrt(T-1)).item()
    draw_rho_prop = lambda: np.random.normal(loc=rho_mean, scale=rho_std)

    # function that checks if rho is in [-1,1]
    check_support_fn = lambda x: np.abs(x) <= 1.0

    # posterior log density
    post_logdensity = lambda x: obs_ll(mu, kappa, kappatheta, sigma_sq_V, x, Y, V, Delta)

    # proposal density
    prop_logdensity = lambda x: stats.norm.logpdf(x, loc=rho_mean, scale=rho_std)

    # do MH-step
    rho_gplus, accept = Metropolis_Hastings(rho_g, draw_rho_prop, prop_logdensity, post_logdensity, check_support_fn)

    return rho_gplus, accept

def V_step(priors, Theta, Theta_fill, Y, V, Delta):

    # par
    mu = Theta_fill['mu']
    kappa = Theta_fill['kappa']
    kappatheta = Theta_fill['kappatheta']
    sigma_sq_V = Theta_fill['sigma_sq_V']
    rho = Theta_fill['rho']

    # t = 1: p(Vt|Vt,Vt+1,Yt,Yt+1,Theta) = p(Yt+1-Yt|Vt,Vt+1,Theta) * p(Vt+1-Vt|Theta)
    # t = 2,3,...,T-1: p(Vt|Vt-1,Vt+1,Yt-1,Yt,Yt+1,Theta) = p(Yt+1-Y_t|Vt,Vt+1,Theta) * p(Yt-Yt-1|Vt-1,Vt,Theta) * p(Vt+1-Vt|Theta) * p(Vt-Vt-1|Theta)
    # t = T: p(Vt|Vt-1,Yt-1,Yt,Theta) = p(Yt-Yt-1|Vt-1,Vt,Theta) * p(Vt-Vt-1|Theta)
    
    # allocate
    V_gplus = np.zeros_like(V) + np.nan # T
    accepts = np.zeros_like(V) + np.nan # T
    T = len(V)
    
    for t in range(0,T):

        # obtain proposal distribution
        if t == 0:
            Vbar = V[t+1]
        
        elif t == T-1:
            Vbar = V[t-1]

        else:
            Vbar = 0.5*(V[t-1]+V[t+1])

        Vt_g = V[t]

        # function to draw proposal variance for period t
        draw_Vt_prop = lambda: np.random.normal(loc=Vt_g, scale=np.sqrt((sigma_sq_V*Vbar*Delta)/2))

        # function that cheks if proposal variance is positive
        check_support_fn = lambda x: x >= 0.0

        # for t = 1 and t = 1,2,...,T-1 -> can always unpack next-period variance
        if t<T-1:

            # t = 1 -> can't unpack former period variance -> zero out ll for t-1
            if t == 0:
                obs_lltmimus = lambda x: 0.0 
                latent_lltminus = lambda x: 0.0

            Vtplus_g = V[t+1]

            # g likelihood in period t
            obs_llt = lambda x: obs_ll_t(mu, kappa, kappatheta, sigma_sq_V, rho, Y[t], Y[t+1], x, Vtplus_g, Delta)
            latent_llt = lambda x: latent_ll_t(kappa, kappatheta, sigma_sq_V, x, Vtplus_g, Delta)

        # for t = 1,2,...,T-1 and t = T -> can always unpack former period variance
        if t>0: 

            # use just-filled in V, rather than old draw
            Vtminus_g = V_gplus[t-1]

            # g likelihood in period t-1
            obs_lltmimus = lambda x: obs_ll_t(mu, kappa, kappatheta, sigma_sq_V, rho, Y[t-1], Y[t], Vtminus_g, x, Delta)
            latent_lltminus = lambda x: latent_ll_t(kappa, kappatheta, sigma_sq_V, Vtminus_g, x, Delta)

            # t = T -> can't unpack next period variance -> zero out ll for t
            if t == T-1:
                obs_llt = lambda x: 0.0
                latent_llt = lambda x: 0.0

        # proposal density is degenerate because of RW-MH
        prop_logdensity = lambda x: 0.0

        # posterior log density
        post_logdensity = lambda x: obs_llt(x) + latent_llt(x) + obs_lltmimus(x) + latent_lltminus(x)

        # do MH-step
        Vt_gplus, accept_t = Metropolis_Hastings(Vt_g, draw_Vt_prop, prop_logdensity, post_logdensity, check_support_fn)

        V_gplus[t] = Vt_gplus
        accepts[t] = accept_t

    return V_gplus, accepts