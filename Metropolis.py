import numpy as np
import scipy.stats as stats

from bayesian_regression import sigma_sq_V_regression
from ll import eps_V_t, eps_Y_t, obs_ll_t, obs_ll, latent_ll_t, latent_ll

########################################
# GENERAL METROPOLIS-HASTINGS FUNCTION #
########################################

def Metropolis_Hastings(x_g, draw_x_prop, prop_logdensity, post_logdensity, check_support_fn=None):

    # 1. draw proposals
    x_prop = draw_x_prop() # (N,)

    # 2. compute log_alpha
    # 2.a evaluate log proposal ratio: log(q(x_g)/q(x_prop)) = log(q(x_g)) - log(q(x_prop))
    prop_logdensity_x_g = prop_logdensity(x_g) # (N,)
    prop_logdensity_x_prop = prop_logdensity(x_prop) # (N,)
    prop_logratio = prop_logdensity_x_g - prop_logdensity_x_prop # (N,)

    # 2.b evaluate posterior log density: log(p(x_prop)/p(x_g)) = log(p(x_prop)) - log(p(x_g))
    post_logdensity_x_g = post_logdensity(x_g) # (N,)
    post_logdensity_x_prop = post_logdensity(x_prop) # (N,)
    post_logratio = post_logdensity_x_prop - post_logdensity_x_g # (N,)

    # 2.c compute log_alpha
    log_alpha = post_logratio + prop_logratio # (N,)

    # 3. accept/reject
    # 3.1 draw uniform in [0,1] and take log
    u = np.random.uniform(size=x_g.shape)
    logu = np.log(u)

    if check_support_fn is None:
        accept = log_alpha > logu

    else:
        accept = (log_alpha > logu) & (check_support_fn(x_prop))

    x_gplus = np.where(accept, x_prop, x_g)

    return x_gplus, accept

#########################################
# STEP FUNCTIONS FOR SIGMA_V, RHO AND V #
#########################################

def draw_inv_gamma(alpha, beta):
    return beta/np.random.gamma(alpha)

def inv_gamma_logpdf_prop(x, alpha, beta):
    return -(alpha+1)*np.log(x) - beta/x # in proportions

def correlation_coefficent(X1,X2):

    T = X1.shape[-1]

    X1_dev = X1 - X1.mean(axis=-1, keepdims=True)
    X2_dev = X2 - X2.mean(axis=-1, keepdims=True)

    num = (X1_dev*X2_dev).sum(axis=-1)
    denom = np.sqrt((X1_dev**2).sum(axis=-1) * (X2_dev**2).sum(axis=-1))

    corr = num/denom
    corr_std = ((1-corr**2)/np.sqrt(T-3))

    return corr, corr_std

def normal_logpdf_prop(x, mu, sigma):

    return -np.log(sigma) - ((x-mu)**2)/(2*sigma**2) # in proportions

def sigma_V_step(priors, Theta, Theta_fill, Y, V, Delta):

    # unpack prior and create prior density function
    alpha_sigma_sq_V_prior = priors['alpha_sigma_sq_V']
    beta_sigma_sq_V_prior = priors['beta_sigma_sq_V']
    prior_density = lambda x: inv_gamma_logpdf_prop(x, alpha_sigma_sq_V_prior, beta_sigma_sq_V_prior)

    # unpack other par
    mu = Theta_fill['mu'] # (N,)
    kappa = Theta_fill['kappa'] # (N,)
    kappatheta = Theta_fill['kappatheta'] # (N,)
    rho = Theta_fill['rho'] # (N,)

    # unpack old MH-draw
    sigma_sq_V_g = Theta['sigma_sq_V'] # (N,)

    # do bayesian regression to obtain proposal draw and define proposal density
    alpha_sigma_sq_V_prop, beta_sigma_sq_V_prop = sigma_sq_V_regression(priors, kappa, kappatheta, V, Delta) # (N,)
    draw_sigma_sq_V_prop = lambda: draw_inv_gamma(alpha_sigma_sq_V_prop, beta_sigma_sq_V_prop) # returns (N,)
    prop_density = lambda x: inv_gamma_logpdf_prop(x, alpha_sigma_sq_V_prop, beta_sigma_sq_V_prop) # returns (N,)

    # define posterior density function
    post_density = lambda x: obs_ll(mu[:,None], kappa[:,None], kappatheta[:,None], x[:,None], rho[:,None], Y, V, Delta) + latent_ll(kappa[:,None], kappatheta[:,None], x[:,None], V, Delta) + prior_density(x) # returns (N,)

    # do MH-step
    sigma_sq_V_gplus, accept = Metropolis_Hastings(sigma_sq_V_g, draw_sigma_sq_V_prop, prop_density, post_density)

    return sigma_sq_V_gplus, accept

def rho_step(priors, Theta, Theta_fill, Y, V, Delta):

    # unpack other par
    mu = Theta_fill['mu'] # (N,)
    kappa = Theta_fill['kappa'] # (N,)
    kappatheta = Theta_fill['kappatheta'] # (N,)
    sigma_sq_V = Theta_fill['sigma_sq_V'] # (N,)

    # unpack old MH-draw
    rho_g = Theta['rho']

    # shape V and Y
    Yt = Y[None,:-1] # T-1: Y1,Y2,...,YT-1, shape (1,T-1)
    Ytplus = Y[None,1:] # T-1: Y2,Y3,...,YT, shape (1,T-1)
    Vt = V[:,:-1] # T-1: V1,V2,...,VT-1, shape (N,T-1)
    Vtplus = V[:,1:] # T-1: V2,V3,...,VT, shape (N,T-1)
    T = Vt.shape[1] + 1

    # compute innovations
    eps_Vt = eps_V_t(kappa[:,None], kappatheta[:,None], sigma_sq_V[:,None], Vt, Vtplus, Delta) # (N,T-1)
    eps_Yt = eps_Y_t(mu[:,None], Yt, Ytplus, Vt, Delta) # (N,T-1)

    # rho proposal
    rho_mean, rho_std = correlation_coefficent(eps_Vt, eps_Yt) # (N,)
    draw_rho_prop = lambda: np.random.normal(loc=rho_mean, scale=rho_std) # returns (N,)

    # function that checks if rho is in [-1,1]
    check_support_fn = lambda x: np.abs(x) < 1.0

    # posterior log density
    post_logdensity = lambda x: obs_ll(mu[:,None], kappa[:,None], kappatheta[:,None], sigma_sq_V[:,None], x[:,None], Y, V, Delta)

    # proposal density
    prop_logdensity = lambda x: normal_logpdf_prop(x, rho_mean, rho_std)

    # do MH-step
    rho_gplus, accept = Metropolis_Hastings(rho_g, draw_rho_prop, prop_logdensity, post_logdensity, check_support_fn=check_support_fn)

    return rho_gplus, accept

def V_step(priors, Theta, Theta_fill, Y, V, Delta):

    # par
    mu = Theta_fill['mu'] # (N,)
    kappa = Theta_fill['kappa'] # (N,)
    kappatheta = Theta_fill['kappatheta'] # (N,)
    sigma_sq_V = Theta_fill['sigma_sq_V'] # (N,)
    rho = Theta_fill['rho'] # (N,)

    # t = 1: p(Vt|Vt,Vt+1,Yt,Yt+1,Theta) = p(Yt+1-Yt|Vt,Vt+1,Theta) * p(Vt+1-Vt|Theta)
    # t = 2,3,...,T-1: p(Vt|Vt-1,Vt+1,Yt-1,Yt,Yt+1,Theta) = p(Yt+1-Y_t|Vt,Vt+1,Theta) * p(Yt-Yt-1|Vt-1,Vt,Theta) * p(Vt+1-Vt|Theta) * p(Vt-Vt-1|Theta)
    # t = T: p(Vt|Vt-1,Yt-1,Yt,Theta) = p(Yt-Yt-1|Vt-1,Vt,Theta) * p(Vt-Vt-1|Theta)
    
    # allocate
    V_gplus = np.zeros_like(V) + np.nan # (N,T)
    accepts = np.zeros_like(V) + np.nan # (N,T)
    T = V.shape[-1]
    
    # function that cheks if proposal variance is positive
    check_support_fn = lambda x: x > 0.0

    for t in range(0,T):

        # obtain proposal distribution
        if t == 0:
            Vbar = V[:,t+1] # (N,)
        
        elif t == T-1:
            Vbar = V_gplus[:,t-1] # (N,)

        else:
            Vbar = 0.5*(V_gplus[:,t-1]+V[:,t+1]) # (N,)

        Vt_g = V[:,t] # (N,)

        # function to draw proposal variance for period t
        draw_Vt_prop = lambda: np.random.normal(loc=Vt_g, scale=2*np.sqrt((sigma_sq_V*Vbar*Delta)/2)) # (N,)

        # for t = 1 and t = 1,2,...,T-1 -> can always unpack next-period variance
        if t<T-1:

            # t = 1 -> can't unpack former period variance -> zero out ll for t-1
            if t == 0:
                obs_lltmimus = lambda x: 0.0 
                latent_lltminus = lambda x: 0.0

            Vtplus_g = V[:,t+1] # (N,)

            # g likelihood in period t
            obs_llt = lambda x: obs_ll_t(mu, kappa, kappatheta, sigma_sq_V, rho, Y[t], Y[t+1], x, Vtplus_g, Delta) # returns (N,)
            latent_llt = lambda x: latent_ll_t(kappa, kappatheta, sigma_sq_V, x, Vtplus_g, Delta) # returns (N,)

        # for t = 1,2,...,T-1 and t = T -> can always unpack former period variance
        if t>0: 

            # use just-filled in V, rather than old draw
            Vtminus_g = V_gplus[:,t-1] # (N,)

            # g likelihood in period t-1
            obs_lltmimus = lambda x: obs_ll_t(mu, kappa, kappatheta, sigma_sq_V, rho, Y[t-1], Y[t], Vtminus_g, x, Delta) # returns (N,)
            latent_lltminus = lambda x: latent_ll_t(kappa, kappatheta, sigma_sq_V, Vtminus_g, x, Delta) # returns (N,)

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

        V_gplus[:,t] = Vt_gplus
        accepts[:,t] = accept_t

    return V_gplus, accepts

def V_step_2blocks(priors, Theta, Theta_fill, Y, V, Delta):

    # par
    mu = Theta_fill['mu'] # (N,)
    kappa = Theta_fill['kappa'] # (N,)
    kappatheta = Theta_fill['kappatheta'] # (N,)
    sigma_sq_V = Theta_fill['sigma_sq_V'] # (N,)
    rho = Theta_fill['rho'] # (N,)

    # allocate
    V_gplus = V.copy()
    accepts = np.zeros_like(V) + np.nan # (N,T)
    T = len(Y)

    check_support_fn = lambda x: x > 0.0

    for parity in [0,1]: # ODD, EVEN

        t = np.arange(parity, T, 2) # 0,2,4,6,..., len = Npar
        p = t+1 # 1,3,5,7,...

        # do padding: ensure that first t=1 and last t=T periods have neighbors
        V_pad = np.concatenate([V_gplus[:, :1], V_gplus, V_gplus[:, -1:]], axis=1) # for each chain: V1,V1,V2,V3,...,VT-1,VT,VT. shape (N,T+2)
        Y_pad = np.concatenate([Y[:1], Y, Y[-1:]]) # shape (T+2,)

        Vt_g, Vtminus_g, Vtplus_g = V_pad[:, p], V_pad[:,p-1], V_pad[:,p+1] # use padding index by fancy indexing, shapes (N,Npar)
        Yt, Ytminus, Ytplus = Y_pad[p], Y_pad[p-1], Y_pad[p+1] # (N,)

        # construct booleans for inner periods
        has_minus = t > 0 # (Npar,)
        has_plus = t < T-1 # (Npar,)

        # compute number of neighbors
        n_nb = has_minus.astype(float) + has_plus.astype(float) # (Npar,)

        # construct proposal drawing function
        Vbar = (np.where(has_minus, Vtminus_g, 0.00) + np.where(has_plus, Vtplus_g, 0.00))/n_nb # (Npar,)
        draw_V_odd_prop = lambda: np.random.normal(loc=Vt_g, scale=2*np.sqrt((sigma_sq_V[:,None]*Vbar*Delta)/2)) # returns (Npar,)

        # forwards and backwards likelihoods
        fwd = lambda x: obs_ll_t(mu[:,None], kappa[:,None], kappatheta[:,None], sigma_sq_V[:,None], rho[:,None], Yt, Ytplus, x, Vtplus_g, Delta) \
                        + latent_ll_t(kappa[:,None], kappatheta[:,None], sigma_sq_V[:,None], x, Vtplus_g, Delta) # returns (Npar,)

        bwd = lambda x: obs_ll_t(mu[:,None], kappa[:,None], kappatheta[:,None], sigma_sq_V[:,None], rho[:,None], Ytminus, Yt, Vtminus_g, x, Delta) \
                        + latent_ll_t(kappa[:,None], kappatheta[:,None], sigma_sq_V[:,None], Vtminus_g, x, Delta) # returns (Npar,)

        # do masks such that bwd likelihood is zeroed out for t=1, and fwd is zeroed out for t=T
        post_logdensity = lambda x: np.where(has_plus, fwd(x), 0.00) + np.where(has_minus, bwd(x), 0.00)
        prop_logdensity = lambda x: 0.00

        Vt_gplus, accept_t = Metropolis_Hastings(Vt_g, draw_V_odd_prop, prop_logdensity, post_logdensity, check_support_fn)

        V_gplus[:,t] = Vt_gplus
        accepts[:,t] = accept_t

    return V_gplus, accepts