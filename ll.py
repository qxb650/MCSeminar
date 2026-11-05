import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

#######################
# IMPLIED INNOVATIONS #
#######################

def eps_V_t(kappa, kappatheta, sigma_sq_V, Vt, Vtplus, Delta):

    num = Vtplus - Vt - kappatheta*Delta + kappa*Vt*Delta
    denom = np.sqrt(sigma_sq_V*Vt * Delta)

    return num/denom

def eps_Y_t(mu, Yt, Ytplus, Vt, Delta):

    num = Ytplus - Yt - mu*Delta + 0.5*Vt*Delta
    denom = np.sqrt(Vt*Delta)

    return num/denom

############################
# PERIOD T LOG-LIKELIHOODs #
############################

def obs_ll_t(mu, kappa, kappatheta, sigma_sq_V, rho, Yt, Ytplus, Vt, Vtplus, Delta):

    eps_V_t_ = eps_V_t(kappa, kappatheta, sigma_sq_V, Vt, Vtplus, Delta)

    fac = -0.5*np.log(2*np.pi*Vt*(1-rho**2)*Delta)

    exp_num = Ytplus - Yt - mu*Delta + 0.5*Vt*Delta - rho*np.sqrt(Vt*Delta)*eps_V_t_
    exp_denom = 2*Vt*Delta*(1-rho**2)

    log_dens = fac - (exp_num**2)/exp_denom

    return log_dens

def latent_ll_t(kappa, kappatheta, sigma_sq_V, Vt, Vtplus, Delta):

    fac = -0.5*np.log(2*np.pi*sigma_sq_V*Vt*Delta)

    exp_num = Vtplus - Vt - kappatheta*Delta+kappa*Vt*Delta
    exp_denom = 2*(sigma_sq_V)*Vt*Delta

    log_dens = fac - (exp_num**2)/exp_denom

    return log_dens

#################################
# LOG-LIKELIHOOD ACROSS PERIODS #
#################################

def obs_ll(mu, kappa, kappatheta, sigma_sq_V, rho, Y, V, Delta):
    
    Y_noT = Y[:-1] # T-1: Y1,Y2,...,YT-1
    Y_leaded = Y[1:] # T-1: Y2,Y3,...,YT
    V_noT = V[:-1] # T-1: V1,V2,...,VT-1
    V_leaded = V[1:] # T-1: V2,V3,...,VT

    obs_loglikelihoods = obs_ll_t(mu, kappa, kappatheta, sigma_sq_V, rho, Y_noT, Y_leaded, V_noT, V_leaded, Delta)

    return np.sum(obs_loglikelihoods)

def latent_ll(kappa, kappatheta, sigma_sq_V, V, Delta):

    V_noT = V[:-1] # T-1: V1,V2,...,VT-1
    V_leaded = V[1:] # T-1: V2,V3,...,VT

    latent_loglikelihoods = latent_ll_t(kappa, kappatheta, sigma_sq_V, V_noT, V_leaded, Delta)

    return np.sum(latent_loglikelihoods)