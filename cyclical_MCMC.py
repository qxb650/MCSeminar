import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

from Gibbs import mu_step, kappatheta_kappa_step
from Metropolis import sigma_V_step, rho_step, V_step


def cyclical_MCMC(priors, init_guess_Theta, init_guess_V, Y, draws, G, Delta):

    def cycle(Theta_g, V_g):

        # allocate
        Theta_fill = Theta_g.copy()

        # mu
        mu_gplus = mu_step(priors, Theta_fill, Y, V_g, Delta)
        Theta_fill['mu'] = mu_gplus

        # kappa, kappatheta
        kappatheta_gplus, kappa_gplus = kappatheta_kappa_step(priors, Theta_fill, Y, V_g, Delta)
        Theta_fill['kappatheta'] = kappatheta_gplus
        Theta_fill['kappa'] = kappa_gplus

        # sigma_sq_V
        sigma_sq_V_gplus = sigma_V_step(priors, Theta_g, Theta_fill, Y, V_g, Delta)
        Theta_fill['sigma_sq_V'] = sigma_sq_V_gplus

        # rho
        rho_gplus = rho_step(priors, Theta_g, Theta_fill, Y, V_g, Delta)
        Theta_fill['rho'] = rho_gplus

        # V
        V_gplus = V_step(priors, Theta_g, Theta_fill, Y, V_g, Delta)

        return Theta_fill, V_gplus


    for g in range(G):

        if (g % 10 == 0): print(f'g = {g}')

        if g == 0:
            Theta_g = init_guess_Theta
            V_g = init_guess_V

        Theta_g, V_g = cycle(Theta_g, V_g)

        draws['mu_draws'][g] = Theta_g['mu']
        draws['kappa_draws'][g] = Theta_g['kappa']
        draws['kappatheta_draws'][g] = Theta_g['kappatheta']
        draws['sigma_sq_V_draws'][g] = Theta_g['sigma_sq_V']
        draws['rho_draws'][g] = Theta_g['rho']
        draws['V_draws'][:,g] = V_g

def plot_distributions(draws, G, burnin=0, bins=100):

    f, ax = plt.subplots(3, 2, figsize=(12, 8))

    ax[0,0].hist(draws['mu_draws'][burnin:], bins=bins)
    ax[0,0].set_title(r'$\mu$ draws')

    ax[0,1].hist(draws['kappa_draws'][burnin:], bins=bins)
    ax[0,1].set_title(r'$\kappa$ draws')

    ax[1,0].hist(draws['kappatheta_draws'][burnin:], bins=bins)
    ax[1,0].set_title(r'$\kappa \theta$ draws')

    ax[1,1].hist(draws['sigma_sq_V_draws'][burnin:], bins=bins)
    ax[1,1].set_title(r'$\sigma^2_V$ draws')

    ax[2,0].hist(draws['rho_draws'][burnin:], bins=bins)
    ax[2,0].set_title(r'$\rho$ draws')

    ax[2,1].hist(draws['V_draws'][:, burnin:].flatten(), bins=bins)
    ax[2,1].set_title(r'$V$ draws')

    plt.tight_layout()

def plot_Vts(draws, periods, burnin=0):

    # periods should be a list of integers corresponding to the periods to plot
    # function should infer rows and cols from the number of periods provided in the list periods

    f, ax = plt.subplots(1, len(periods), figsize=(12, 8))

    for t in periods:
        ax[periods.index(t)].hist(draws['V_draws'][t, burnin:], bins=100)
        ax[periods.index(t)].set_title(rf'$V_{t}$ draws')
    
    
def plot_VY(draws, Y, burnin=0):

    f, ax = plt.subplots(2, 1, figsize=(12, 8))

    ax[0].plot(np.nanmean(draws['V_draws'][:, burnin:], axis=1))
    ax[0].set_title(r'$\mathbb{E}[V_t]$ draws')

    ax[1].plot(Y[1:] - Y[:-1])
    ax[1].set_title(r'$dY_t$')

    plt.tight_layout()
    plt.xlabel('t')
    plt.show()