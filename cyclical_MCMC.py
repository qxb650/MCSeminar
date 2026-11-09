import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

from Gibbs import mu_step, kappatheta_kappa_step
from Metropolis import sigma_V_step, rho_step, V_step


def cyclical_MCMC(priors, init_guess_Theta, init_guess_V, Y, draws, accepts, G, Delta):

    def cycle(Theta_g, V_g):

        # allocate
        Theta_fill = Theta_g.copy()
        accepts_g = {}

        # mu
        mu_gplus = mu_step(priors, Theta_fill, Y, V_g, Delta)
        Theta_fill['mu'] = mu_gplus

        # kappa, kappatheta
        kappatheta_gplus, kappa_gplus = kappatheta_kappa_step(priors, Theta_fill, Y, V_g, Delta)
        Theta_fill['kappatheta'] = kappatheta_gplus
        Theta_fill['kappa'] = kappa_gplus

        # sigma_sq_V
        sigma_sq_V_gplus, sigma_sq_V_accept = sigma_V_step(priors, Theta_g, Theta_fill, Y, V_g, Delta)
        Theta_fill['sigma_sq_V'] = sigma_sq_V_gplus
        accepts_g['sigma_sq_V'] = sigma_sq_V_accept

        # rho
        rho_gplus, rho_accept = rho_step(priors, Theta_g, Theta_fill, Y, V_g, Delta)
        Theta_fill['rho'] = rho_gplus
        accepts_g['rho'] = rho_accept

        # V
        V_gplus, accepts_V = V_step(priors, Theta_g, Theta_fill, Y, V_g, Delta)
        accepts_g['V'] = accepts_V

        return Theta_fill, V_gplus, accepts_g


    for g in range(G):

        if (g % 10 == 0): print(f'g = {g}')

        if g == 0:
            Theta_g = init_guess_Theta
            V_g = init_guess_V

        Theta_g, V_g, accepts_g = cycle(Theta_g, V_g)

        draws['mu_draws'][g] = Theta_g['mu']
        draws['kappa_draws'][g] = Theta_g['kappa']
        draws['kappatheta_draws'][g] = Theta_g['kappatheta']
        draws['sigma_sq_V_draws'][g] = Theta_g['sigma_sq_V']
        draws['rho_draws'][g] = Theta_g['rho']
        draws['V_draws'][:,g] = V_g

        accepts['sigma_sq_V'][g] = accepts_g['sigma_sq_V']
        accepts['rho'][g] = accepts_g['rho']
        accepts['V'][:,g] = accepts_g['V']

def plot_distributions(draws, G, burnin=0, bins=100):

    f, ax = plt.subplots(3, 2, figsize=(12, 8))

    ax[0,0].hist(draws['mu_draws'][burnin:], bins=bins)
    ax[0,0].set_title(r'$\mu$ draws')

    ax[0,1].hist(draws['kappa_draws'][burnin:], bins=bins)
    ax[0,1].set_title(r'$\kappa$ draws')

    theta_draws = draws['kappatheta_draws'] / draws['kappa_draws']
    ax[1,0].hist(theta_draws[burnin:], bins=bins)
    ax[1,0].set_title(r'$\theta$ draws')

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
    
    
def plot_VY(draws, Y, index, burnin=0):

    f, ax = plt.subplots(2, 1, figsize=(12, 8))

    V = draws['V_draws'][:, burnin:]
    t = np.arange(V.shape[0])

    lo, hi = np.nanpercentile(V, [2.5, 97.5], axis=1)

    ax[0].plot(index, np.nanmean(V, axis=1), label=r'$\mathbb{E}[V_t]$')
    ax[0].fill_between(index, lo, hi, alpha=0.3, label='95% interval')
    ax[0].grid()

    ax[0].set_title(r'$\mathbb{E}[V_t]$')
    ax[0].legend()

    ax[1].plot(index[1:], Y[1:] - Y[:-1])
    ax[1].set_title(r'SP500 log-returns: $dY_t$')
    ax[1].grid()

    for i in range(2):
        plt.setp(ax[i].get_xticklabels(), rotation=45, ha='right')
        ax[i].grid(alpha=0.5)

    f.tight_layout()