import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import time
import jax
import jax.numpy as jnp

from Gibbs_jax import mu_step_jax, kappatheta_kappa_step_jax
from Metropolis_jax import sigma_V_step_jax, rho_step_jax, V_step_jax

def make_cycle(priors, Y, Delta):

    def cycle(state, key):

        # unpack state
        Theta, V = state

        # construct keys
        key_mu, key_kappa_kappatheta, key_sigma_sq_V, key_rho, key_V = jax.random.split(key, 5)

        Theta_fill = dict(Theta)
        Theta_fill['mu'] = mu_step_jax(priors, Theta_fill, Y, V, Delta, key_mu)
        Theta_fill['kappatheta'], Theta_fill['kappa'] = kappatheta_kappa_step_jax(priors, Theta_fill, Y, V, Delta, key_kappa_kappatheta)
        Theta_fill['sigma_sq_V'], acc_sigma_sq_V = sigma_V_step_jax(priors, Theta, Theta_fill, Y, V, Delta, key_sigma_sq_V)
        Theta_fill['rho'], acc_rho = rho_step_jax(priors, Theta, Theta_fill, Y, V, Delta, key_rho)
        V_new, acc_V = V_step_jax(priors, Theta, Theta_fill, Y, V, Delta, key_V)

        out = dict(Theta_fill); out.update(acc_sigma_sq_V=acc_sigma_sq_V, acc_rho=acc_rho, acc_V_mean=acc_V.mean())
        
        return (Theta_fill, V_new), out
    
    
    return cycle

def cyclical_MCMC_jax(priors, init_guess_Theta, init_guess_V, Y, G, Delta, key, chunk=100):

    Y = jnp.asarray(Y, dtype=jnp.float64)
    cycle = make_cycle(priors, Y, Delta)

    @jax.jit
    def run_chunk(state, key):
        return jax.lax.scan(cycle, state, jax.random.split(key, chunk))
    
    state = ({k: jnp.asarray(v, dtype=jnp.float64) for k, v in init_guess_Theta.items()},
         jnp.asarray(init_guess_V, dtype=jnp.float64))
    
    #state = (init_guess_Theta, init_guess_V)
    outs, Vs = [], []

    for c, key_chunk in enumerate(jax.random.split(key, G // chunk)):
        print(f'g = {c*chunk}')
        t0 = time.perf_counter()
        state, out = run_chunk(state, key_chunk)
        outs.append(jax.tree_util.tree_map(np.asarray, out))    # parametre + acceptrater, hver iteration
        Vs.append(np.asarray(state[1]))                         # V kun ved slutningen af hver bid
        print(f'chunk {c}: {time.perf_counter()-t0:.2f}s  cache={run_chunk._cache_size()}')


    draws = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
    draws['V_draws'] = np.stack(Vs, axis=1)                     # (T, G/chunk)
    return draws


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

    lo, hi = np.nanpercentile(V, [0.5, 99.5], axis=1)

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