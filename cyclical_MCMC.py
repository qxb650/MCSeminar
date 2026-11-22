import numpy as np
import pandas as pd
from types import SimpleNamespace
import matplotlib.pyplot as plt

from Gibbs import mu_step, kappatheta_kappa_step
from Metropolis import sigma_V_step, rho_step, V_step, V_step_2blocks

def load_data(path, do_pct=True, do_print=True, VIX_path=None):

    # 1. load data
    SP500_df = pd.read_csv(path)

    # 2. drop nans in prices
    SP500_df = SP500_df.dropna(subset=['SP500'])

    # 3. compute log-prices
    SP500_df['log_prices'] = SP500_df['SP500'].apply(lambda x: np.log(x))

    # 4. compute log-returns
    SP500_df['log_returns'] = SP500_df['log_prices'].diff()

    # 5. set index corresponding to log-prices
    SP500_df['observation_date'] = pd.DatetimeIndex(SP500_df['observation_date'])
    SP500_df = SP500_df.set_index('observation_date')

    # 6. print dates
    if do_print: print(f'Data from {SP500_df.index[0]} to {SP500_df.index[-1]}')

    # 7. to numpy
    Y = SP500_df['log_prices'].to_numpy() # (T,)
    dY = SP500_df['log_returns'].to_numpy()[1:] # (T,)
    dY_pd = SP500_df['log_returns'][1:]
    Y_indx = SP500_df.index
    T = len(Y)

    if do_pct: dY_pd = 100*dY_pd
    if do_pct: dY = 100*dY
    if do_pct: Y = 100*Y

    if VIX_path is not None:

        VIX_df = pd.read_csv(VIX_path)

        # set index
        VIX_df['observation_date'] = pd.DatetimeIndex(VIX_df['observation_date'])
        VIX_df = VIX_df.set_index('observation_date')

        VIX_df = VIX_df.loc[SP500_df.index[0]:SP500_df.index[-1]]

        return Y, dY, dY_pd, Y_indx, T, VIX_df

    else:
        
        return Y, dY, dY_pd, Y_indx, T

def allocate_draws(N,T,G):

    draws = SimpleNamespace()

    draws.mu = np.zeros((N,G)) + np.nan
    draws.kappa = np.zeros((N,G)) + np.nan
    draws.kappatheta = np.zeros((N,G)) + np.nan
    draws.sigma_sq_V = np.zeros((N,G)) + np.nan
    draws.rho = np.zeros((N,G)) + np.nan
    draws.V = np.zeros((N,T,G)) + np.nan

    draws.sigma_sq_V_accepts = np.zeros((N,G)) + np.nan
    draws.rho_accepts = np.zeros((N,G)) + np.nan
    draws.V_accepts = np.zeros((N,T,G)) + np.nan

    return draws

################################
# FUNCTIONS FOR INITIAL VALUES #
################################

def draw_init_Theta_and_V(dY, bounds, lens_roll, bounds_facs_roll, N, Delta, plot_Vrolls=True, for_SMM=False):

    init_guess_Theta = {}

    # par
    if for_SMM: mu = np.random.uniform(bounds['mu_l'],bounds['mu_h'],size=(N,))
    kappa = np.random.uniform(bounds['kappa_l'],bounds['kappa_h'],size=(N,))
    theta = np.random.uniform(bounds['theta_l'],bounds['theta_h'],size=(N,))
    sigma_sq_V = np.random.uniform(bounds['sigma_sq_V_l'],bounds['sigma_sq_V_h'],size=(N,))
    rho = np.random.uniform(bounds['rho_l'],bounds['rho_h'],size=(N,))

    init_guess_Theta['kappa'] = kappa
    init_guess_Theta['theta'] = theta
    init_guess_Theta['sigma_sq_V'] = sigma_sq_V
    init_guess_Theta['rho'] = rho

    # V
    rv = dY**2
    T = len(dY)+1

    assert len(lens_roll) == N

    # compute rolling average of realized variances
    av_var = dY.var() / Delta
    min_rv = bounds_facs_roll[0]*av_var
    max_rv = bounds_facs_roll[1]*av_var

    rv_rolling_pd = [(pd.Series(rv).rolling(window=len_roll, center=True, min_periods=1).mean()).to_numpy() / Delta for len_roll in lens_roll]
    rv_rolling = np.array(np.clip(rv_rolling_pd, min_rv, max_rv)) # (N,T-1)

    # add VT-1 as VT. rvt is for Vt-Vt+1 but is based on Vt
    rv_rolling = np.concatenate([rv_rolling, rv_rolling[:, -1:]], axis=1) # (N,T-1) -> (N,T)

    if plot_Vrolls:
        f, ax = plt.subplots()
        for i in range(N): ax.plot(rv_rolling[i], label=f'{lens_roll[i]} window')
        ax.plot(np.ones(T)*av_var, label=r'Average variance')
        ax.grid(alpha=0.5)
        f.legend()

    if for_SMM:
        return np.array([mu, kappa, theta, sigma_sq_V, rho]).T

    else:
        return init_guess_Theta, rv_rolling

##########################################
# CYCLICAL METROPOLIS-HASTINGS ALGORITHM #
##########################################

def cyclical_MH(priors, init_guess_Theta, init_guess_V, Y, draws, G, Delta, print_iter=10, do_2blocks=False):

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
        if do_2blocks:
            V_gplus, accepts_V = V_step_2blocks(priors, Theta_g, Theta_fill, Y, V_g, Delta)
        else:
            V_gplus, accepts_V = V_step(priors, Theta_g, Theta_fill, Y, V_g, Delta)

        accepts_g['V'] = accepts_V

        return Theta_fill, V_gplus, accepts_g

    for g in range(G):

        if (g % print_iter == 0) and (g > 0):
            AR_sigma_sq_V = draws.sigma_sq_V_accepts[:,g-print_iter:g].mean()
            AR_rho = draws.rho_accepts[:,g-print_iter:g].mean()
            AR_V = draws.V_accepts[:,:,g-print_iter:g].mean()
            AR_V_min = draws.V_accepts[:,:,g-print_iter:g].mean(axis=(0,2)).min() # average over (N,G)-dimensions
            AR_V_max = draws.V_accepts[:,:,g-print_iter:g].mean(axis=(0,2)).max()
            print(f'g = {g}\t\tAR:[sigma_sq_V={100*AR_sigma_sq_V:.1f}%,\trho={100*AR_rho:.1f}%,\tV={100*AR_V:.1f}%,\tmin(V)={100*AR_V_min:.1f}%,\tmax(V)={100*AR_V_max:.1f}%]')

        if g == 0:
            Theta_g = init_guess_Theta
            V_g = init_guess_V

        # do cycle
        Theta_g, V_g, accepts_g = cycle(Theta_g, V_g)

        # save draws
        draws.mu[:,g] = Theta_g['mu']
        draws.kappa[:,g] = Theta_g['kappa']
        draws.kappatheta[:,g] = Theta_g['kappatheta']
        draws.sigma_sq_V[:,g] = Theta_g['sigma_sq_V']
        draws.rho[:,g] = Theta_g['rho']
        draws.V[:,:,g] = V_g

        # save accepts
        draws.sigma_sq_V_accepts[:,g] = accepts_g['sigma_sq_V']
        draws.rho_accepts[:,g] = accepts_g['rho']
        draws.V_accepts[:,:,g] = accepts_g['V']

#######################
# LOAD AND SAVE DRAWS #
#######################

def save_fn(ns, filename):
    np.savez(filename + '.npz', **ns.__dict__)

def load_fn(filename):
    with np.load(filename + '.npz') as f:
        return SimpleNamespace(**{k: f[k] for k in f.files})