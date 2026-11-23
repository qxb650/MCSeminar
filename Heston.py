import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import least_squares
from types import SimpleNamespace

from cyclical_MCMC import draw_init_Theta_and_V

#####################
# SIMULATING HESTON #
#####################

def draw_dWs(rho, Delta, N=1):

    # draw normals
    N1 = np.random.normal(size=(N))
    N2 = np.random.normal(size=(N))

    # setup BMs
    dW1 = np.sqrt(Delta) * N1
    dW2 = rho * dW1 + np.sqrt(1-rho**2) * np.sqrt(Delta) * N2

    return dW1, dW2

def full_trunc_euler(mu, kappa, theta, sigma_sq_V, rho, Delta, N=1, T=1, burnin=0, do_print=False):

    if do_print:
        print(f'Feller is fulfilled: {2*kappa*theta > sigma_sq_V}')
        print(f'2 kappa theta = {2*kappa*theta:.5f}')
        print(f'sigma_V^2 = {sigma_sq_V:.5f}')

    # allocate
    V = np.zeros((T+burnin, N)) + np.nan
    S = np.zeros((T+burnin, N)) + np.nan
    Y = np.zeros((T+burnin, N)) + np.nan
    dY = np.zeros((T-1+burnin, N)) + np.nan

    # initialize logged stock price
    Yt = np.log(np.ones(N)) # (N,)

    # initial variance
    Vt = np.ones(N)*theta # (N,)

    for i in range(T+burnin):

        # draw Brownian motions
        dW1, dW2 = draw_dWs(rho, Delta, N=N)

        # simulate logged stock price
        Vtpos = np.maximum(Vt, 0.00)
        Yt = Yt + mu * Delta + np.sqrt(Vtpos) * dW1

        # simulate instantenous variance
        Vt = Vt + kappa * (theta - Vtpos) * Delta + np.sqrt(sigma_sq_V*Vtpos) * dW2

        # save
        V[i] = Vt
        Y[i] = Yt

        if i > 0:
            dY[i-1] = Y[i] - Y[i-1]

    return S[burnin:], V[burnin:], Y[burnin:], dY[burnin:]

#####################################
# SIMULATED METHOD OF MOMENTS (SMM) #
#####################################

def print_x(x, cost=None):
    
    mu = x[0]
    kappa = x[1]
    theta = x[2]
    sigma_sq_V = x[3]
    rho = x[4]

    print(f'mu = {mu:.5f}')
    print(f'kappa = {kappa:.5f}')
    print(f'theta = {theta:.5f}')
    print(f'sigma_sq_V = {sigma_sq_V:.5f}')
    print(f'rho = {rho:.5f}')

    if cost is not None: print(f'cost = {2*cost:.8f}\n')

def print_SMM(SMM_ns):

    names = ['mu', 'kappa', 'theta', 'sigma_sq_V', 'rho']
    z = 1.96 # 95% normal quantile

    print(f'{"":12s}{"estimate":>10s}{"s.e.":>10s}{"95% CI lower":>14s}{"95% CI upper":>14s}')
    for i, name in enumerate(names):
        est = SMM_ns.Theta_hat[i]
        se = SMM_ns.se_hat[i]
        print(f'{name:12s}{est:10.5f}{se:10.5f}{est - z*se:14.5f}{est + z*se:14.5f}')

def crit(
        x, seed, Lc, M_data,
        Delta, N, T, burnin, # sim
        K_y, L, K_rv, K_yrv # moments
        ):

    # unpack par
    mu = x[0]
    kappa = x[1]
    theta = x[2]
    sigma_sq_V = x[3]
    rho = x[4]

    # compute gs by simulation
    np.random.seed(seed)
    _, _, _, dY_sim = full_trunc_euler(mu, kappa, theta, sigma_sq_V, rho, Delta, N=N, T=T, burnin=burnin, do_print=False) # (T-1, N)
    M_Theta = moment_vec(dY_sim, K_y, L, K_rv, K_yrv)
    gs = (M_data-M_Theta)[:,None] # (1+K_y+L+K_rv,K_yrv,1)

    # compute objective for least squares minimization
    obj = (Lc.T @ gs).ravel()

    return obj

def SMM(
        dY,
        N_multi, bounds,
        lb, ub,
        K_y, L, K_rv, K_yrv,
        seed, Delta, N, T_SMM, burnin
        ):
    
    # compute data moments
    T = len(dY)+1
    M_data = moment_vec(dY, K_y, L, K_rv, K_yrv)

    # draw multistart guesses, (N_multi, 5)
    np.random.seed(seed)
    lens_roll = [10]*N_multi # not used
    V_bounds = [0, 0] # not used
    Theta_multi = draw_init_Theta_and_V(dY, bounds, lens_roll, V_bounds, N_multi, Delta, plot_Vrolls=False, for_SMM=True)

    # define objective for multistart
    W = compute_W(dY, K_y, L, K_rv, K_yrv)
    Lc = np.linalg.cholesky(W)
    f = lambda x: crit(x, seed, Lc, M_data, Delta, N, T_SMM, burnin, K_y, L, K_rv, K_yrv)

    # multi start
    par_list = []
    obj_list = []
    res_list = []
    for i_multi in range(N_multi):

        # unpack guess
        x0_i = Theta_multi[i_multi]

        print(f'{50*'--'}')
        print(f'Starting Multistart i = {i_multi} with guess:')
        print_x(x0_i)
    
        # minimize        
        res = least_squares(f, x0_i, bounds=(lb,ub), method='trf', diff_step=1e-2, x_scale=[0.05, 0.03, 1.0, 0.05, 0.5])

        # unpack result and save
        print(f'Ended after {res.nfev} iterations with status = {res.success}\nTheta:')
        print_x(res.x, cost=res.cost)
        print(f'{50*'--'}')
        par_list.append(res.x)
        obj_list.append(2*res.cost)
        res_list.append(res)

    # find minimizer
    Theta_hat = par_list[np.array(obj_list).argmin()]
    res = res_list[np.array(obj_list).argmin()]

    # compute variance-covariance
    Jac = res.jac # (#moments, 5)
    cov = (1/(T-1))*(1+(T-1)/((T_SMM-1)*N))*np.linalg.inv(Jac.T @ Jac)
    se = np.sqrt(np.diag(cov))

    # save
    SMM_ns = SimpleNamespace()
    SMM_ns.par_list = par_list
    SMM_ns.obj_list = obj_list
    SMM_ns.res_list = res_list
    SMM_ns.Theta_hat = Theta_hat
    SMM_ns.cov_hat = cov
    SMM_ns.se_hat = se

    return SMM_ns

########################################
# SMM: FUNCTIONS FOR COMPUTING MOMENTS #
########################################

# compute moments for data series
def moment_vec(x, K_y, L, K_rv, K_yrv):

    # compute realized variance
    rv = x**2

    # compute mean
    m  = [x.mean()]                                                  # E[y]
    
    # compute k-order moments
    m += [(np.abs(x)**k).mean() for k in range(1, K_y+1)]            # E|y|^k
    
    # compute autocorrelation moments
    m += [np.abs(x[i:]*x[:-i]).mean() for i in range(1, L+1)]        # E|y_t y_{t-i}|
    
    # compute autocorrelation moments in realized variance
    m += [(rv[i:]*rv[:-i]).mean() for i in range(1, K_rv+1)]         # E[y²_t y²_{t-i}]
    
    # compute cross-periods return and rv correlation
    m += [(x[:-i]*rv[i:]).mean() for i in range(1, K_yrv+1)]         # leverage E[y_t y²_{t+i}]

    return np.array(m)

def moment_labels(K_y, L, K_rv, K_yrv):
    return (["E[y]"]
          + [f"E|y|^{k}"            for k in range(1, K_y+1)]
          + [f"E|y_t y_t-{i}|"      for i in range(1, L+1)]
          + [f"E[y²_t y²_t-{i}]"    for i in range(1, K_rv+1)]
          + [f"E[y_t-{i}y²_t]"     for i in range(1, K_yrv+1)])

# compute daily contributions for moments for data series
def daily_moment_cont(x, K_y, L, K_rv, K_yrv):

    # compute realized variance
    rv = x**2

    # compute maximum number of lags possible
    Lmax = max(L, K_rv, K_yrv)

    # construct time indeces
    t = np.arange(Lmax, len(x))

    # compute mean, E[y]
    cols  = [x[t]]
    
    # compute k-order moments, E|y|^k
    cols += [(np.abs(x[t])**k) for k in range(1, K_y+1)]
    
    # compute autocorrelation moments, E|y_t y_{t-i}|
    cols += [np.abs(x[t]*x[t-i]) for i in range(1, L+1)]
    
    # compute autocorrelation moments in realized variance, # E[y²_t y²_{t-i}]
    cols += [(rv[t]*rv[t-i]) for i in range(1, K_rv+1)]
    
    # compute cross-periods return and rv correlation, E[y_t y²_{t+i}]
    cols += [(x[t-i]*rv[t]) for i in range(1, K_yrv+1)]

    # construct G, which has element g_(t,k), where t=Lmax,...,T and k = 0,1,2,...,#moments
    G_daily = np.column_stack(cols) # shape (T-Lmax-1, K_y+L+K_rv+K_yrv+1)

    return G_daily, Lmax

################################################
# SMM: FUNCTIONS FOR UPDATING WEIGHTING MATRIX #
################################################

def bartlett_weight(lag, lags):
    return 1-np.abs(lag)/(1+lags)

def newey_west(G, lags):
    
    TminusLmaxminusone = G.shape[0]
    n_moments = G.shape[1]

    # allocate
    S = np.zeros((n_moments, n_moments))

    for k in range(-lags, lags+1): # k = -lags, -lags+1, ..., 0, 1, ..., lags-1, lags

        if k >= 0:
            g_t = G[:TminusLmaxminusone-k] # (TminusLmaxminusone-k, #moments)
            g_t_plusk = G[k:] # (TminusLmaxminusone-k, #moments)

        else: # k is negative, k<0
            g_t = G[-k:] # drop first k obs, (TminusLmaxminusone+k, #moments)
            g_t_plusk = G[:TminusLmaxminusone+k] # drop last k objs, (TminusLmaxminusone+k, #moments)

        S += bartlett_weight(k, lags) * (g_t.T @ g_t_plusk / TminusLmaxminusone) # (#moments, #moments)

    return S

def compute_W(x, K_y, L, K_rv, K_yrv):

    # compute daily moment contributions
    G_daily, Lmax = daily_moment_cont(x, K_y, L, K_rv, K_yrv) # shape (TminusLmaxminusone, #moments)

    # compute deviation of daily contrubutions from data moments
    G_dev = G_daily - G_daily.mean(axis=0) # (TminusLmaxminusone, #moments)

    # compute Newey-West matrix
    W_inv = newey_west(G_dev, Lmax)
    W = np.linalg.inv(W_inv)

    # be sure to make W symmetric before Cholesky
    W = (W + W.T)/2

    return W