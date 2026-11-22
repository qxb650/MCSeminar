# from bayesian_regression import mu_regression, kappa_kappatheta_regression, sigma_sq_V_regression

# def mu_step(priors, Theta_fill, V):

#     # unpack other par
#     kappa = Theta_fill['kappa']
#     kappatheta = Theta_fill['kappatheta']
#     sigma_sq_V = Theta_fill['sigma_sq_V']
#     rho = Theta_fill['rho']

#     # do bayesian regression to obtain posterior density
#     mu_post, sigma_sq_post = mu_regression(priors, kappa, kappatheta, sigma_sq_V, rho, Y, V, Delta)

#     # draw normal
#     mu_new = np.random.normal(loc=mu_post, scale=np.sqrt(sigma_sq_post))

#     return mu_new

# def kappatheta_kappa_step(priors, Theta_fill, V):

#     # unpack other par
#     mu = Theta_fill['mu']
#     sigma_sq_V = Theta_fill['sigma_sq_V']
#     rho = Theta_fill['rho']

#     # do bayesian regression to obtain posterior density
#     Mu_post, Omega_post = kappa_kappatheta_regression(priors, mu, sigma_sq_V, rho, Y, V, Delta)

#     # draw normal
#     draws = np.random.multivariate_normal(mean=Mu_post.flatten(), cov=Omega_post)
#     kappatheta, kappa = draws[0].item(), draws[1].item()

#     return kappatheta, kappa

from Metropolis import sigma_V_step, rho_step, V_step
from Gibbs import mu_step, kappatheta_kappa_step

# def sigma_V_step(priors, Theta, Theta_fill, V):

#     # unpack prior and create prior density function
#     alpha_sigma_sq_V_prior = priors['alpha_sigma_sq_V']
#     beta_sigma_sq_V_prior = priors['beta_sigma_sq_V']
#     prior_density = lambda x: stats.invgamma.logpdf(x, alpha_sigma_sq_V_prior, scale=beta_sigma_sq_V_prior)

#     # unpack other par
#     mu = Theta_fill['mu']
#     kappa = Theta_fill['kappa']
#     kappatheta = Theta_fill['kappatheta']
#     rho = Theta_fill['rho']

#     # do bayesian regression to obtain proposal draw and define proposal density
#     alpha_sigma_sq_V_prop, beta_sigma_sq_V_prop = sigma_sq_V_regression(priors, kappa, kappatheta, V, Delta)
#     sigma_sq_V_proposal = stats.invgamma.rvs(alpha_sigma_sq_V_prop, scale=beta_sigma_sq_V_prop)
#     proposal_density = lambda x: stats.invgamma.logpdf(x, alpha_sigma_sq_V_prop, scale=beta_sigma_sq_V_prop)

#     # unpack old MH-draw
#     sigma_sq_V_g = Theta['sigma_sq_V']

#     # do MH-step
#     # posterior log density for gth draw
#     obs_ll_g = obs_ll(mu, kappa, kappatheta, sigma_sq_V_g, rho, Y, V, Delta)
#     latent_ll_g = latent_ll(kappa, kappatheta, sigma_sq_V_g, V, Delta)
#     prior_g = prior_density(sigma_sq_V_g)
#     post_g = obs_ll_g + latent_ll_g + prior_g

#     # posterior log density for proposal draw
#     obs_ll_proposal = obs_ll(mu, kappa, kappatheta, sigma_sq_V_proposal, rho, Y, V, Delta)
#     latent_ll_proposal = latent_ll(kappa, kappatheta, sigma_sq_V_proposal, V, Delta)
#     prior_proposal = prior_density(sigma_sq_V_proposal)
#     post_proposal = obs_ll_proposal + latent_ll_proposal + prior_proposal

#     # proposal density for gth draw
#     proposal_dens_g = proposal_density(sigma_sq_V_g)

#     # proposal density for proposal
#     proposal_dens_proposal = proposal_density(sigma_sq_V_proposal)

#     # compute log(alpha)
#     log_alpha = post_proposal - post_g + proposal_dens_g - proposal_dens_proposal

#     # accept if log(alpha) > log(u)
#     u = np.random.uniform()
#     logu = np.log(u)

#     if log_alpha > logu:
#         sigma_sq_V_gplus = sigma_sq_V_proposal

#     else:
#         sigma_sq_V_gplus = sigma_sq_V_g

#     return sigma_sq_V_gplus

# def rho_step(priors, Theta, Theta_fill, V):

#     # unpack other par
#     mu = Theta_fill['mu']
#     kappa = Theta_fill['kappa']
#     kappatheta = Theta_fill['kappatheta']
#     sigma_sq_V = Theta_fill['sigma_sq_V']

#     # obtain proposal
#     Yt = np.array(SP500_df['log_prices'])[:-1] # T-1: Y1,Y2,...,YT-1
#     Ytplus = np.array(SP500_df['log_prices'].shift(-1))[:-1] # T-1: Y2,Y3,...,YT
#     Vt = V[:-1] # T-1: V1,V2,...,VT-1
#     Vtplus = V[1:] # T-1: V2,V3,...,VT
#     eps_Vt = eps_V_t(kappa, kappatheta, sigma_sq_V, Vt, Vtplus, Delta)
#     eps_Yt = eps_Y_t(mu, Yt, Ytplus, Vt, Delta)

#     rho_mean = np.corrcoef(eps_Vt, eps_Yt)[0,1].item()
#     rho_std = ((1-rho_mean**2)/np.sqrt(T-1)).item()

#     rho_proposal = np.random.normal(loc=rho_mean, scale=rho_std)

#     # reject all proposals where rho is not in [-1,1]
#     if np.abs(rho_proposal) > 1.00:
#         return rho_g

#     # unpack old MH-draw
#     rho_g = Theta['rho']

#     # do MH-step, prior is uninformative
#     # posterior log density for gth draw, latent likelihood is unaffected by rho
#     obs_ll_g = obs_ll(mu, kappa, kappatheta, sigma_sq_V, rho_g, Y, V, Delta)
#     post_g = obs_ll_g

#     # posterior log density for proposal draw
#     obs_ll_proposal = obs_ll(mu, kappa, kappatheta, sigma_sq_V, rho_proposal, Y, V, Delta)
#     post_proposal = obs_ll_proposal

#     # proposal density for gth draw
#     proposal_dens_g = stats.norm.logpdf(rho_g, loc=rho_mean, scale=rho_std)

#     # proposal density for proposal draw
#     proposal_dens_proposal = stats.norm.logpdf(rho_proposal, loc=rho_mean, scale=rho_std)

#     # compute log(alpha)
#     log_alpha = post_proposal - post_g + proposal_dens_g - proposal_dens_proposal

#     # accept if log(alpha) > log(u)
#     u = np.random.uniform()
#     logu = np.log(u)

#     if log_alpha > logu:
#         rho_gplus = rho_proposal

#     else:
#         rho_gplus = rho_g

#     return rho_gplus


# def V_step(priors, Theta, Theta_fill, V):

#     # par
#     mu = Theta_fill['mu']
#     kappa = Theta_fill['kappa']
#     kappatheta = Theta_fill['kappatheta']
#     sigma_sq_V = Theta_fill['sigma_sq_V']
#     rho = Theta_fill['rho']

#     # t = 1: p(Vt|Vt,Vt+1,Yt,Yt+1,Theta) = p(Yt+1-Yt|Vt,Vt+1,Theta) * p(Vt+1-Vt|Theta)
#     # t = 2,3,...,T-1: p(Vt|Vt-1,Vt+1,Yt-1,Yt,Yt+1,Theta) = p(Yt+1-Y_t|Vt,Vt+1,Theta) * p(Yt-Yt-1|Vt-1,Vt,Theta) * p(Vt+1-Vt|Theta) * p(Vt-Vt-1|Theta)
#     # t = T: p(Vt|Vt-1,Yt-1,Yt,Theta) = p(Yt-Yt-1|Vt-1,Vt,Theta) * p(Vt-Vt-1|Theta)
    
#     # allocate
#     V_gplus = np.zeros_like(V) + np.nan # T
    
#     for t in range(0,T):

#         if t == 0:
#             Vbar = V[t+1]
        
#         elif t == T-1:
#             Vbar = V[t-1]

#         else:
#             Vbar = 0.5*(V[t-1]+V[t+1])

#         Vt_g = V[t]
#         rw_shock_t = np.random.normal(loc=0.0, scale=np.sqrt((sigma_sq_V*Vbar*Delta)/2))
#         Vt_proposal = Vt_g + rw_shock_t

#         # reject all proposal with negative variance
#         if Vt_proposal < 0:
#             V_gplus[t] = Vt_g
#             continue

#         # for t = 1 and t = 1,2,...,T-1 -> can always unpack next-period variance
#         if t<T-1:

#             Vtplus_g = V[t+1]

#             # g likelihood in period t
#             obs_llt_g = obs_ll_t(mu, kappa, kappatheta, sigma_sq_V, rho, Y[t], Y[t+1], Vt_g, Vtplus_g, Delta)
#             latent_llt_g = latent_ll_t(kappa, kappatheta, sigma_sq_V, Vt_g, Vtplus_g, Delta)

#             # proposal likelihood in period t
#             obs_llt_proposal = obs_ll_t(mu, kappa, kappatheta, sigma_sq_V, rho, Y[t], Y[t+1], Vt_proposal, Vtplus_g, Delta)
#             latent_llt_proposal = latent_ll_t(kappa, kappatheta, sigma_sq_V, Vt_proposal, Vtplus_g, Delta)

#             # t = 1 -> can't unpack former period variance -> zero out ll for t-1
#             if t == 0:
#                 obs_lltmimus_g, obs_lltminus_proposal = 0.0, 0.0
#                 latent_lltminus_g, latent_lltminus_proposal = 0.0, 0.0
        
#         # for t = 1,2,...,T-1 and t = T -> can always unpack former period variance
#         if t>0: 

#             Vtminus_g = V_gplus[t-1]

#             # g likelihood in period t-1
#             obs_lltmimus_g = obs_ll_t(mu, kappa, kappatheta, sigma_sq_V, rho, Y[t-1], Y[t], Vtminus_g, Vt_g, Delta)
#             latent_lltminus_g = latent_ll_t(kappa, kappatheta, sigma_sq_V, Vtminus_g, Vt_g, Delta)

#             # proposal likelihood in period t-1
#             obs_lltminus_proposal = obs_ll_t(mu, kappa, kappatheta, sigma_sq_V, rho, Y[t-1], Y[t], Vtminus_g, Vt_proposal, Delta)
#             latent_lltminus_proposal = latent_ll_t(kappa, kappatheta, sigma_sq_V, Vtminus_g, Vt_proposal, Delta)

#             # t = T -> can't unpack next period variance -> zero out ll for t
#             if t == T-1:
#                 obs_llt_g, obs_llt_proposal = 0.0, 0.0
#                 latent_llt_g, latent_llt_proposal = 0.0, 0.0

#         # compute posterior ll for both g and proposal
#         post_g = obs_llt_g + latent_llt_g + obs_lltmimus_g + latent_lltminus_g
#         post_proposal = obs_llt_proposal + latent_llt_proposal + obs_lltminus_proposal + latent_lltminus_proposal

#         log_alpha = post_proposal - post_g

#         # accept if log(alpha) > log(u)
#         u = np.random.uniform()
#         logu = np.log(u)

#         if log_alpha > logu:
#             Vt_gplus = Vt_proposal

#         else:
#             Vt_gplus = Vt_g

#         V_gplus[t] = Vt_gplus

#     return V_gplus

# def drop_nan_draws(d):

#     # infer number of draws by taking shape in last dim of first dict element
#     n_draws = next(iter(d.values())).shape[-1]

#     # boolean for which elements to choose
#     valid = np.ones(n_draws, dtype=bool)

#     # loop over elements in dict
#     for v in d.values():

#         # find nans
#         bad = np.isnan(v)

#         # if it is V or other with 2D
#         if v.ndim > 1:
#             bad = bad.any(axis=tuple(range(v.ndim - 1)))
#         valid &= ~bad

#     return {k: v[..., valid] for k, v in d.items()}

# draws = drop_nan_draws(draws)
# accepts = drop_nan_draws(accepts)

# mu_draws_burnedin = draws['mu_draws'][int(0.5*len(draws['mu_draws'])):]

# m = 5
# n = len(mu_draws_burnedin) // m

# assert len(mu_draws_burnedin) % m == 0

# mu_draws_burnedin_ms = np.array(np.split(mu_draws_burnedin, [i*n for i in range(1,m)])) # list which is m long
# # mu_draws_burnin_ms_mean = np.array([arr.mean() for arr in mu_draws_burnedin_ms]) # (m,)
# # B = (n/(m-1))*np.sum((mu_draws_burnin_ms_mean - np.mean(mu_draws_burnedin))**2) # (m,) - scalar -> sum -> scalar
# # W = np.mean(np.sum((1/(n-1))*((mu_draws_burnedin_ms) - mu_draws_burnin_ms_mean[:,None])**2, axis=1))

# # var_mu = ((n-1)/n)*W + (1/n)*B

# # f, ax = plt.subplots()
# # for i in range(m):
# #     i_lo = i*n
# #     i_hi = (i+1)*n
# #     mu_draws_burnedin_m = mu_draws_burnedin[i_lo:i_hi]
# #     ax.plot(mu_draws_burnedin_m, label=str(i))

# # f.legend()

# Rhat_ess(mu_draws_burnedin_ms)

# # parameters defined in CIR paper used to derive ergodic Gamma dist. for volatility
# omega_CIR = (2*kappa)/(sigma_V**2)
# nu_CIR = (2*kappa*theta)/(sigma_V**2)

# shape_Gamma = nu_CIR
# scale_Gamma = 1/omega_CIR

# # plot
# f, ax = plt.subplots()
# x = np.linspace(0,0.5,1000)
# ax.plot(x,stats.gamma.pdf(x, a=shape_Gamma, scale=scale_Gamma), label='pdf')
# ax.vlines(shape_Gamma*scale_Gamma, 0.00, 20.0, ls='--', color='red', label='mean')
# ax.set_xlabel('V')
# ax.set_ylabel('Density')
# ax.legend()

# # draw volatility from ergodic distribution
# V_draws = np.random.gamma(shape=shape_Gamma, scale=scale_Gamma, size=(N))

# # draw brownian motions
# dW1s = np.sqrt(dt) * np.random.normal(size=(N))

# # draw Xs
# dX_draws = (mu-0.5*V_draws)*dt + np.sqrt(V_draws)*dW1s

    # # draw from ergodic distributoon
    # else:

    #     # CIR parameters
    #     omega_CIR = (2*kappa)/(sigma_sq_V)
    #     nu_CIR = (2*kappa*theta)/(sigma_sq_V)

    #     # Gamma dist parameters
    #     shape_Gamma = nu_CIR
    #     scale_Gamma = 1/omega_CIR

    #     Vt = np.random.gamma(shape=shape_Gamma, scale=scale_Gamma, size=(N)) # (N,)
# # res = minimize(f, x0, bounds=Bounds, method='Nelder-Mead')

# # guess on parameters
# mu = dY.mean()
# theta = dY.var()
# kappa = 0.0231 # from Eraker
# sigma_sq_V = 0.1434**2 # from Eraker
# rho = -0.3974 # from Eraker

# def compute_gs(seed, mu, kappa, theta, sigma_sq_V, rho):

#     np.random.seed(seed)
#     _, _, _, dY_sim = full_trunc_euler(mu, kappa, theta, sigma_sq_V, rho, Delta, N=N, T=T, burnin=burnin, do_print=False) # (T-1, N)

#     M_Theta = moment_vec_(dY_sim)

#     return M_data - M_Theta

# def compute_crit(seed, W, mu, kappa, theta, sigma_sq_V, rho, do_ls=True):

#     gs = compute_gs(seed, mu, kappa, theta, sigma_sq_V, rho)[:,None] # (K_y+L+K_rv,K_yrv,1)

#     if do_ls:

#         # do Cholesky decomposition of weighting matrix
#         Lc = np.linalg.cholesky(W)

#         # return (K_y+L+K_rv+K_yrv,)
#         return (Lc.T @ gs).ravel()

#     else:
#         # return scalar
#         return (gs.T @ W @ gs).item()

# def f_outer(seed, W, x):

#     mu = x[0]
#     kappa = x[1]
#     theta = x[2]
#     sigma_sq_V = x[3]
#     rho = x[4]

#     obj = compute_crit(seed, W, mu, kappa, theta, sigma_sq_V, rho)

#     print(f'mu = {mu:.5f}')
#     print(f'kappa = {kappa:.5f}')
#     print(f'theta = {theta:.5f}')
#     print(f'sigma_sq_V = {sigma_sq_V:.5f}')
#     print(f'rho = {rho:.5f}')
#     print(f'obj = {np.abs(obj).mean():.5f}', '\n')

#     return obj

# def SMM(dY, N_multi, iter, seed, W_start, lb, ub):

#     # compute data moments

#     # (N_multi, 5)
#     Theta_multi = draw_init_Theta_and_V(dY, bounds, lens_roll, [0, 0], N_multi, Delta, plot_Vrolls=False, for_SMM=True)

#     par_list = []
#     obj_list = []

#     # define objective
#     f = lambda x: f_outer(seed, W_start, x)

#     for i_multi in range(N_multi):

#         # unpack guess
#         x0_i = Theta_multi[i_multi]

#         print(f'Starting SMM with guess: {x0_i}')
#         print(f'{50*'--'}')
    
#         # minimize        
#         res = least_squares(f, x0_i, bounds=(lb,ub), method='trf', diff_step=1e-2, x_scale=[0.05, 0.03, 1.0, 0.05, 0.5])

#         # unpack result and save
#         par_list.append(res.x)
#         obj_list.append(2*res.cost)

#     # find minimzer
#     Theta_init = par_list[np.array(obj_list).argmin()]

#     # allocate
#     W_list = []
#     res_list = []

#     # update W and minimize again
#     for i in range(iter):

#         print(f'\nStarting SMM with new W: i = {i}')
#         print(f'{50*'--'}')

#         if i == 0: Theta = Theta_init

#         new_W = update_W(dY, Theta[0], Theta[1], Theta[2], Theta[3], Theta[4], Delta, T, N, burnin, K_y, L, K_rv, K_yrv)

#         f = lambda x: f_outer(seed, new_W, x)

#         res = least_squares(f, Theta, bounds=(lb,ub), method='trf', diff_step=1e-2, x_scale=[0.05, 0.03, 1.0, 0.05, 0.5])

#         Theta = res.x

#         print(f'Converged after {res.nfev} iterations')

#         W_list.append(new_W)
#         res_list.append(res)

#     return W_list, res_list