import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from types import SimpleNamespace

def plot_distributions(draws, G, burnin=0, bins=100, xlim_dict={}, bins_=[100,100,100,100,100,100,100]):

    f, ax = plt.subplots(2, 4, figsize=(20, 8))

    ax[0,0].hist(draws.mu[:,burnin:].flatten(), bins=bins_[0], density=True)
    ax[0,0].set_title(r'$\mu$ draws')

    ax[0,1].hist(draws.kappa[:,burnin:].flatten(), bins=bins_[1], density=True)
    ax[0,1].set_title(r'$\kappa$ draws')

    theta_draws = draws.kappatheta / draws.kappa
    ax[0,2].hist(theta_draws[:,burnin:].flatten(), bins=bins_[2], density=True)
    ax[0,2].set_title(r'$\theta$ draws')

    ax[0,3].hist(draws.sigma_sq_V[:,burnin:].flatten(), bins=bins_[3], density=True)
    ax[0,3].set_title(r'$\sigma^2_V$ draws')

    ax[1,0].hist(draws.rho[:,burnin:].flatten(), bins=bins_[4], density=True)
    ax[1,0].set_title(r'$\rho$ draws')

    ax[1,1].hist(draws.V[:, :, burnin:].flatten(), bins=bins_[5], density=True)
    ax[1,1].set_title(r'Pooled $V$')

    ax[1,2].hist(draws.V[:, 882, burnin:].flatten(), bins=bins_[6], density=True)
    ax[1,2].set_title(r'Variance March 19 2020: $V_{882}$')
    
    ax[1,3].hist(draws.V[:, 0, burnin:].flatten(), bins=bins_[7], density=True, alpha=0.5, label='First period')
    ax[1,3].hist(draws.V[:, -1, burnin:].flatten(), bins=bins_[7], density=True, alpha=0.5, label='Last period')
    ax[1,3].legend()
    ax[1,3].set_title(r'Variance in first and last period')

    for k,v in xlim_dict.items():
        ax[k].set_xlim(v)

    plt.tight_layout()

def plot_Vts(draws, periods, burnin=0):

    # periods should be a list of integers corresponding to the periods to plot
    # function should infer rows and cols from the number of periods provided in the list periods

    f, ax = plt.subplots(1, len(periods), figsize=(12, 8))

    for t in periods:
        ax[periods.index(t)].hist(draws['V_draws'][t, burnin:], bins=100)
        ax[periods.index(t)].set_title(rf'$V_{t}$ draws')
    
    
def plot_VY(draws, index, burnin=0, marked_intervals=None, vlines=None, ylim=None, filename=None):


    V = draws.V[:,:, burnin:]
    lo, hi = np.nanpercentile(V, [0.5, 99.5], axis=(0,2))

    # if plot_dY:

    #     f, ax = plt.subplots(2, 1, figsize=(12, 8))

    #     ax[0].plot(index, np.mean(V, axis=(0,2)), label=r'$\mathbb{E}[V_t]$')
    #     ax[0].fill_between(index, lo, hi, alpha=0.3, label='99% interval')
    #     ax[0].grid()

    #     ax[0].set_title(r'$\mathbb{E}[V_t]$')
    #     ax[0].legend()

    #     ax[1].plot(index[1:], dY)
    #     ax[1].set_title(r'SP500 log-returns: $dY_t$')
    #     ax[1].grid()

    #     for i in range(2):
    #         plt.setp(ax[i].get_xticklabels(), rotation=45, ha='right')
    #         ax[i].grid(alpha=0.5)

    #     if marked_intervals is not None:
    #         for (text, start, end, color, place) in marked_intervals:

    #             start, end = pd.Timestamp(start), pd.Timestamp(end)

    #             ax[0].axvspan(start, end, color=color, alpha=0.2)
    #             ax[1].axvspan(start, end, color=color, alpha=0.2)

    #             fx, fy = place
    #             x = start + fx * (end - start)   # fx=0.5 -> midt i intervallet

    #             ax[0].text(
    #                 x, fy, text,
    #                 transform=ax[0].get_xaxis_transform(),  # x = data, y = andel af aksehøjde (0-1)
    #                 ha="center", va="center",
    #                 fontsize=9, color=color,
    #                 clip_on=True,
    #             )

    #else:

    f, ax = plt.subplots(figsize=(12, 4))

    ax.plot(index, np.mean(V, axis=(0,2)), label=r'$\mathbb{E}[V_t \vert Y]$')
    ax.fill_between(index, lo, hi, alpha=0.3, label='99 pct. interval')
    ax.grid()
    ax.set_ylim(ylim)
    ax.set_xlim([pd.Timedelta(days=-30) + index[0], pd.Timedelta(days=30) +  index[-1]])
    ax.set_ylabel('$V_t$', fontsize=10, rotation=0)

    ax.set_title('Estimated Time Series of Variance')
    f.legend(loc='lower center', bbox_to_anchor=(0.5,-0.1), ncols=2, fontsize=10)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.grid(alpha=0.5)

    if marked_intervals is not None:
        for (text, start, end, color, place) in marked_intervals:

            start, end = pd.Timestamp(start), pd.Timestamp(end)

            ax.axvspan(start, end, color=color, alpha=0.2)

            fx, fy = place
            x = start + fx * (end - start)

            ax.text(
                x, fy, text,
                transform=ax.get_xaxis_transform(),
                ha="center", va="center",
                fontsize=9, color=color,
                clip_on=True,
            )
    
    if vlines is not None:
        for (text, date, color, place) in vlines:

            date = pd.Timestamp(date)
            datep = date + pd.Timedelta(days=1) 

            ax.vlines(date, -10, 100, color=color, ls='--', alpha=0.5)
            
            fx, fy = place
            x = date + fx * (datep - date)

            ax.text(
                x, fy, text,
                transform=ax.get_xaxis_transform(),
                ha="center", va="center",
                fontsize=9, color=color,
                clip_on=True,
            )

    f.tight_layout()

    if filename is not None: f.savefig(f'plots/{filename}', bbox_inches='tight')

def plot_VYconv(draws, Rhat, ESS, index, burnin=0, marked_intervals=None, vlines=None, ylim=None, filename=None):


    V = draws.V[:,:, burnin:]
    lo, hi = np.nanpercentile(V, [0.5, 99.5], axis=(0,2))

    f, ax = plt.subplots(2,1,figsize=(12, 8))

    # Variance time series
    ax[0].plot(index, np.mean(V, axis=(0,2)), label=r'$\mathbb{E}[V_t \vert Y]$')
    ax[0].fill_between(index, lo, hi, alpha=0.3, label='99 pct. interval')
    ax[0].set_ylim(ylim)
    ax[0].set_ylabel('$V_t$', fontsize=10, rotation=0)
    ax[0].set_title('Estimated Time Series of Variance')
    ax[0].legend(loc='upper right', fontsize=10)

    # Rhat and ESS series
    ESS_color = 'brown'
    l1,  = ax[1].plot(index, ESS, label=r'$n_{eff}$ (left axis)', color=ESS_color)
    ax[1].set_ylabel(r'Effective Sample Size: $n_{eff}$', color=ESS_color)
    ax[1].set_ylim([-100, ESS.max()+100])
    ax[1].tick_params(axis="y", labelcolor=ESS_color)

    Rhat_color = 'grey'
    ax1_twin = ax[1].twinx()
    l2, = ax1_twin.plot(index, Rhat, label=r'$\hat{R}$ (right axis)', color=Rhat_color)
    ax1_twin.set_ylim([0.99,1.12])
    ax1_twin.set_ylabel(r'Potential Scale Reduction: $\hat{R}$', color=Rhat_color)
    ax1_twin.tick_params(axis="y", labelcolor=Rhat_color)

    ax[1].set_title('Convergence Metrics for Estimated Variance Time Series')
    ax[1].legend(handles=[l1,l2])


    for ax_ in ax:
        plt.setp(ax_.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax_.set_xlim([pd.Timedelta(days=-30) + index[0], pd.Timedelta(days=30) +  index[-1]])
        ax_.grid(alpha=0.5)

        if marked_intervals is not None:
            for (text, start, end, color, place) in marked_intervals:

                start, end = pd.Timestamp(start), pd.Timestamp(end)

                ax_.axvspan(start, end, color=color, alpha=0.4)

                fx, fy = place
                x = start + fx * (end - start)

                
                if ax_ == ax[0]: ax_.text(
                    x, fy, text,
                    transform=ax_.get_xaxis_transform(),
                    ha="center", va="center",
                    fontsize=9, color=color,
                    clip_on=True,
                )
        
        if vlines is not None:
            for (text, date, color, place) in vlines:

                date = pd.Timestamp(date)
                datep = date + pd.Timedelta(days=1) 

                ax_.vlines(date, -20000, 20000, color=color, ls='--', alpha=0.5, linewidth=2.5)
                
                fx, fy = place
                x = date + fx * (datep - date)

                if ax_ == ax[0]: ax_.text(
                    x, fy, text,
                    transform=ax_.get_xaxis_transform(),
                    ha="center", va="center",
                    fontsize=9, color=color,
                    clip_on=True,
                )

    f.tight_layout()

    if filename is not None: f.savefig(f'plots/{filename}', bbox_inches='tight')