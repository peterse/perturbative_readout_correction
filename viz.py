import matplotlib.pyplot as plt
import numpy as np

import utils


def u_imshow(u, n, ax=None, sort_by_weight=True):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # Rearrange labels on the space of U according to bitstring weight

    if sort_by_weight:
        idx = utils.idxsort_by_weight(n)
        U_ordered = u[:,idx][idx,:]
    else:
        idx = np.arange(1 << n)
        U_ordered = u
    re = ax.imshow(U_ordered, cmap='seismic', vmin=-1, vmax=1)

    # fig.subplots_adjust(right=0.85)
    # cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    ticks = np.arange(1 << n)
    # ticklabels get shuffled according to weight
    ticklabs = [utils.int2bin_lendian(i, n) for i in ticks[idx]]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabs, rotation=90, size=12)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabs, size=12)

    # Put boundaries between fixed-particle number regions
    if sort_by_weight:
        k = 0
        for i in range(n):
            k += utils.ncr(n, i)
            ax.axhline(k - 0.5, ls=':', lw=1, c='k', alpha=1)
            ax.axvline(k - 0.5, ls=':', lw=1, c='k', alpha=1)

    return idx


def u_imshow_trunc(u, n, trunc=None, ax=None, sort_by_weight=True):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # Rearrange labels on the space of U according to bitstring weight

    if sort_by_weight:
        idx = utils.idxsort_by_weight(n)[:trunc]
        U_ordered = u[:,idx][idx,:]
    else:
        idx = np.arange(1 << n)[:trunc]
        U_ordered = u
    re = ax.imshow(U_ordered, cmap='seismic', vmin=-1, vmax=1)

    # fig.subplots_adjust(right=0.85)
    # cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    ticks = np.arange(1 << n)
    # ticklabels get shuffled according to weight
    ticklabs = [utils.int2bin_lendian(i, n) for i in ticks[idx]]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabs, rotation=90, size=12)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabs, size=12)

    # Put boundaries between fixed-particle number regions
    if sort_by_weight:
        k = 0
        for i in range(n):
            k += utils.ncr(n, i)
            ax.axhline(k - 0.5, ls=':', lw=1, c='k', alpha=1)
            ax.axvline(k - 0.5, ls=':', lw=1, c='k', alpha=1)

    return idx