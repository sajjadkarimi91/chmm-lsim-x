"""Emission likelihoods: B = P(y_t | state) and B_comp = the per-GMM-component
breakdown B is summed from. Ports `common function/gmm_pdf_fast.m` (diagonal
covariance) and `common function/gmm_pdf.m` (full covariance).

NaN in an observation means "missing" (§2.1.1): a missing *dimension*
contributes zero log-density (marginalised out); a channel missing in *every*
dimension at a timestep contributes likelihood 1 for every state, matching
canonical MATLAB post-P0b (§0.4.9 for the diagonal path, §3bis.2 for full).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal

from chmm_lsim.params import ChannelGMM, LSIMParams
from chmm_lsim.structure import LSIMStructure

_LOG_2PI = float(np.log(2 * np.pi))


def gmm_pdf_diag(structure: LSIMStructure, params: LSIMParams, Y: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
    """Diagonal-covariance emissions. Returns (B (S,T), B_comp (S,max_K,T))."""
    if not params.sigma_diag:
        raise ValueError("gmm_pdf_diag requires params.sigma_diag=True; use gmm_pdf_full for full covariance")
    T = Y.shape[1]
    B_comp = np.zeros((structure.S, structure.max_K, T))

    for c in range(structure.C):
        gmm = params.emissions[c]
        Yc = Y[structure.dim_slice(c), :]  # (d_c, T)
        nan_mask = np.isnan(Yc)
        s0 = structure.state_offsets[c]
        for s in range(gmm.M_c):
            for k in range(gmm.K_c):
                mu = gmm.mu[:, s, k][:, None]  # (d_c, 1)
                sigma = gmm.sigma[:, s, k][:, None]  # (d_c, 1) variances
                quad = (Yc - mu) ** 2 / sigma
                log_sqrt_det = 0.5 * np.log(sigma)
                log_dens = -0.5 * quad - log_sqrt_det - 0.5 * _LOG_2PI  # (d_c, T)
                log_dens = np.where(nan_mask, 0.0, log_dens)
                loglik = log_dens.sum(axis=0)  # (T,)
                B_comp[s0 + s, k, :] = gmm.weights[s, k] * np.exp(loglik)

    B = B_comp.sum(axis=1)
    return B, B_comp


def gmm_pdf_full(structure: LSIMStructure, params: LSIMParams, Y: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
    """Full-covariance emissions. Returns (B (S,T), B_comp (S,max_K,T))."""
    if params.sigma_diag:
        raise ValueError("gmm_pdf_full requires params.sigma_diag=False; use gmm_pdf_diag for diagonal covariance")
    T = Y.shape[1]
    B_comp = np.zeros((structure.S, structure.max_K, T))

    for c in range(structure.C):
        gmm: ChannelGMM = params.emissions[c]
        Yc = Y[structure.dim_slice(c), :].T  # (T, d_c)
        nan_mask = np.isnan(Yc)
        patterns, inverse = np.unique(nan_mask, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        s0 = structure.state_offsets[c]

        for s in range(gmm.M_c):
            for k in range(gmm.K_c):
                mu = gmm.mu[:, s, k]
                sigma = gmm.sigma[:, :, s, k]
                comp_vals = np.empty(T)
                for p_idx in range(patterns.shape[0]):
                    rows = inverse == p_idx
                    missing = patterns[p_idx]
                    observed = ~missing
                    if not observed.any():
                        comp_vals[rows] = 1.0
                        continue
                    X = Yc[np.ix_(rows, observed)]
                    mu_sub = mu[observed]
                    sigma_sub = sigma[np.ix_(observed, observed)]
                    comp_vals[rows] = multivariate_normal.pdf(X, mean=mu_sub, cov=sigma_sub, allow_singular=True)
                B_comp[s0 + s, k, :] = gmm.weights[s, k] * comp_vals

    B = B_comp.sum(axis=1)
    return B, B_comp


def compute_emissions(structure: LSIMStructure, params: LSIMParams, Y: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
    """Dispatches on `params.sigma_diag`, matching how the MATLAB kernel picks
    `gmm_pdf_fast` vs `gmm_pdf` from `packed.sigma_diag`."""
    if params.sigma_diag:
        return gmm_pdf_diag(structure, params, Y)
    return gmm_pdf_full(structure, params, Y)
