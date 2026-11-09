"""GEMM formulation of the batched POISE backward pass (raw POISE, no refinement).

``chmm_lsim.backends.torch_batched._backward_poise_batch`` forms, at every time step, the (Bn, S, S)
elementwise product ``transition * beta_next`` and block-sums it:

    child[b, i, c] = sum_{j in channel c} A[i, j] beta_next[b, j]

which is pure memory traffic (O(Bn S^2) bytes per step, ~2% of fp64 peak at S = 250). The same
quantity is C small matrix products, ``beta_next[:, block c] @ A[:, block c]^T`` (Bn x M_c times
M_c x S), so this module computes it with batched GEMMs and never materializes an (Bn, S, S)
tensor. Everything else (exponent theta_out, mixture rho, normalization, error on a nonpositive
channel normalizer) is the backend's own code path, in the same order, so results agree with
``_backward_poise_batch`` up to the summation order inside the GEMM (~1e-15 relative; tested).

Memory per sequence is O(S T) (alpha, scale, beta, gamma, B, B_comp), so far larger chunks fit
than with SED; :func:`poise_chunk_size` sizes them.
"""

from __future__ import annotations

import numpy as np


def backward_poise_gemm(torch, tb, kernel, alpha, scale, lengths):
    """Raw-POISE backward (``poise_refinement=False``); returns beta, gamma (Bn, S, T)."""
    Bn, S, T = alpha.shape
    st = kernel.structure
    C = st.C
    rho, theta_out, zero_theta = tb._poise_constants(kernel)
    _b, broadcast_channels, block_sum_last, _bb = tb._onehot_ops(kernel)
    A = kernel.transition
    M_eq = kernel.equal_block_size
    if M_eq is not None:
        # A3[c, m, i] = A[i, c*M + m]: child = (beta3 @ A3) batched over channels
        A3 = A.T.reshape(C, M_eq, S).contiguous()
    else:
        A_blocks = [A[:, sl].T.contiguous() for sl in kernel.channel_slices]   # (M_c, S)
    beta = torch.zeros(Bn, S, T, dtype=alpha.dtype, device=alpha.device)
    gamma = torch.zeros(Bn, S, T, dtype=alpha.dtype, device=alpha.device)
    beta[:, :, T - 1] = scale[:, :, T - 1]
    gamma[:, :, T - 1] = alpha[:, :, T - 1] * beta[:, :, T - 1]
    first_bad = torch.full((Bn, C), -1, dtype=torch.int64, device=alpha.device)
    rho_b = rho[None, :, :]
    theta_b = theta_out[None, :]
    zero_b = zero_theta[None, :]
    for tau in range(T - 2, -1, -1):
        beta_next = beta[:, :, tau + 1]
        if M_eq is not None:
            b3 = beta_next.reshape(Bn, C, M_eq).transpose(0, 1)               # (C, Bn, M)
            child = torch.bmm(b3, A3).permute(1, 2, 0)                        # (Bn, S, C)
        else:
            child = torch.stack([beta_next[:, sl] @ Ab for sl, Ab in zip(kernel.channel_slices, A_blocks)], dim=-1)
        q = (child * rho_b).sum(dim=-1)
        beta_tau = scale[:, :, tau] * q ** theta_b
        beta_tau = torch.where(zero_b, scale[:, :, tau], beta_tau)
        alpha_T_tau = beta_tau * alpha[:, :, tau]
        norm_c = block_sum_last(alpha_T_tau)
        bad = (~torch.isfinite(norm_c) | (norm_c <= 0)) & (tau < lengths)[:, None]
        first_bad = torch.where(bad & (first_bad < 0), tau, first_bad)
        norm = broadcast_channels(norm_c)
        beta[:, :, tau] = beta_tau / norm
        gamma[:, :, tau] = alpha_T_tau / norm
    if bool((first_bad >= 0).any()):
        b, c = (int(x) for x in np.argwhere(first_bad.cpu().numpy() >= 0)[0])
        raise FloatingPointError(f"POISE backward: channel {c} normalizer of batch row {b} is nonpositive or non-finite")
    return beta, gamma


def forward_backward_poise_gemm(torch, tb, kernel, Y, lengths):
    """Emissions, forward pass (shared by every variant) and the GEMM POISE backward.
    Returns alpha, beta, gamma, scale, B, B_comp, loglik (Bn,)."""
    B, Bc, log_scale = tb.compute_emissions_torch(kernel, Y)
    alpha, scale, ll = tb._forward_pass_batch(kernel, B, list(range(Y.shape[0])), lengths, False)
    loglik = ll + log_scale.sum(dim=(1, 2))
    if torch.isnan(loglik).any():
        raise FloatingPointError("forward pass produced a NaN log-likelihood")
    beta, gamma = backward_poise_gemm(torch, tb, kernel, alpha, scale, lengths)
    return alpha, beta, gamma, scale, B, Bc, loglik


def poise_chunk_size(torch, device, S: int, T: int, max_K: int, D: int, n_models: int = 1, fraction: float = 0.6,
                     cap: int = 20000) -> int:
    """Sequences per GEMM-POISE work unit from free device memory: per sequence ~ (6 S + 2 S K + 3 D)
    T doubles for the posteriors / emissions / data, plus the counts reduction's (R=2) weighted
    occupancies (2 S K T), times n_models when several models' posteriors are alive at once."""
    per_seq = 8 * T * (6 * S + 4 * S * max_K + 3 * D) * n_models * 1.5
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        budget = fraction * free
    else:
        budget = 4 * 1024 ** 3
    return int(max(1, min(cap, budget // per_seq)))
