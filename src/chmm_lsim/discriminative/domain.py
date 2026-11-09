"""The compact feasible domain the discriminative trainer optimizes over (port of lsim_disc_domain.m).

* probabilities: supported entries of every probability block >= prob_floor; entries that are
  exactly zero in the initial model are structural zeros and stay zero;
* means in [lo - m*span, hi + m*span] of the pooled training data, per feature dimension;
* variances in [var_lo_frac, var_hi_frac] x the pooled training variance.

All constraints are convex, so the segment between two feasible models stays feasible.
Probability blocks: pi per channel, each transition row restricted to a destination channel,
each coupling column, each state's mixture weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.params import LSIMParams


class DomainError(ValueError):
    pass


@dataclass
class Domain:
    mu_lo: NDArray[np.float64]
    mu_hi: NDArray[np.float64]
    var_lo: NDArray[np.float64]
    var_hi: NDArray[np.float64]
    prob_floor: float
    support_pi: NDArray[np.bool_]
    support_A: NDArray[np.bool_]
    support_theta: NDArray[np.bool_]
    support_w: list[NDArray[np.bool_]]


def pooled_feature_stats(Y_list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """nanmin, nanmax and nanvar (ddof 1, MATLAB var omitnan) of every feature over all frames."""
    D = Y_list[0].shape[0]
    cnt = np.zeros(D)
    tot = np.zeros(D)
    lo = np.full(D, np.inf)
    hi = np.full(D, -np.inf)
    for Y in Y_list:
        m = ~np.isnan(Y)
        cnt += m.sum(axis=1)
        tot += np.where(m, Y, 0.0).sum(axis=1)
        with np.errstate(all="ignore"):
            lo = np.minimum(lo, np.nanmin(np.where(m, Y, np.inf), axis=1))
            hi = np.maximum(hi, np.nanmax(np.where(m, Y, -np.inf), axis=1))
    mean = tot / cnt
    ss = np.zeros(D)
    for Y in Y_list:
        dev = Y - mean[:, None]
        ss += np.where(np.isnan(dev), 0.0, dev * dev).sum(axis=1)
    var = ss / (cnt - 1)
    return lo, hi, var


def build_domain(Y_list, p: LSIMParams, *, prob_floor: float, mean_margin: float, var_lo_frac: float,
                 var_hi_frac: float, feature_stats=None) -> Domain:
    lo, hi, v = pooled_feature_stats(Y_list) if feature_stats is None else feature_stats
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(np.isfinite(v))) or np.any(v <= 0):
        raise DomainError("every feature dimension needs finite, non-constant training data")
    span = hi - lo
    return Domain(
        mu_lo=lo - mean_margin * span, mu_hi=hi + mean_margin * span,
        var_lo=var_lo_frac * v, var_hi=var_hi_frac * v, prob_floor=prob_floor,
        support_pi=p.pi_0 > 0, support_A=p.transition > 0, support_theta=p.coupling > 0,
        support_w=[g.weights > 0 for g in p.emissions],
    )


def floor_block(q: NDArray[np.float64], support: NDArray[np.bool_], floor_value: float) -> NDArray[np.float64]:
    """Block restricted to its support, renormalized, every supported entry >= floor_value
    (mixing with the uniform distribution on the support by the smallest sufficient weight)."""
    q = np.array(q, dtype=np.float64, copy=True)
    q[~support] = 0.0
    m = int(np.count_nonzero(support))
    if m == 0:
        return q
    if floor_value * m >= 1:
        raise DomainError(f"prob_floor {floor_value:g} is infeasible for a block with {m} supported entries")
    q[support] = q[support] / q[support].sum()
    low = q[support].min()
    if low < floor_value:
        u = 1.0 / m
        lam = (floor_value - low) / (u - low)
        q[support] = (1 - lam) * q[support] + lam * u
    return q


def project(p: LSIMParams, dom: Domain) -> LSIMParams:
    s = p.structure
    q = p.copy()
    S = s.S
    for z in range(s.C):
        zi = s.state_slice(z)
        q.pi_0[zi] = floor_block(q.pi_0[zi], dom.support_pi[zi], dom.prob_floor)
        for i in range(S):
            q.transition[i, zi] = floor_block(q.transition[i, zi], dom.support_A[i, zi], dom.prob_floor)
        q.coupling[:, z] = floor_block(q.coupling[:, z], dom.support_theta[:, z], dom.prob_floor)
        g = q.emissions[z]
        ds = s.dim_slice(z)
        for i in range(g.M_c):
            g.weights[i, :] = floor_block(g.weights[i, :], dom.support_w[z][i, :], dom.prob_floor)
        g.mu[:] = np.minimum(np.maximum(g.mu, dom.mu_lo[ds, None, None]), dom.mu_hi[ds, None, None])
        g.sigma[:] = np.minimum(np.maximum(g.sigma, dom.var_lo[ds, None, None]), dom.var_hi[ds, None, None])
    return q


def check(p: LSIMParams, dom: Domain, tol: float = 1e-9) -> bool:
    s = p.structure

    def block_ok(q, sup):
        return (np.all(q[~sup] == 0) and abs(q.sum() - 1) <= tol
                and (not sup.any() or q[sup].min() >= dom.prob_floor * (1 - tol)))

    for z in range(s.C):
        zi = s.state_slice(z)
        if not block_ok(p.pi_0[zi], dom.support_pi[zi]):
            return False
        for i in range(s.S):
            if not block_ok(p.transition[i, zi], dom.support_A[i, zi]):
                return False
        if not block_ok(p.coupling[:, z], dom.support_theta[:, z]):
            return False
        g = p.emissions[z]
        ds = s.dim_slice(z)
        for i in range(g.M_c):
            if not block_ok(g.weights[i], dom.support_w[z][i]):
                return False
        lo, hi = dom.mu_lo[ds, None, None], dom.mu_hi[ds, None, None]
        if np.any(g.mu < lo - tol) or np.any(g.mu > hi + tol):
            return False
        if np.any(g.sigma < dom.var_lo[ds, None, None] * (1 - tol)) or np.any(g.sigma > dom.var_hi[ds, None, None] * (1 + tol)):
            return False
    return True


def at_bounds(p: LSIMParams, dom: Domain) -> dict[str, float]:
    """Fractions of parameters at the domain bounds (blocks with >= 2 supported entries)."""
    s = p.structure
    rel = 1e-9
    nprob = nfloor = nvar = nlo = nhi = nmean = nmb = 0
    blocks = []
    for z in range(s.C):
        zi = s.state_slice(z)
        blocks.append((p.pi_0[zi], dom.support_pi[zi]))
        for i in range(s.S):
            blocks.append((p.transition[i, zi], dom.support_A[i, zi]))
        blocks.append((p.coupling[:, z], dom.support_theta[:, z]))
        g = p.emissions[z]
        ds = s.dim_slice(z)
        for i in range(g.M_c):
            blocks.append((g.weights[i], dom.support_w[z][i]))
        v, mu = g.sigma, g.mu
        vlo, vhi = dom.var_lo[ds, None, None], dom.var_hi[ds, None, None]
        mlo, mhi = dom.mu_lo[ds, None, None], dom.mu_hi[ds, None, None]
        nvar += v.size
        nlo += int(np.count_nonzero(v <= vlo * (1 + rel)))
        nhi += int(np.count_nonzero(v >= vhi * (1 - rel)))
        nmean += mu.size
        nmb += int(np.count_nonzero((mu <= mlo + rel * np.abs(mlo)) | (mu >= mhi - rel * np.abs(mhi))))
    for q, sup in blocks:
        if np.count_nonzero(sup) >= 2:
            nprob += int(np.count_nonzero(sup))
            nfloor += int(np.count_nonzero(q[sup] <= dom.prob_floor * (1 + rel)))
    return {"prob_at_floor": nfloor / max(nprob, 1), "var_at_lower": nlo / max(nvar, 1),
            "var_at_cap": nhi / max(nvar, 1), "mean_at_bound": nmb / max(nmean, 1)}
