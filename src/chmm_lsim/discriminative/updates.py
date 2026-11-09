"""Parameter updates from numerator / denominator counts computed under the current model.

* :func:`published_update` -- MMI, published recipe (Woodland & Povey 2002): I-smoothing of the
  Gaussian numerator statistics (tau back-off pseudo-counts), the Gaussian EBW update with
  D = max(E gamma^den, 2 D_min), and the weak-sense auxiliary update for discrete blocks.
  Full step, no acceptance test. Port of lsim_disc_published_update.m; the update of the trainer.
* :func:`weak_sense` -- the minorize-maximize update of one probability vector
  (port of lsim_disc_weak_sense.m).
* :func:`ebw_proposal` -- the plain Extended-Baum-Welch update of every block from the signed
  counts d = num - den, with D = max(E * den mass, 2 * D_min) per block (Povey & Woodland
  ICASSP 2001, eqs. 3-4), and :func:`interpolate`, the segment between two models. Ports of
  lsim_disc_ebw_proposal.m, the update of MATLAB's ``mmi_recipe='legacy'`` (which searches the
  step along that segment); standalone utilities, not used by the Python trainer.

The model updates return a NEW LSIMParams projected into the domain; structural zeros stay zero.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.discriminative.counts import DiscCounts
from chmm_lsim.discriminative.domain import Domain, project
from chmm_lsim.params import LSIMParams

_FAMILIES = {"pi": "pi_0", "A": "transition", "theta": "coupling", "w": "weights", "mu": "mu", "var": "sigma"}


def _freeze(q: LSIMParams, p: LSIMParams, update_blocks) -> None:
    """Families not in update_blocks are restored exactly to p (in place on q)."""
    if not update_blocks:
        return
    ub = set(update_blocks)
    if "pi" not in ub:
        q.pi_0 = p.pi_0.copy()
    if "A" not in ub:
        q.transition = p.transition.copy()
    if "theta" not in ub:
        q.coupling = p.coupling.copy()
    for gq, gp in zip(q.emissions, p.emissions):
        if "w" not in ub:
            gq.weights = gp.weights.copy()
        if "mu" not in ub:
            gq.mu = gp.mu.copy()
        if "var" not in ub:
            gq.sigma = gp.sigma.copy()


# ----------------------------------------------------------------------------- EBW proposal
def _ebw_block(p, dnum, dden, support, E):
    p_new = p.copy()
    idx = support.astype(bool)
    if np.count_nonzero(idx) <= 1:
        return p_new, 0.0
    pv = p[idx]
    d = dnum[idx] - dden[idx]
    den_mass = dden[idx].sum()
    if np.abs(dnum[idx]).sum() + den_mass == 0:
        return p_new, 0.0
    D_min = max(0.0, np.max(-d / pv), -d.sum())
    D = max(E * den_mass, 2 * D_min)
    denom = d.sum() + D
    if not denom > 0:
        return p_new, float(D)
    p_new[idx] = (d + D * pv) / denom
    return p_new, float(D)


def _gauss_D_min(mu, v, n, s, r):
    a = v
    b = r + n * (v + mu ** 2) - 2 * s * mu
    c = r * n - s ** 2
    disc = b ** 2 - 4 * a * c
    root = np.full(mu.shape, -np.inf)
    has = disc >= 0
    with np.errstate(invalid="ignore"):
        root[has] = (-b[has] + np.sqrt(disc[has])) / (2 * a[has])
    return max(0.0, float(np.max(root, initial=-np.inf)), float(np.max(-n, initial=-np.inf)))


def _gauss_apply(mu, v, n, s, r, D):
    mu_new = mu.copy()
    v_new = v.copy()
    denom = n + D
    ok = denom > 1e-12 * max(1.0, abs(D))
    mu_new[ok] = (s[ok] + D * mu[ok]) / denom[ok]
    v_new[ok] = (r[ok] + D * (v[ok] + mu[ok] ** 2)) / denom[ok] - mu_new[ok] ** 2
    bad = ~(v_new > 0) | ~np.isfinite(v_new) | ~np.isfinite(mu_new)
    mu_new[bad] = mu[bad]
    v_new[bad] = v[bad]
    return mu_new, v_new


def _ebw_gauss(mu, v, n, s, r, den_occ, E):
    if np.all(n == 0) and np.all(s == 0) and den_occ == 0:
        return mu.copy(), v.copy(), 0.0
    D_min = _gauss_D_min(mu, v, n, s, r)
    D = max(E * den_occ, 2 * D_min)
    mu_new, v_new = _gauss_apply(mu, v, n, s, r, D)
    return mu_new, v_new, float(D)


def ebw_proposal(p: LSIMParams, num: DiscCounts, den: DiscCounts, dom: Domain, damping_E: float,
                 update_blocks=()) -> tuple[LSIMParams, dict]:
    """EBW update of every block with damping multiplier ``damping_E``, projected into ``dom``."""
    st = p.structure
    q = p.copy()
    D_used: list[float] = []
    for z in range(st.C):
        zi = st.state_slice(z)
        q.pi_0[zi], Du = _ebw_block(p.pi_0[zi], num.pi[zi], den.pi[zi], dom.support_pi[zi], damping_E)
        D_used.append(Du)
        for i in range(st.S):
            q.transition[i, zi], Du = _ebw_block(p.transition[i, zi], num.A[i, zi], den.A[i, zi],
                                                 dom.support_A[i, zi], damping_E)
            D_used.append(Du)
        q.coupling[:, z], Du = _ebw_block(p.coupling[:, z], num.theta[:, z], den.theta[:, z],
                                          dom.support_theta[:, z], damping_E)
        D_used.append(Du)
        gp, gq = p.emissions[z], q.emissions[z]
        for i in range(gp.M_c):
            gq.weights[i, :], Du = _ebw_block(gp.weights[i, :], num.occ[z][i, :], den.occ[z][i, :],
                                              dom.support_w[z][i, :], damping_E)
            D_used.append(Du)
            for k in range(gp.K_c):
                mu, v, Du = _ebw_gauss(
                    gp.mu[:, i, k], gp.sigma[:, i, k],
                    num.n[z][:, i, k] - den.n[z][:, i, k], num.s[z][:, i, k] - den.s[z][:, i, k],
                    num.r[z][:, i, k] - den.r[z][:, i, k], den.occ[z][i, k], damping_E)
                gq.mu[:, i, k] = mu
                gq.sigma[:, i, k] = v
                D_used.append(Du)
    q = project(q, dom)
    _freeze(q, p, update_blocks)
    return q, {"max_D": max(D_used + [0.0]), "median_D": float(np.median(D_used))}


# ----------------------------------------------------------------------------- weak sense
def weak_sense(gnum: NDArray, gden: NDArray, p0: NDArray, max_iter: int = 100, tol: float = 1e-10):
    """Maximize G(p) = sum_j gnum_j log p_j - gden_j p_j / p0_j over the simplex on supp(p0)."""
    gnum = np.asarray(gnum, float).ravel()
    gden = np.asarray(gden, float).ravel()
    p0 = np.asarray(p0, float).ravel()
    p = p0.copy()
    sup = p0 > 0
    info = {"iterations": 0, "aux": np.zeros(0), "monotone": True}
    if np.count_nonzero(sup) <= 1 or (gnum[sup].sum() <= 0 and np.all(gden[sup] == 0)):
        return p, info
    if np.any(gnum[sup] < 0) or np.any(gden[sup] < 0):
        raise ValueError("weak_sense: numerator and denominator counts must be nonnegative")
    g = gnum[sup]
    c = gden[sup] / p0[sup]
    k = c.max() - c
    x = p0[sup].copy()
    gp = g > 0

    def aux(y):
        return np.sum(g[gp] * np.log(y[gp])) - np.sum(c * y)

    G = [aux(x)]
    it = 0
    tiny = np.finfo(float).tiny
    for it in range(1, max_iter + 1):
        y = g + k * x
        y = y / y.sum()
        G.append(aux(y))
        change = np.max(np.abs(y - x) / np.maximum(x, tiny))
        x = y
        if change < tol:
            break
    G = np.asarray(G)
    p[sup] = x
    p[~sup] = 0.0
    info["iterations"] = it
    info["aux"] = G
    info["monotone"] = bool(np.all(np.diff(G) >= -1e-12 * np.maximum(1, np.abs(G[:-1]))))
    return p, info


# ----------------------------------------------------------------------------- published MMI
def _discrete(p0, gnum, gden, support, tau_d, iters, tol):
    p0 = np.asarray(p0, float)
    sup = support.astype(bool) & (p0 > 0)
    if np.count_nonzero(sup) <= 1:
        return p0.copy(), 0, True
    g = np.where(sup, gnum, 0.0)
    h = np.where(sup, gden, 0.0)
    if tau_d > 0 and g.sum() > 0:
        g = g + tau_d * (g / g.sum())
    pin = np.where(sup, p0, 0.0)
    x, inf = weak_sense(g, h, pin, iters, tol)
    return x, inf["iterations"], inf["monotone"]


def _published_gauss(mu, v, gn, xn, Sn, gd, xd, Sd, den_occ, E, tau, backoff):
    if tau > 0:
        if backoff == "ml":
            mu_b = mu.copy()
            v_b = v.copy()
            ok = gn > 0
            mu_b[ok] = xn[ok] / gn[ok]
            vb_ml = Sn[ok] / gn[ok] - mu_b[ok] ** 2
            vb = v_b[ok]
            good = (vb_ml > 0) & np.isfinite(vb_ml)
            vb[good] = vb_ml[good]
            v_b[ok] = vb
            bad = ok.copy()
            bad[ok] = ~good
            mu_b[bad] = mu[bad]
        elif backoff == "previous":
            mu_b, v_b = mu, v
        else:
            raise ValueError("ismooth_backoff must be 'ml' or 'previous'")
        gn = gn + tau
        xn = xn + tau * mu_b
        Sn = Sn + tau * (v_b + mu_b ** 2)
    n = gn - gd
    s = xn - xd
    r = Sn - Sd
    if np.all(n == 0) and np.all(s == 0) and den_occ == 0:
        return mu.copy(), v.copy(), 0.0
    D_min = _gauss_D_min(mu, v, n, s, r)
    D = max(E * den_occ, 2 * D_min)
    mu_new, v_new = _gauss_apply(mu, v, n, s, r, D)
    return mu_new, v_new, float(D)


def published_update(p: LSIMParams, num: DiscCounts, den: DiscCounts, dom: Domain, *, ebw_E: float = 2.0,
                     ismooth_tau: float = 100.0, ismooth_backoff: str = "ml", ismooth_tau_discrete: float = 0.0,
                     weak_sense_iters: int = 100, weak_sense_tol: float = 1e-10,
                     update_blocks=()) -> tuple[LSIMParams, dict]:
    st = p.structure
    q = p.copy()
    D_used: list[float] = []
    inner: list[int] = []
    aux_ok = True
    td, wi, wt = ismooth_tau_discrete, weak_sense_iters, weak_sense_tol
    for z in range(st.C):
        zi = st.state_slice(z)
        q.pi_0[zi], it, ok = _discrete(p.pi_0[zi], num.pi[zi], den.pi[zi], dom.support_pi[zi], td, wi, wt)
        inner.append(it); aux_ok &= ok
        for i in range(st.S):
            q.transition[i, zi], it, ok = _discrete(p.transition[i, zi], num.A[i, zi], den.A[i, zi],
                                                    dom.support_A[i, zi], td, wi, wt)
            inner.append(it); aux_ok &= ok
        q.coupling[:, z], it, ok = _discrete(p.coupling[:, z], num.theta[:, z], den.theta[:, z],
                                             dom.support_theta[:, z], td, wi, wt)
        inner.append(it); aux_ok &= ok
        gp, gq = p.emissions[z], q.emissions[z]
        for i in range(gp.M_c):
            gq.weights[i, :], it, ok = _discrete(gp.weights[i, :], num.occ[z][i, :], den.occ[z][i, :],
                                                 dom.support_w[z][i, :], td, wi, wt)
            inner.append(it); aux_ok &= ok
            for k in range(gp.K_c):
                mu, v, Du = _published_gauss(
                    gp.mu[:, i, k], gp.sigma[:, i, k],
                    num.n[z][:, i, k], num.s[z][:, i, k], num.r[z][:, i, k],
                    den.n[z][:, i, k], den.s[z][:, i, k], den.r[z][:, i, k], den.occ[z][i, k],
                    ebw_E, ismooth_tau, ismooth_backoff)
                gq.mu[:, i, k] = mu
                gq.sigma[:, i, k] = v
                D_used.append(Du)
    _freeze(q, p, update_blocks)
    qp = project(q, dom)
    changed = int(np.count_nonzero(qp.pi_0 != q.pi_0) + np.count_nonzero(qp.transition != q.transition)
                  + np.count_nonzero(qp.coupling != q.coupling))
    for a, b in zip(qp.emissions, q.emissions):
        changed += int(np.count_nonzero(a.weights != b.weights) + np.count_nonzero(a.mu != b.mu)
                       + np.count_nonzero(a.sigma != b.sigma))
    info = {"max_D": max(D_used + [0.0]), "median_D": float(np.median(D_used)) if D_used else float("nan"),
            "projection_changed_total": changed, "weak_sense_max_iterations": max(inner + [0]),
            "weak_sense_monotone": bool(aux_ok)}
    return qp, info


def interpolate(p: LSIMParams, q: LSIMParams, tau: float, update_blocks=()) -> LSIMParams:
    """The segment p + tau (q - p) (stays in the convex domain); frozen families copied exactly."""
    ub = set(update_blocks) if update_blocks else set(_FAMILIES)
    c = p.copy()
    if "pi" in ub:
        c.pi_0 = (1 - tau) * p.pi_0 + tau * q.pi_0
    if "A" in ub:
        c.transition = (1 - tau) * p.transition + tau * q.transition
    if "theta" in ub:
        c.coupling = (1 - tau) * p.coupling + tau * q.coupling
    for gc, gp, gq in zip(c.emissions, p.emissions, q.emissions):
        if "w" in ub:
            gc.weights = (1 - tau) * gp.weights + tau * gq.weights
        if "mu" in ub:
            gc.mu = (1 - tau) * gp.mu + tau * gq.mu
        if "var" in ub:
            gc.sigma = (1 - tau) * gp.sigma + tau * gq.sigma
    return c
