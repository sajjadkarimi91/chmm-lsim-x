"""Block-level expected counts: the currency of every discriminative update.

Port of ``lsim_disc_sequence_stats.m`` / ``lsim_disc_counts.m``. For one sequence under one
model (``sed`` forward-backward, MATLAB's default), the expected complete-data sufficient
statistics are

* ``pi``    (S,)      gamma at t = 1
* ``A``     (S, S)    xi(i, j) = sum_t alpha(i,t) b(i,t) theta(c(i), z(j)) A(i,j) beta(j,t+1)
                      (em_lsim's ``sum3_zeta``; rows of a destination block are transition counts)
* ``theta`` (C, C)    block sums of xi (coupling source-selection counts)
* per channel c: ``occ[c]`` (M_c, K_c) component occupancy (unmasked) and ``n[c]``, ``s[c]``,
  ``r[c]`` (d_c, M_c, K_c) masked occupancy, first and elementwise second raw moments
  (missing entries contribute to none).

Counts are RAW sums (not central moments) so numerator and denominator statistics can be
weighted, summed and subtracted exactly as in MATLAB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.core.structure import LSIMStructure


class NonfiniteStatistics(FloatingPointError):
    """A sequence the model assigns zero mass to (MATLAB em_lsim_discriminative:nonfiniteStats)."""


@dataclass
class DiscCounts:
    pi: NDArray[np.float64]
    A: NDArray[np.float64]
    theta: NDArray[np.float64]
    occ: list[NDArray[np.float64]]
    n: list[NDArray[np.float64]]
    s: list[NDArray[np.float64]]
    r: list[NDArray[np.float64]]

    @classmethod
    def zeros(cls, structure: LSIMStructure) -> "DiscCounts":
        S, C = structure.S, structure.C
        M, K, d = structure.channel_state_num, structure.num_gmm_component, structure.channel_obs_dim
        return cls(
            pi=np.zeros(S), A=np.zeros((S, S)), theta=np.zeros((C, C)),
            occ=[np.zeros((M[c], K[c])) for c in range(C)],
            n=[np.zeros((d[c], M[c], K[c])) for c in range(C)],
            s=[np.zeros((d[c], M[c], K[c])) for c in range(C)],
            r=[np.zeros((d[c], M[c], K[c])) for c in range(C)],
        )

    def add_(self, other: "DiscCounts", w: float = 1.0) -> "DiscCounts":
        """In place: self += w * other (MATLAB lsim_disc_counts('add'))."""
        if w == 1.0:
            self.pi += other.pi
            self.A += other.A
            self.theta += other.theta
            for c in range(len(self.occ)):
                self.occ[c] += other.occ[c]
                self.n[c] += other.n[c]
                self.s[c] += other.s[c]
                self.r[c] += other.r[c]
            return self
        self.pi += w * other.pi
        self.A += w * other.A
        self.theta += w * other.theta
        for c in range(len(self.occ)):
            self.occ[c] += w * other.occ[c]
            self.n[c] += w * other.n[c]
            self.s[c] += w * other.s[c]
            self.r[c] += w * other.r[c]
        return self

    def scaled(self, a: float) -> "DiscCounts":
        return DiscCounts(self.pi * a, self.A * a, self.theta * a, [x * a for x in self.occ],
                          [x * a for x in self.n], [x * a for x in self.s], [x * a for x in self.r])

    def copy(self) -> "DiscCounts":
        return self.scaled(1.0)

    def arrays(self) -> list[NDArray[np.float64]]:
        return [self.pi, self.A, self.theta, *self.occ, *self.n, *self.s, *self.r]

    def max_abs_diff(self, other: "DiscCounts") -> float:
        return max(float(np.max(np.abs(a - b), initial=0.0)) for a, b in zip(self.arrays(), other.arrays()))


def block_sum_xi(xi: NDArray[np.float64], structure: LSIMStructure) -> NDArray[np.float64]:
    C = structure.C
    th = np.zeros((C, C))
    for c in range(C):
        sc = structure.state_slice(c)
        for z in range(C):
            th[c, z] = xi[sc, structure.state_slice(z)].sum()
    return th


def sequence_counts(structure: LSIMStructure, post, Y: NDArray[np.float64], W: NDArray[np.float64]) -> DiscCounts:
    """Counts of one sequence from its ``sed`` posterior (``chmm_lsim.Posterior``); ``W`` = build_W."""
    T = Y.shape[1]
    gamma, alpha, beta, scale = post.gamma, post.alpha, post.beta, post.scale
    if T > 1:
        xi = ((alpha[:, : T - 1] * scale[:, : T - 1]) @ beta[:, 1:].T) * W
    else:
        xi = np.zeros((structure.S, structure.S))
    out = DiscCounts(pi=gamma[:, 0].copy(), A=xi, theta=block_sum_xi(xi, structure), occ=[], n=[], s=[], r=[])
    B, Bc = post.B, post.B_comp
    for c in range(structure.C):
        sl = structure.state_slice(c)
        K = structure.num_gmm_component[c]
        Yc = Y[structure.dim_slice(c), :]
        mask = ~np.isnan(Yc)
        Y0 = np.where(mask, Yc, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = Bc[sl, :K, :] / B[sl, None, :]
        g = gamma[sl, None, :] * ratio                        # (M, K, T)
        g = np.where(np.isnan(g), 0.0, g)
        out.occ.append(g.sum(axis=2))
        mf = mask.astype(np.float64)
        out.n.append(np.einsum("dt,mkt->dmk", mf, g))
        out.s.append(np.einsum("dt,mkt->dmk", Y0, g))
        out.r.append(np.einsum("dt,mkt->dmk", Y0 * Y0, g))
    for name, arrs in (("pi", [out.pi]), ("xi", [out.A]), ("occ", out.occ), ("n", out.n), ("s", out.s), ("r", out.r)):
        if any(not np.all(np.isfinite(a)) for a in arrs):
            raise NonfiniteStatistics(
                f"non-finite {name} statistics (score {post.loglik}); the model assigns this sequence zero mass")
    return out


def counts_from_packed_struct(d: dict, structure: LSIMStructure) -> DiscCounts:
    """MATLAB lsim_disc_counts struct (padded C x maxS x maxK / D x maxS x maxK) -> DiscCounts."""
    out = DiscCounts(pi=np.asarray(d["pi"], float).ravel(), A=np.atleast_2d(np.asarray(d["A"], float)),
                     theta=np.atleast_2d(np.asarray(d["theta"], float)), occ=[], n=[], s=[], r=[])
    C = structure.C
    occ = np.asarray(d["occ"], float).reshape(C, structure.max_M, structure.max_K)
    shp = (structure.D, structure.max_M, structure.max_K)
    nn, ss, rr = (np.asarray(d[k], float).reshape(shp) for k in ("n", "s", "r"))
    for c in range(C):
        M, K = structure.channel_state_num[c], structure.num_gmm_component[c]
        ds = structure.dim_slice(c)
        out.occ.append(occ[c, :M, :K].copy())
        out.n.append(nn[ds, :M, :K].copy())
        out.s.append(ss[ds, :M, :K].copy())
        out.r.append(rr[ds, :M, :K].copy())
    return out

