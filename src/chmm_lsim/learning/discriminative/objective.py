"""The MMI objective and its per-sequence statistic weights (port of lsim_disc_objective.m).

MMI (Bahl et al. 1986; Woodland & Povey 2002), published recipe::

    q(n,k) = softmax_k(kappa L(n,k) + log pi_k),  a_n = 1/(M N_{y_n}),  J = sum_n a_n log q(n,y_n)
    W.num(n,y) = N a_n,  W.den(n,k) = N a_n q(n,k)   (frame units; kappa only inside q)

Any non-finite term log q(n, y_n) gives J = -inf; -inf COMPETING scores are tolerated and
only make q = 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class ObjectiveConfig:
    objective: str
    log_prior: NDArray[np.float64]
    score_scale: float = 1.0
    N_stat: int = 0


@dataclass
class ObjectiveParts:
    W_num: NDArray[np.float64]
    W_den: NDArray[np.float64]
    a: NDArray[np.float64]
    extra: dict = field(default_factory=dict)


def objective(L: NDArray[np.float64], label: NDArray[np.int64], T: NDArray[np.float64],
              cfg: ObjectiveConfig, class_counts=None) -> tuple[float, ObjectiveParts]:
    """``label`` is 0-based. Returns J and the statistic weights (N x M, >= 0).
    ``T`` (sequence lengths) is part of the weight interface of the compute engines; MMI does not
    use it. ``class_counts`` (M,): the N_y of the full set when L holds only a subset of its rows
    (the weights are then exactly those rows of the full set's weights; J is not meaningful)."""
    L = np.asarray(L, dtype=np.float64)
    N, M = L.shape
    label = np.asarray(label, dtype=np.int64).ravel()
    lp = np.asarray(cfg.log_prior, dtype=np.float64).ravel()[None, :]
    rows = np.arange(N)
    Np = (np.bincount(label, minlength=M).astype(np.float64) if class_counts is None
          else np.asarray(class_counts, np.float64))
    if cfg.objective == "mmi":
        return _mmi(L, label, cfg, lp, rows, N, M, Np)
    raise ValueError(f"objective must be 'mmi', got {cfg.objective!r}")


@dataclass
class WeightSpec:
    """Picklable, row-local statistic weights: rows of ``objective(L_full, ...)``'s W from the rows'
    own scores and the full set's constants (class counts, kappa, N_stat). Lets a compute
    engine weight each work unit's statistics in the same pass that produced its scores."""
    cfg: ObjectiveConfig
    class_counts: NDArray[np.float64]

    def weights(self, L, label, T):
        _J, parts = objective(L, label, T, self.cfg, class_counts=self.class_counts)
        return parts.W_num, parts.W_den


def _mmi(L, label, cfg, lp, rows, N, M, Npc):
    Z = cfg.score_scale * L + lp if cfg.score_scale != 1 else L + lp
    mx = np.max(Z, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", over="ignore"):
        lse = mx + np.log(np.sum(np.exp(Z - mx), axis=1, keepdims=True))
    a = 1.0 / (M * Npc[label])
    terms = Z[rows, label] - lse[:, 0]
    J = -np.inf if np.any(~np.isfinite(terms)) else float(np.sum(a * terms))
    with np.errstate(invalid="ignore", over="ignore"):
        q = np.exp(Z - lse)
    q[~np.isfinite(q)] = 0.0
    ca = cfg.N_stat * a
    W_num = np.zeros((N, M))
    W_num[rows, label] = ca
    W_den = ca[:, None] * q
    qy = q[rows, label]
    extra = {"q": q, "log_posterior_own": terms,
             "frac_q_confident_wrong": float(np.mean(qy < 0.01)),
             "frac_q_uncertain": float(np.mean((qy >= 0.01) & (qy <= 0.99))),
             "frac_q_confident_right": float(np.mean(qy > 0.99))}
    return J, ObjectiveParts(W_num, W_den, a, extra)


def balanced_accuracy(L: NDArray[np.float64], label: NDArray[np.int64], log_prior, score_scale: float = 1.0):
    """argmax_k(kappa L + log pi) balanced accuracy and macro-F1 (lsim_disc_metrics.m)."""
    L = np.asarray(L, float)
    M = L.shape[1]
    Z = (score_scale * L if score_scale != 1 else L) + np.asarray(log_prior, float).ravel()[None, :]
    pred = np.argmax(Z, axis=1)
    label = np.asarray(label, np.int64).ravel()
    conf = np.zeros((M, M))
    np.add.at(conf, (label, pred), 1)
    support = conf.sum(axis=1)
    pc = conf.sum(axis=0)
    tp = np.diag(conf)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = tp / support
        precision = np.where(pc > 0, tp / np.where(pc > 0, pc, 1), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    present = support > 0
    return float(np.mean(recall[present])), float(np.mean(f1[present]))
