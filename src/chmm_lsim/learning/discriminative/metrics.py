"""Classification metrics for LSIM class scores (no scikit-learn dependency).

Scores: ``L`` (N x M) are per-class LSIM scores (``predict`` / engine ``scores``). The decision
rule is argmax_k(kappa L + log pi). For two classes the detection score of the positive class
(``positive=1``, e.g. seizure) is the log-likelihood ratio

    s_n = kappa (L[n, 1] - L[n, 0]) + log pi_1 - log pi_0,

and the threshold-free metrics (AUROC, AUPRC) do not depend on kappa or the priors. For M > 2,
AUROC / AUPRC are one-vs-rest macro averages over the normalized scores; max-F1 and the
best-threshold BA are defined for M = 2 only (NaN otherwise).

* :func:`binary_metrics` -- AUROC, AUPRC (= average precision, as sklearn), max-F1 and the best
  balanced accuracy over all thresholds, with the thresholds attaining them;
* :func:`tune_threshold` -- threshold maximizing BA or F1 on a (validation) set;
* :func:`threshold_metrics` -- BA, F1, sensitivity, specificity, precision at a given threshold;
* :func:`score_metrics` -- all of the above from an N x M score matrix (+ argmax BA / macro-F1);
* :func:`evaluate_models` -- score models on several sets and report every metric, with the
  decision threshold tuned on the validation set and applied unchanged to the others.

Threshold convention: a sequence is called positive when ``s >= threshold``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

SELECTION_METRICS = ("balanced_accuracy", "auroc", "auprc", "max_f1", "ba_best_threshold", "objective")


def _sweep(s: NDArray, y: NDArray):
    """Confusion counts at every distinct threshold (descending), positive when s >= t."""
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order].astype(np.float64)
    last = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s_sorted) - 1]  # last index of every tie block
    tp = np.cumsum(y_sorted)[last]
    fp = (last + 1) - tp
    return s_sorted[last], tp, fp


def binary_metrics(score, label) -> dict:
    """``score``: higher = more positive; ``label``: 1 = positive, 0 = negative (bool or int)."""
    s = np.asarray(score, dtype=np.float64).ravel()
    y = np.asarray(label).ravel().astype(bool)
    s = np.where(np.isnan(s), -np.inf, s)                 # an unscorable sequence ranks lowest
    s = np.clip(s, -np.finfo(float).max, np.finfo(float).max)
    P, N = int(y.sum()), int((~y).sum())
    out = {"auroc": np.nan, "auprc": np.nan, "max_f1": np.nan, "max_f1_threshold": np.nan,
           "ba_best_threshold": np.nan, "best_ba_threshold": np.nan, "n_pos": P, "n_neg": N}
    if P == 0 or N == 0:
        return out
    thr, tp, fp = _sweep(s, y)
    # AUROC: trapezoid over the ROC (ties handled by the tie blocks) = Mann-Whitney U with average ranks
    tpr = np.r_[0.0, tp / P]
    fpr = np.r_[0.0, fp / N]
    out["auroc"] = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    # AUPRC: average precision sum_k (R_k - R_{k-1}) P_k (sklearn's definition)
    precision = tp / (tp + fp)
    recall = tp / P
    out["auprc"] = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    # max-F1 and best BA over every threshold (plus the all-negative threshold +inf)
    f1 = 2 * tp / (2 * tp + fp + (P - tp))
    j = int(np.argmax(f1))
    out["max_f1"], out["max_f1_threshold"] = float(f1[j]), float(thr[j])
    ba = 0.5 * (tp / P + (N - fp) / N)
    k = int(np.argmax(ba))
    if ba[k] >= 0.5:
        out["ba_best_threshold"], out["best_ba_threshold"] = float(ba[k]), float(thr[k])
    else:
        out["ba_best_threshold"], out["best_ba_threshold"] = 0.5, float("inf")
    return out


def tune_threshold(score, label, criterion: str = "ba") -> float:
    """Threshold maximizing balanced accuracy (``'ba'``) or F1 (``'f1'``) on the given set."""
    m = binary_metrics(score, label)
    if criterion == "ba":
        return m["best_ba_threshold"]
    if criterion == "f1":
        return m["max_f1_threshold"]
    raise ValueError("criterion must be 'ba' or 'f1'")


def threshold_metrics(score, label, threshold: float) -> dict:
    s = np.asarray(score, dtype=np.float64).ravel()
    s = np.where(np.isnan(s), -np.inf, s)
    y = np.asarray(label).ravel().astype(bool)
    pred = s >= threshold
    tp = int(np.sum(pred & y)); fp = int(np.sum(pred & ~y)); fn = int(np.sum(~pred & y)); tn = int(np.sum(~pred & ~y))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"threshold": float(threshold), "balanced_accuracy": 0.5 * (sens + spec), "f1": f1,
            "sensitivity": sens, "specificity": spec, "precision": prec, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def detection_score(L, log_prior=None, score_scale: float = 1.0, positive: int = 1) -> NDArray:
    """Two-class detection score s = kappa (L_pos - L_neg) + log pi_pos - log pi_neg."""
    L = np.asarray(L, dtype=np.float64)
    if L.shape[1] != 2:
        raise ValueError("detection_score needs exactly two classes")
    lp = np.zeros(2) if log_prior is None else np.asarray(log_prior, float).ravel()
    neg = 1 - positive
    return score_scale * (L[:, positive] - L[:, neg]) + (lp[positive] - lp[neg])


def score_metrics(L, label, log_prior=None, score_scale: float = 1.0, positive: int = 1) -> dict:
    """Every metric from an N x M score matrix and 0-based labels."""
    L = np.asarray(L, dtype=np.float64)
    label = np.asarray(label, np.int64).ravel()
    N, M = L.shape
    lp = np.log(np.full(M, 1.0 / M)) if log_prior is None else np.asarray(log_prior, float).ravel()
    Z = (score_scale * L if score_scale != 1 else L) + lp[None, :]
    Zf = np.where(np.isnan(Z), -np.inf, Z)
    pred = np.argmax(Zf, axis=1)
    rec, f1s = [], []
    for k in range(M):
        tp = np.sum((pred == k) & (label == k)); sup = np.sum(label == k); pc = np.sum(pred == k)
        if sup == 0:
            continue
        r = tp / sup
        p = tp / pc if pc else 0.0
        rec.append(r)
        f1s.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
    out = {"balanced_accuracy": float(np.mean(rec)), "macro_f1": float(np.mean(f1s)),
           "accuracy": float(np.mean(pred == label))}
    if M == 2:
        b = binary_metrics(detection_score(L, lp, score_scale, positive), label == positive)
        out.update({k: b[k] for k in ("auroc", "auprc", "max_f1", "max_f1_threshold", "ba_best_threshold",
                                      "best_ba_threshold")})
    else:
        mx = Zf.max(axis=1, keepdims=True)
        P = np.exp(Zf - mx)
        P /= P.sum(axis=1, keepdims=True)
        per = [binary_metrics(P[:, k], label == k) for k in range(M) if np.any(label == k) and np.any(label != k)]
        out.update({"auroc": float(np.mean([b["auroc"] for b in per])), "auprc": float(np.mean([b["auprc"] for b in per])),
                    "max_f1": np.nan, "max_f1_threshold": np.nan, "ba_best_threshold": np.nan,
                    "best_ba_threshold": np.nan})
    return out


def evaluate_models(models, sets: dict, *, tune_on: str = "val", log_prior=None, score_scale: float = 1.0,
                    positive: int = 1, threshold_criterion: str = "ba", engine: str = "numpy", **engine_opts) -> dict:
    """Score ``models`` on every set and report every metric.

    ``sets``: {name: (Y_list, labels_0based)}, e.g. {"train": ..., "val": ..., "test": ...}.
    For two classes the decision threshold is tuned on ``sets[tune_on]`` (``threshold_criterion``
    'ba' or 'f1') and applied unchanged to every set (``*_at_tuned``). Returns
    {name: metrics dict, ..., "threshold": t, "scores": {name: L}}.
    """
    from chmm_lsim.learning.discriminative.engines import make_engine
    from chmm_lsim.learning.discriminative.options import make_options

    e = make_engine(models[0].structure, make_options(engine=engine, **engine_opts))
    try:
        L = {}
        for name, (Y, _y) in sets.items():
            e.register(name, list(Y))
            L[name] = e.scores(models, name)
    finally:
        e.close()
    out = {"scores": L}
    M = len(models)
    lp = np.log(np.full(M, 1.0 / M)) if log_prior is None else np.asarray(log_prior, float).ravel()
    thr = None
    if M == 2 and tune_on in sets:
        thr = tune_threshold(detection_score(L[tune_on], lp, score_scale, positive),
                             np.asarray(sets[tune_on][1]).ravel() == positive, threshold_criterion)
    out["threshold"] = thr
    for name, (_Y, y) in sets.items():
        m = score_metrics(L[name], y, lp, score_scale, positive)
        if thr is not None:
            t = threshold_metrics(detection_score(L[name], lp, score_scale, positive),
                                  np.asarray(y).ravel() == positive, thr)
            m.update({f"{k}_at_tuned": v for k, v in t.items() if k != "threshold"})
        out[name] = m
    return out
