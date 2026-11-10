"""Discriminative LSIM training with maximum mutual information (MMI), published recipe
(port of em_lsim_discriminative.m).

    result = em_lsim_discriminative((Y_train, y_train), init_models, max_iteration=8,
                                  validation=(Y_val, y_val), score_scale=1e-3,
                                  engine="torch", devices=("cuda:0", "cuda:1"))

* ``Y_*``: list of (D, T_n) arrays (NaN = missing); ``y_*``: 0-based class labels.
* ``init_models``: one LSIMParams per class (normally the class-wise EM models); they are
  projected into the compact domain, and that projected model is iteration 0.

Schedule: Jacobi -- every model's statistics come from the same scores. Each iteration applies
the full published update to every model (I-smoothing, Gaussian EBW with fixed E, weak-sense
discrete updates; Woodland & Povey 2002); there is no acceptance test, and ``max_iteration``
iterations are run (default 8) unless early stopping ends the run.

Stochastic mini-batches (``minibatches=K > 1``): one iteration is an epoch of K updates, each on
a class-stratified batch (all classes in every batch, fixed seed per epoch). Each batch update is
exactly the full-batch update applied to the batch (frame-unit weights N_batch a_n). The domain
comes from the full training set. The full-training objective is recorded after each epoch.

Validation (optional, ``validation=(Y_val, y_val)``): every iteration records the validation
balanced accuracy of argmax(kappa L + log pi), macro-F1, AUROC, AUPRC, max-F1, best-threshold BA and
the validation objective (two classes: detection score kappa (L_pos - L_neg) + log prior ratio).
The checkpoint (``res.models``) is the iteration with the highest ``selection_metric`` (default
'balanced_accuracy', as MATLAB; ties: higher validation objective, then earlier).
``early_stopping_patience=k`` stops once the selection metric has not improved by more than
``early_stopping_min_delta`` for k iterations.

History: one entry per iteration (0 = the projected initial models). ``step_tau``,
``backtracks`` and ``accepted`` follow the MATLAB history layout; the published update always
takes the full step (1, 0, True).

Long runs: ``state_path`` saves a resumable state every ``save_every`` iterations
(``em_lsim_discriminative(..., resume=True)`` continues it; ``max_iteration`` may be raised);
``log_path`` appends one JSON line of convergence diagnostics per iteration.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.learning.discriminative.domain import at_bounds, build_domain, pooled_feature_stats, project
from chmm_lsim.learning.discriminative.engines import make_engine
from chmm_lsim.learning.discriminative.metrics import score_metrics
from chmm_lsim.learning.discriminative.objective import ObjectiveConfig, WeightSpec, objective
from chmm_lsim.learning.discriminative.options import DiscOptions, default_max_iteration, make_options, options_to_dict
from chmm_lsim.learning.discriminative.updates import published_update
from chmm_lsim.models.lsim.params import LSIMParams

_STATE_VERSION = 1


class TrainingError(RuntimeError):
    pass


@dataclass
class TrainResult:
    models: list[LSIMParams]
    final_models: list[LSIMParams]
    initial_models: list[LSIMParams]
    history: dict[str, NDArray]
    diagnostics: dict = field(default_factory=dict)


def _fingerprint(Y, label) -> str:
    h = hashlib.sha1()
    h.update(np.asarray(label, np.int64).tobytes())
    h.update(np.array([y.shape[1] for y in Y], np.int64).tobytes())
    for y in (Y[0], Y[-1]):
        h.update(np.nan_to_num(y, nan=-7.25).tobytes())
    return h.hexdigest()


def _stratified_batches(label: NDArray, K: int, rng: np.random.Generator) -> list[NDArray]:
    M = int(label.max()) + 1
    parts = [[] for _ in range(K)]
    for k in range(M):
        idx = np.flatnonzero(label == k)
        if idx.size < K:
            raise TrainingError(f"class {k} has {idx.size} training sequences, fewer than minibatches={K}")
        for b, chunk in enumerate(np.array_split(rng.permutation(idx), K)):
            parts[b].append(chunk)
    return [np.sort(np.concatenate(p)) for p in parts]


def _param_change(p: LSIMParams, q: LSIMParams) -> tuple[float, float]:
    """Largest absolute change of any probability, largest |d mu| / sd."""
    dprob = max(np.max(np.abs(p.pi_0 - q.pi_0)), np.max(np.abs(p.transition - q.transition)),
                np.max(np.abs(p.coupling - q.coupling)))
    dmu = 0.0
    for a, b in zip(p.emissions, q.emissions):
        dprob = max(dprob, np.max(np.abs(a.weights - b.weights)))
        dmu = max(dmu, float(np.max(np.abs(a.mu - b.mu) / np.sqrt(a.sigma))))
    return float(dprob), dmu


class _Trainer:
    def __init__(self, opts: DiscOptions, train, init_models, validation, max_iteration):
        self.o = opts
        Y, y = train
        self.Y = list(Y)
        self.label = np.asarray(y, np.int64).ravel()
        if len(self.Y) != self.label.size:
            raise TrainingError("train: one label per sequence")
        self.M = len(init_models)
        if self.M < 2:
            raise TrainingError("discriminative training needs M >= 2 classes")
        if self.label.min() < 0 or self.label.max() >= self.M or np.any(np.bincount(self.label, minlength=self.M) < 1):
            raise TrainingError("labels must be 0..M-1 with every class present in the training data")
        self.N = len(self.Y)
        self.T = np.array([y_.shape[1] for y_ in self.Y], np.float64)
        self.st = init_models[0].structure
        for p in init_models:
            if p.structure != self.st:
                raise TrainingError("all init models must share one LSIMStructure")
            if not p.sigma_diag:
                raise TrainingError("full-covariance emissions are not supported (diagonal only, as in MATLAB)")
        if self.Y[0].shape[0] != self.st.D:
            raise TrainingError(f"observations have D={self.Y[0].shape[0]}, the models expect D={self.st.D}")
        self.lp = (np.log(np.full(self.M, 1.0 / self.M)) if opts.log_prior is None else opts.log_prior)
        if self.lp.size != self.M:
            raise TrainingError("log_prior must have one entry per class")
        self.generative = [p.copy() for p in init_models]
        self.has_val = validation is not None
        if self.has_val:
            Yv, yv = validation
            self.Yv = list(Yv)
            self.label_v = np.asarray(yv, np.int64).ravel()
            self.Tv = np.array([y_.shape[1] for y_ in self.Yv], np.float64)
        if not self.has_val and (opts.early_stopping_patience is not None or opts.selection_metric != "balanced_accuracy"):
            raise TrainingError("selection_metric / early stopping need validation data (validation=...)")
        if opts.selection_metric in ("max_f1", "ba_best_threshold") and self.M != 2:
            raise TrainingError(f"selection_metric {opts.selection_metric!r} needs exactly two classes")
        if not (0 <= opts.positive_class < self.M):
            raise TrainingError("positive_class must be a class index")
        self.max_iteration = default_max_iteration(opts) if max_iteration is None else int(max_iteration)
        self.engine = make_engine(self.st, opts)
        if hasattr(self.engine, "n_models"):
            self.engine.n_models = self.M
        self.pending = None   # counts of the current models, computed in the pass that scored them
        self.engine.register("train", self.Y)
        if self.has_val:
            self.engine.register("val", self.Yv)
        self.all_idx = np.arange(self.N)
        self.minibatch = opts.minibatches > 1
        self.t_scores = self.t_counts = self.t_update = 0.0
        self.n_score_passes = 0

    # ------------------------------------------------------------------ helpers
    def scores(self, models, name="train", idx=None):
        t0 = time.perf_counter()
        L = self.engine.scores(models, name, idx)
        self.t_scores += time.perf_counter() - t0
        self.n_score_passes += 1
        return L

    def fused(self, models, idx, N_stat: int):
        """Scores of idx under models and the weighted counts of the update that follows, in one
        forward-backward per model (the weights are row-local)."""
        t0 = time.perf_counter()
        lab = self.label[idx]
        spec = WeightSpec(self.cfg(N_stat), np.bincount(lab, minlength=self.M).astype(np.float64))
        L, cnts = self.engine.fused_counts(models, "train", idx, spec, lab, self.T[idx])
        self.t_counts += time.perf_counter() - t0
        self.n_score_passes += 1
        return L, cnts

    def cfg(self, N_stat: int) -> ObjectiveConfig:
        return ObjectiveConfig("mmi", self.lp, score_scale=self.o.score_scale, N_stat=N_stat)

    def validate(self, row: dict, itr: int, p) -> None:
        """Validation metrics of the iteration; checkpoint (selection_metric, ties: higher validation
        objective, then earlier) and the early-stopping counter."""
        o = self.o
        Lv = self.scores(p, "val")
        m = score_metrics(Lv, self.label_v, self.lp, o.score_scale, o.positive_class)
        Jv, _ = objective(Lv, self.label_v, self.Tv, self.cfg(len(self.Yv)))
        row.update(val_balanced_accuracy=m["balanced_accuracy"], val_macro_f1=m["macro_f1"], val_objective=Jv,
                   val_auroc=m["auroc"], val_auprc=m["auprc"], val_max_f1=m["max_f1"],
                   val_ba_best_threshold=m["ba_best_threshold"])
        v = Jv if o.selection_metric == "objective" else m[o.selection_metric]
        v = -np.inf if not np.isfinite(v) else float(v)
        row["val_selection_value"] = v
        b = self.best
        if v > b["value"] or (v == b["value"] and Jv > b["val_obj"]):
            self.best = {"iteration": itr, "p": [q.copy() for q in p], "value": v, "val_obj": Jv,
                         "bal_acc": m["balanced_accuracy"]}
        if v > self.es_best + o.early_stopping_min_delta:
            self.es_best = v
            self.since_best = 0
        else:
            self.since_best += 1

    def record(self, itr, J, J_old, parts, L, p, pinfo, secs, changes):
        o, M = self.o, self.M
        row = {"iteration": itr, "objective": J,
               "rel_change": (np.nan if J_old is None else (J - J_old) / max(1.0, abs(J_old))),
               "step_tau": [np.nan if itr == 0 else 1.0] * M, "backtracks": [0] * M, "accepted": [True] * M,
               "time_s": secs}
        own, other, bounds = [], [], []
        for k in range(M):
            own.append(float(np.mean(L[self.label == k, k])))
            other.append(float(np.mean(L[self.label != k, k])) if np.any(self.label != k) else np.nan)
            bounds.append(at_bounds(p[k], self.doms[k]))
        row.update(mean_own_score=own, mean_other_score=other,
                   frac_prob_at_floor=[b["prob_at_floor"] for b in bounds],
                   frac_var_at_lower=[b["var_at_lower"] for b in bounds],
                   frac_var_at_cap=[b["var_at_cap"] for b in bounds],
                   frac_mean_at_bound=[b["mean_at_bound"] for b in bounds])
        row["D_max"] = [pi.get("max_D", np.nan) if pi else np.nan for pi in pinfo]
        row["D_median"] = [pi.get("median_D", np.nan) if pi else np.nan for pi in pinfo]
        row["projection_changed"] = [pi.get("projection_changed_total", np.nan) if pi else np.nan for pi in pinfo]
        row["weak_sense_iterations"] = [pi.get("weak_sense_max_iterations", np.nan) if pi else np.nan for pi in pinfo]
        row["weak_sense_monotone"] = [bool(pi.get("weak_sense_monotone", True)) if pi else True for pi in pinfo]
        ex = parts.extra
        row.update(frac_q_confident_wrong=ex["frac_q_confident_wrong"], frac_q_uncertain=ex["frac_q_uncertain"],
                   frac_q_confident_right=ex["frac_q_confident_right"])
        mt = score_metrics(L, self.label, self.lp, o.score_scale, o.positive_class)
        row.update(train_balanced_accuracy=mt["balanced_accuracy"], train_auroc=mt["auroc"], train_auprc=mt["auprc"],
                   train_max_f1=mt["max_f1"])
        row["param_change_prob"] = [c[0] for c in changes]
        row["param_change_mu_sd"] = [c[1] for c in changes]
        row.update(time_scores_s=self.t_scores, time_counts_s=self.t_counts, time_update_s=self.t_update,
                   n_score_passes=self.n_score_passes)
        self.t_scores = self.t_counts = self.t_update = 0.0
        self.n_score_passes = 0
        if self.has_val:
            self.validate(row, itr, p)
        self.rows.append(row)
        if o.log_path:
            with open(o.log_path, "a") as fh:
                fh.write(json.dumps(_jsonable(row)) + "\n")
        if o.verbose:
            msg = f"itr {itr:3d}  J {J:.6f}"
            if itr > 0:
                msg += f"  rel {row['rel_change']:+.2e}"
            if self.has_val:
                msg += (f"  val BA {row['val_balanced_accuracy']:.4f} AUROC {row['val_auroc']:.4f} "
                        f"AUPRC {row['val_auprc']:.4f} maxF1 {row['val_max_f1']:.4f}")
            msg += f"  {secs:.1f}s"
            print(msg, flush=True)
        if o.iteration_hook is not None:
            o.iteration_hook(itr, p, {"J": J, "row": row, "parts": parts, "proposal_info": pinfo})

    # ------------------------------------------------------------------ setup / resume
    def setup(self) -> None:
        o = self.o
        stats = pooled_feature_stats(self.Y)
        self.doms = [build_domain(self.Y, g, prob_floor=o.prob_floor, mean_margin=o.mean_margin,
                                  var_lo_frac=o.var_lo_frac, var_hi_frac=o.var_hi_frac, feature_stats=stats)
                     for g in self.generative]
        self.p = [project(g, d) for g, d in zip(self.generative, self.doms)]
        self.initial = [q.copy() for q in self.p]
        if not self.minibatch and self.max_iteration > 0:
            self.L, self.pending = self.fused(self.p, self.all_idx, self.N)
        else:
            self.L = self.scores(self.p)
        self.J, self.parts = objective(self.L, self.label, self.T, self.cfg(self.N))
        if not np.isfinite(self.J):
            raise TrainingError("the (projected) initial models give a non-finite objective; "
                                "some training sequence has zero score")
        self.rows: list[dict] = []
        self.best = {"iteration": 0, "p": [q.copy() for q in self.p], "value": -np.inf, "val_obj": -np.inf,
                     "bal_acc": -np.inf}
        self.es_best, self.since_best = -np.inf, 0
        self.start_itr = 1
        self.stop_reason = "max_iteration"
        self.record(0, self.J, None, self.parts, self.L, self.p, [None] * self.M, 0.0, [(0.0, 0.0)] * self.M)

    def state(self, itr: int) -> dict:
        return {"version": _STATE_VERSION, "iteration": itr, "p": self.p, "L": self.L,
                "rows": self.rows, "best": self.best,
                "es_best": self.es_best, "since_best": self.since_best, "doms": self.doms, "initial": self.initial,
                "generative": self.generative, "stop_reason": self.stop_reason,
                "fingerprint": _fingerprint(self.Y, self.label), "options": options_to_dict(self.o)}

    def save(self, itr: int) -> None:
        if not self.o.state_path:
            return
        tmp = self.o.state_path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(self.state(itr), fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.o.state_path)

    def resume(self) -> bool:
        path = self.o.state_path
        if not path or not os.path.exists(path):
            return False
        with open(path, "rb") as fh:
            s = pickle.load(fh)
        if s.get("version") != _STATE_VERSION:
            raise TrainingError(f"{path}: unsupported state version {s.get('version')}")
        if s["fingerprint"] != _fingerprint(self.Y, self.label):
            raise TrainingError(f"{path}: the saved state was trained on different data")
        now = options_to_dict(self.o)
        fixed = [k for k in now if k not in ("verbose", "engine", "n_jobs", "numpy_backend", "devices", "chunk_size",
                                              "state_path", "log_path", "save_every")]
        diff = [k for k in fixed if now[k] != s["options"].get(k)]
        if diff:
            raise TrainingError(f"{path}: options differ from the saved run: {diff}")
        self.p, self.L, self.rows, self.best = s["p"], s["L"], s["rows"], s["best"]
        self.es_best, self.since_best = s["es_best"], s["since_best"]
        self.doms, self.initial = s["doms"], s["initial"]
        self.generative, self.stop_reason = s["generative"], s["stop_reason"]
        self.J, self.parts = objective(self.L, self.label, self.T, self.cfg(self.N))
        self.start_itr = s["iteration"] + 1
        if self.stop_reason != "max_iteration":
            self.start_itr = self.max_iteration + 1      # the saved run had already stopped
        if self.o.verbose:
            print(f"resumed {path} after iteration {s['iteration']} (J {self.J:.6f})", flush=True)
        return True

    # ------------------------------------------------------------------ one update on an index set
    def update(self, idx: NDArray, p: list[LSIMParams], L_idx: NDArray, pre_counts=None):
        """One Jacobi update of all models from the sequences idx (scores L_idx under p).
        ``pre_counts``: the weighted counts of idx under p when already computed (fused pass).
        Returns the new models and the per-model update infos."""
        o, M = self.o, self.M
        _J_idx, parts = objective(L_idx, self.label[idx], self.T[idx], self.cfg(len(idx)))
        props, pinfo = [], []
        for k in range(M):
            t0 = time.perf_counter()
            if pre_counts is not None:
                num, den = pre_counts[k]
            else:
                W = np.column_stack([parts.W_num[:, k], parts.W_den[:, k]])
                num, den = self.engine.counts(p[k], "train", idx, W)
            t1 = time.perf_counter()
            q, info = published_update(p[k], num, den, self.doms[k], ebw_E=o.ebw_E, ismooth_tau=o.ismooth_tau,
                                       ismooth_backoff=o.ismooth_backoff,
                                       ismooth_tau_discrete=o.ismooth_tau_discrete,
                                       weak_sense_iters=o.weak_sense_iters, weak_sense_tol=o.weak_sense_tol,
                                       update_blocks=o.update_blocks)
            self.t_counts += t1 - t0
            self.t_update += time.perf_counter() - t1
            props.append(q)
            pinfo.append(info)
        return props, pinfo

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        o = self.o
        for itr in range(self.start_itr, self.max_iteration + 1):
            t_itr = time.perf_counter()
            J_old = self.J
            p_before = self.p
            if not self.minibatch:
                pre, self.pending = self.pending, None
                self.p, pinfo = self.update(self.all_idx, self.p, self.L, pre)
                if itr < self.max_iteration:
                    # the scores of the new models are needed anyway: get the next update's counts too
                    self.L, self.pending = self.fused(self.p, self.all_idx, self.N)
                else:
                    self.L = self.scores(self.p)
            else:
                rng = np.random.default_rng([o.minibatch_seed, itr])
                pinfo = [None] * self.M
                for b in _stratified_batches(self.label, o.minibatches, rng):
                    Lb, pre = self.fused(self.p, b, len(b))
                    self.p, pinfo = self.update(b, self.p, Lb, pre)
                self.L = self.scores(self.p)
            self.J, self.parts = objective(self.L, self.label, self.T, self.cfg(self.N))
            changes = [_param_change(a, b) for a, b in zip(p_before, self.p)]
            self.record(itr, self.J, J_old, self.parts, self.L, self.p, pinfo, time.perf_counter() - t_itr, changes)
            done = False
            if self.has_val and o.early_stopping_patience is not None and self.since_best >= o.early_stopping_patience:
                self.stop_reason = (f"early stopping: validation {o.selection_metric} did not improve by more than "
                                    f"{o.early_stopping_min_delta:g} for {o.early_stopping_patience} iterations")
                done = True
            if done or itr % o.save_every == 0 or itr == self.max_iteration:
                self.save(itr)
            if done:
                break

    def result(self) -> TrainResult:
        hist = _history_arrays(self.rows)
        last = int(hist["iteration"][-1])
        if self.has_val:
            models, ck, rule = self.best["p"], self.best["iteration"], \
                f"validation {self.o.selection_metric}, ties by validation objective, then earliest"
        else:
            models, ck, rule = self.p, last, "final iterate (no validation data supplied)"
        diag = {"objective": self.o.objective, "stop_reason": self.stop_reason, "iterations_run": last,
                "checkpoint_iteration": ck, "checkpoint_rule": rule,
                "checkpoint_validation_value": self.best["value"] if self.has_val else None,
                "domains": self.doms, "generative_models": self.generative, "options": options_to_dict(self.o),
                "engine": self.engine.name,
                "score_semantics": ("exact HMM log-likelihood (C = 1)" if self.st.C == 1 else
                                    "approximate LSIM forward score (per-channel factorized filter), "
                                    "not the exact joint log-likelihood")}
        return TrainResult(models=[q.copy() for q in models], final_models=[q.copy() for q in self.p],
                           initial_models=self.initial, history=hist, diagnostics=diag)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


def _history_arrays(rows: list[dict]) -> dict[str, NDArray]:
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    out = {}
    for k in keys:
        vals = [r.get(k, np.nan) for r in rows]
        try:
            out[k] = np.array(vals, dtype=float)
        except (TypeError, ValueError):
            out[k] = np.array(vals, dtype=object)
    return out


def em_lsim_discriminative(train, init_models: list[LSIMParams], *, max_iteration: int | None = None,
                         validation=None, resume: bool = False, **options) -> TrainResult:
    """Train M class models discriminatively with MMI (published recipe). See the module docstring."""
    opts = make_options(**options)
    tr = _Trainer(opts, train, init_models, validation, max_iteration)
    try:
        if not (resume and tr.resume()):
            if opts.log_path and os.path.exists(opts.log_path) and not resume:
                os.remove(opts.log_path)
            tr.setup()
        tr.run()
        return tr.result()
    finally:
        tr.engine.close()


def lsim_discriminative_predict(models: list[LSIMParams], Y, *, log_prior=None, score_scale: float = 1.0, engine: str = "numpy", **eng):
    """Scores L (N x M), argmax(kappa L + log pi) predictions and normalized scores."""
    e = make_engine(models[0].structure, make_options(engine=engine, **eng))
    try:
        e.register("x", list(Y))
        L = e.scores(models, "x")
    finally:
        e.close()
    M = len(models)
    lp = np.log(np.full(M, 1.0 / M)) if log_prior is None else np.asarray(log_prior, float).ravel()
    Z = (score_scale * L if score_scale != 1 else L) + lp[None, :]
    pred = np.argmax(Z, axis=1)
    Zs = np.exp(Z - Z.max(axis=1, keepdims=True))
    return L, pred, Zs / Zs.sum(axis=1, keepdims=True)


# Names used before the MATLAB-matching names above; kept so existing code keeps working.
train_discriminative = em_lsim_discriminative
predict = lsim_discriminative_predict
