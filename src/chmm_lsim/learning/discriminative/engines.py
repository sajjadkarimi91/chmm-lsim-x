"""Compute engines: per-sequence scores and weighted numerator / denominator counts.

Both engines expose the same two operations on a registered dataset (a list of (D, T) arrays):

* ``scores(params_list, name, idx)`` -> (n, M) forward scores L(n, k) (the ``sed``/``fast``
  forward pass, MATLAB ``P_O_model``); a failing or NaN sequence gives NaN (a rejected trial point);
* ``counts(params, name, idx, W)`` -> R :class:`DiscCounts`, ``sum_n W[n, r] * counts_n`` under
  ``params`` (``sed`` posteriors). Sequences whose weights are all zero are skipped.

Determinism: sequences are split into FIXED work units of ``chunk_size`` sequences, partial sums
are computed per unit and combined in unit order. With a given ``chunk_size`` results therefore do
not depend on ``n_jobs`` or on the devices used (TorchEngine's automatic chunk size depends on the
number of worker processes; pass ``chunk_size`` for identical results across device lists).

* :class:`NumpyEngine` -- CPU, the package's numba (or numpy) kernels, optional process pool
  (``spawn``; every worker receives the datasets once at start-up, never per task).
* :class:`TorchEngine` -- the batched torch kernels of ``chmm_lsim.backends.torch_batched``
  (float64) on one device in-process, or on several devices (``cuda:0``, ``cuda:1``, ...; a
  repeated entry = several processes on one GPU) with one spawned worker process per entry.
  Work units are assigned round-robin; padded inputs stay resident on their device.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.typing import NDArray

from chmm_lsim.backends.base import run_backend, run_backend_loglik
from chmm_lsim.backends.numpy_ import build_W
from chmm_lsim.core.constants import FbConstants
from chmm_lsim.learning.discriminative.counts import DiscCounts, NonfiniteStatistics, sequence_counts
from chmm_lsim.models.lsim.params import LSIMParams
from chmm_lsim.core.structure import LSIMStructure

_DATASETS: dict[str, list[NDArray[np.float64]]] = {}   # set in each worker by _init_worker


def _init_worker(datasets: dict) -> None:
    import os as _os

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        _os.environ[v] = "1"
    _DATASETS.update(datasets)


def _check_counts(cs: list[DiscCounts]) -> None:
    for c in cs:
        if any(not np.all(np.isfinite(a)) for a in c.arrays()):
            raise NonfiniteStatistics("non-finite accumulated statistics; some weighted sequence has zero mass")


def _units(idx: NDArray[np.int64], chunk: int) -> list[NDArray[np.int64]]:
    return [idx[i:i + chunk] for i in range(0, len(idx), chunk)]


# ============================================================================ NumPy / Numba
def _resolve_numpy_backend(name: str) -> str:
    if name != "auto":
        return name
    try:
        import numba  # noqa: F401
        return "numba"
    except ImportError:
        return "numpy"


def _np_score_unit(key, unit, params_list, fbc, backend):
    Y = _DATASETS[key]
    out = np.empty((len(unit), len(params_list)))
    for j, n in enumerate(unit):
        for k, p in enumerate(params_list):
            try:
                out[j, k] = run_backend_loglik(backend, fbc.variant, p.structure, p, Y[n], fbc, validate="off")
            except Exception:
                out[j, k] = np.nan
    return out


def _np_counts_unit(key, unit, Wu, params, fbc, backend):
    Y = _DATASETS[key]
    st = params.structure
    R = Wu.shape[1]
    acc = [DiscCounts.zeros(st) for _ in range(R)]
    Wk = build_W(params.transition, params.coupling, st.block_of_state)
    for j, n in enumerate(unit):
        if not np.any(Wu[j] != 0):
            continue
        try:
            post = run_backend(backend, fbc.variant, st, params, Y[n], fbc, validate="off")
        except Exception as e:  # kernel failure: the model gives this sequence no mass
            raise NonfiniteStatistics(f"forward-backward failed on sequence {n}: {e}") from e
        sc = sequence_counts(st, post, Y[n], Wk)
        for r in range(R):
            if Wu[j, r] > 0:
                acc[r].add_(sc, Wu[j, r])
    return acc


def _np_fused_unit(key, unit, params_list, spec, lab, Tn, fbc, backend):
    """One unit, one forward-backward per model and sequence: scores, row-local weights, counts."""
    Y = _DATASETS[key]
    st = params_list[0].structure
    M = len(params_list)
    Ws = [build_W(p.transition, p.coupling, st.block_of_state) for p in params_list]
    acc = [[DiscCounts.zeros(st) for _ in range(2)] for _ in range(M)]
    L = np.empty((len(unit), M))
    for j, n in enumerate(unit):
        posts = []
        for k, p in enumerate(params_list):
            try:
                post = run_backend(backend, fbc.variant, st, p, Y[n], fbc, validate="off")
            except Exception as e:
                raise NonfiniteStatistics(f"forward-backward failed on sequence {n}: {e}") from e
            posts.append(post)
            L[j, k] = post.loglik
        W_num, W_den = spec.weights(L[j:j + 1], lab[j:j + 1], Tn[j:j + 1])
        for k in range(M):
            w = (W_num[0, k], W_den[0, k])
            if w[0] == 0 and w[1] == 0:
                continue
            sc = sequence_counts(st, posts[k], Y[n], Ws[k])
            for r in range(2):
                if w[r] > 0:
                    acc[k][r].add_(sc, w[r])
    return L, acc


class NumpyEngine:
    def __init__(self, structure: LSIMStructure, fbc: FbConstants, *, n_jobs: int = 1, backend: str = "auto",
                 chunk_size: int | None = None):
        self.structure = structure
        self.fbc = fbc
        self.backend = _resolve_numpy_backend(backend)
        self.chunk = chunk_size or 32
        self.n_jobs = n_jobs
        self._keys: dict[str, str] = {}
        self._pool = None
        self.name = f"numpy[{self.backend}, n_jobs={n_jobs}]"

    def register(self, name: str, Y: list[NDArray[np.float64]]) -> None:
        key = f"{id(self)}:{name}"
        _DATASETS[key] = [np.ascontiguousarray(y, dtype=np.float64) for y in Y]
        self._keys[name] = key
        if self._pool is not None:       # workers must see the new dataset: restart the pool
            self._pool.shutdown()
            self._pool = None

    def size(self, name: str) -> int:
        return len(_DATASETS[self._keys[name]])

    def lengths(self, name: str) -> NDArray[np.int64]:
        return np.array([y.shape[1] for y in _DATASETS[self._keys[name]]])

    def data(self, name: str) -> list[NDArray[np.float64]]:
        return _DATASETS[self._keys[name]]

    def _map(self, fn, tasks):
        if self.n_jobs == 1:
            return [fn(*t) for t in tasks]
        if self._pool is None:
            # spawn: workers start as clean interpreters (fork of a process
            # with numba / torch threads can deadlock); each worker receives the datasets once
            mine = {k: _DATASETS[k] for k in self._keys.values()}
            self._pool = ProcessPoolExecutor(self.n_jobs, mp_context=mp.get_context("spawn"),
                                             initializer=_init_worker, initargs=(mine,))
        return list(self._pool.map(fn, *zip(*tasks)))

    def scores(self, params_list: list[LSIMParams], name: str, idx=None) -> NDArray[np.float64]:
        key = self._keys[name]
        idx = np.arange(self.size(name)) if idx is None else np.asarray(idx, np.int64)
        tasks = [(key, u, params_list, self.fbc, self.backend) for u in _units(idx, self.chunk)]
        parts = self._map(_np_score_unit, tasks)
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, len(params_list)))

    def counts(self, params: LSIMParams, name: str, idx, W: NDArray[np.float64]) -> list[DiscCounts]:
        key = self._keys[name]
        idx = np.asarray(idx, np.int64)
        W = np.asarray(W, np.float64)
        if np.any(W < 0) or not np.all(np.isfinite(W)):
            raise ValueError("statistic weights must be finite and nonnegative")
        units = _units(np.arange(len(idx)), self.chunk)
        tasks = [(key, idx[u], W[u], params, self.fbc, self.backend) for u in units]
        parts = self._map(_np_counts_unit, tasks)
        R = W.shape[1]
        total = [DiscCounts.zeros(self.structure) for _ in range(R)]
        for part in parts:
            for r in range(R):
                total[r].add_(part[r])
        _check_counts(total)
        return total

    def fused_counts(self, params_list, name, idx, spec, label, T):
        """Scores and weighted counts in one forward-backward per model and sequence (see TorchEngine)."""
        key = self._keys[name]
        idx = np.asarray(idx, np.int64)
        label = np.asarray(label, np.int64)
        T = np.asarray(T, np.float64)
        units = _units(np.arange(len(idx)), self.chunk)
        tasks = [(key, idx[u], params_list, spec, label[u], T[u], self.fbc, self.backend) for u in units]
        parts = self._map(_np_fused_unit, tasks)
        L = np.concatenate([pt[0] for pt in parts], axis=0) if parts else np.zeros((0, len(params_list)))
        total = [[DiscCounts.zeros(self.structure) for _ in range(2)] for _ in params_list]
        for _L, acc in parts:
            for k in range(len(params_list)):
                for r in range(2):
                    total[k][r].add_(acc[k][r])
        for t in total:
            _check_counts(t)
        return L, total

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        for key in self._keys.values():
            _DATASETS.pop(key, None)


# ============================================================================ Torch
class _TorchCore:
    """Single-device torch work: per-unit scores and weighted counts. Padded units stay resident on
    the device (LRU cache keyed by the unit's dataset indices)."""

    def __init__(self, structure: LSIMStructure, fbc: FbConstants, device: str, cache_entries: int = 64):
        import torch

        from chmm_lsim.backends import torch_batched as tb

        self.torch, self.tb = torch, tb
        self.structure, self.fbc = structure, fbc
        self.dtype = torch.float64
        dev = torch.device(device)
        if dev.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"device {device!r} requested but CUDA is not available")
            if dev.index is None:
                dev = torch.device("cuda", torch.cuda.current_device())
            if dev.index >= torch.cuda.device_count():
                raise RuntimeError(f"device {device!r} requested but only {torch.cuda.device_count()} CUDA devices exist")
            torch.cuda.set_device(dev)
        self.device = dev
        self.data: dict[str, list[NDArray[np.float64]]] = {}
        self.cache: OrderedDict = OrderedDict()
        self.cache_entries = cache_entries

    def register(self, name, Y):
        self.data[name] = Y
        for k in [k for k in self.cache if k[0] == name]:
            del self.cache[k]

    @property
    def gemm_poise(self) -> bool:
        return self.fbc.variant == "poise" and not self.fbc.poise_refinement

    def estimate_chunk(self, N, T_max, n_models=1):
        st = self.structure
        if self.gemm_poise:   # O(S T) memory per sequence: no (Bn, S, S) temporaries
            from chmm_lsim.learning.discriminative.kernels import poise_chunk_size

            return poise_chunk_size(self.torch, self.device, st.S, T_max, st.max_K, st.D, n_models)
        n = self.tb.estimate_chunk_size(N, st.S, T_max, st.max_K, device=self.device, dtype=self.dtype,
                                        max_memory_bytes=None, D=st.D, max_d=st.max_d)
        return max(1, n // (2 * n_models))    # the counts reduction holds a few extra (Bn, M, K, T) tensors

    def _fb(self, ker, Yt, lt, seq_idx):
        """Forward-backward of one unit: alpha, beta, gamma, scale, B, B_comp, loglik."""
        if self.gemm_poise:
            from chmm_lsim.learning.discriminative.kernels import forward_backward_poise_gemm

            return forward_backward_poise_gemm(self.torch, self.tb, ker, Yt, lt)
        alpha, beta, gamma, scale, B, Bc, _ls, ll, _d = self.tb._compute_forward_backward_device(
            ker, Yt, self.fbc, seq_idx, lt, check_all=False, gz_form="block")
        return alpha, beta, gamma, scale, B, Bc, ll

    def _tensors(self, name, unit):
        key = (name, len(unit), hashlib.sha1(np.ascontiguousarray(unit).tobytes()).hexdigest())
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        Y = self.data[name]
        Ynp, lengths = self.tb._pad_chunk(Y, list(unit), self.structure.D)
        val = (self.torch.as_tensor(Ynp, dtype=self.dtype, device=self.device),
               self.torch.as_tensor(lengths, dtype=self.torch.int64, device=self.device))
        self.cache[key] = val
        while len(self.cache) > self.cache_entries:
            self.cache.popitem(last=False)
        return val

    def _kernel(self, params):
        return self.tb.TorchKernel.build(self.structure, params, device=self.device, dtype=self.dtype,
                                         pi0_reg=self.fbc.pi0_reg)

    def unit_scores(self, params_list, name, units):
        torch, tb = self.torch, self.tb
        kernels = [self._kernel(p) for p in params_list]
        out = []
        with torch.inference_mode():
            for unit in units:
                Yt, lt = self._tensors(name, unit)
                res = np.empty((len(unit), len(params_list)))
                for k, ker in enumerate(kernels):
                    try:
                        B_resc, _bc, log_scale = tb.compute_emissions_torch(ker, Yt)
                        _a, _s, ll = tb._forward_pass_batch(ker, B_resc, list(unit), lt, False)
                        res[:, k] = (ll + log_scale.sum(dim=(1, 2))).to("cpu").numpy()
                    except Exception:
                        res[:, k] = np.nan
                out.append(res)
        return out

    def unit_counts(self, params, name, units, weights):
        torch, tb = self.torch, self.tb
        ker = self._kernel(params)
        out = []
        for unit, Wu in zip(units, weights):
            keep = np.flatnonzero(np.any(Wu != 0, axis=1))
            if keep.size == 0:
                out.append(None)
                continue
            Yt, lt = self._tensors(name, unit)
            if keep.size < len(unit):
                kt = torch.as_tensor(keep, device=self.device)
                Yt, lt, Wu = Yt.index_select(0, kt), lt.index_select(0, kt), Wu[keep]
            wt = torch.as_tensor(Wu, dtype=self.dtype, device=self.device)
            with torch.inference_mode():
                try:
                    alpha, beta, gamma, scale, B, Bc, _ll = self._fb(ker, Yt, lt, list(np.asarray(unit)[keep]))
                except Exception as e:
                    raise NonfiniteStatistics(f"forward-backward failed on a work unit: {e}") from e
                out.append(_weighted_counts_device(torch, ker, self.structure, Yt, alpha, beta, gamma, scale, B, Bc,
                                                   lt, wt))
        return out

    def unit_fused(self, params_list, name, units, spec, labels, lengths_T):
        """Scores AND weighted counts of every model in ONE forward-backward per model and unit:
        the unit's scores give its weights (spec.weights, row-local), which weight the statistics
        of the same pass. Returns per unit (L_unit, [counts of model k (R)])."""
        torch = self.torch
        kernels = [self._kernel(p) for p in params_list]
        out = []
        for unit, lab, Tn in zip(units, labels, lengths_T):
            Yt, lt = self._tensors(name, unit)
            posts, L = [], np.empty((len(unit), len(params_list)))
            with torch.inference_mode():
                for k, ker in enumerate(kernels):
                    try:
                        post = self._fb(ker, Yt, lt, list(unit))
                    except Exception as e:
                        raise NonfiniteStatistics(f"forward-backward failed on a work unit: {e}") from e
                    posts.append(post)
                    L[:, k] = post[6].to("cpu").numpy()
                W_num, W_den = spec.weights(L, lab, Tn)
                cnts = []
                for k, (ker, post) in enumerate(zip(kernels, posts)):
                    alpha, beta, gamma, scale, B, Bc, _ll = post
                    wt = torch.as_tensor(np.column_stack([W_num[:, k], W_den[:, k]]), dtype=self.dtype, device=self.device)
                    cnts.append(_weighted_counts_device(torch, ker, self.structure, Yt, alpha, beta, gamma, scale, B,
                                                        Bc, lt, wt))
            out.append((L, cnts))
        return out


def _torch_worker(conn, structure, fbc, device):
    import traceback

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        core = _TorchCore(structure, fbc, device)
        conn.send(("ok", None))
    except Exception as e:  # pragma: no cover
        conn.send(("err", f"{e!r}\n{traceback.format_exc()}"))
        return
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            break
        try:
            conn.send(("ok", getattr(core, msg[0])(*msg[1:])))
        except Exception as e:
            conn.send(("err", f"{e!r}\n{traceback.format_exc()}"))


class _RemoteCore:
    """A _TorchCore in its own (spawned) process: true parallelism across devices (no GIL)."""

    def __init__(self, structure, fbc, device):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_torch_worker, args=(child, structure, fbc, device), daemon=True)
        self.proc.start()
        self._get()

    def send(self, *msg):
        self.conn.send(msg)

    def _get(self):
        status, val = self.conn.recv()
        if status != "ok":
            raise RuntimeError(f"torch worker failed: {val}")
        return val

    def call(self, *msg):
        self.send(*msg)
        return self._get()

    def close(self):
        try:
            self.conn.send(("stop",))
        except Exception:  # pragma: no cover
            pass
        self.proc.join(timeout=30)


class TorchEngine:
    """One device: computed in-process. Several entries in ``devices`` (``("cuda:0", "cuda:1")``;
    repeating a device runs several processes on it): one spawned worker process per entry, work
    units assigned round-robin, partial sums combined in unit order.

    Work units: the requested sequences sorted by length and cut into ``chunk_size`` pieces. With an
    explicit ``chunk_size`` results are identical for any device list; the automatic size (device
    memory, at most ceil(N / number of workers)) depends on the number of workers."""

    def __init__(self, structure: LSIMStructure, fbc: FbConstants, *, devices=("cuda",), chunk_size: int | None = None,
                 dtype: str = "float64"):
        if dtype != "float64":
            raise ValueError("discriminative training needs float64 (float32 sed statistics saturate beta_floor)")
        devices = (devices,) if isinstance(devices, str) else tuple(devices)
        self.structure, self.fbc = structure, fbc
        self.chunk_size = chunk_size
        self.n_models = 2          # models whose posteriors share a unit in fused passes (set by the trainer)
        self._data: dict[str, list[NDArray[np.float64]]] = {}
        if len(devices) == 1:
            self.cores = [_TorchCore(structure, fbc, devices[0])]
            self.remote = False
        else:
            self.cores = [_RemoteCore(structure, fbc, d) for d in devices]
            self.remote = True
        self.name = "torch[" + ",".join(devices) + ("; processes" if self.remote else "") + "]"

    def register(self, name: str, Y: list[NDArray[np.float64]]) -> None:
        Y = [np.ascontiguousarray(y, dtype=np.float64) for y in Y]
        self._data[name] = Y
        for c in self.cores:
            c.call("register", name, Y) if self.remote else c.register(name, Y)

    def size(self, name: str) -> int:
        return len(self._data[name])

    def lengths(self, name: str) -> NDArray[np.int64]:
        return np.array([y.shape[1] for y in self._data[name]])

    def data(self, name: str) -> list[NDArray[np.float64]]:
        return self._data[name]

    def _chunk(self, name: str) -> int:
        if self.chunk_size:
            return self.chunk_size
        N, T_max = self.size(name), int(self.lengths(name).max())
        c0 = self.cores[0]
        nm = self.n_models
        mem = c0.call("estimate_chunk", N, T_max, nm) if self.remote else c0.estimate_chunk(N, T_max, nm)
        self.chunk_size = max(1, min(mem, -(-N // len(self.cores))))
        return self.chunk_size

    def _units(self, name, idx):
        chunk = self._chunk(name)
        lens = self.lengths(name)[idx]
        order = np.argsort(lens, kind="stable")
        return [order[i:i + chunk] for i in range(0, len(order), chunk)]     # positions in idx

    def _dispatch(self, method, units, per_unit_args):
        """Run ``method`` on every unit; returns per-unit results in unit order."""
        n = len(self.cores)
        assign = [list(range(i, len(units), n)) for i in range(n)]
        if not self.remote:
            res = getattr(self.cores[0], method)(*per_unit_args(assign[0]))
            return res
        for c, js in zip(self.cores, assign):
            c.send(method, *per_unit_args(js))
        out = [None] * len(units)
        for c, js in zip(self.cores, assign):
            for j, r in zip(js, c._get()):
                out[j] = r
        return out

    def scores(self, params_list: list[LSIMParams], name: str, idx=None) -> NDArray[np.float64]:
        idx = np.arange(self.size(name)) if idx is None else np.asarray(idx, np.int64)
        units = self._units(name, idx)
        res = self._dispatch("unit_scores", units, lambda js: (params_list, name, [idx[units[j]] for j in js]))
        out = np.empty((len(idx), len(params_list)))
        for u, r in zip(units, res):
            out[u] = r
        return out

    def counts(self, params: LSIMParams, name: str, idx, W: NDArray[np.float64]) -> list[DiscCounts]:
        idx = np.asarray(idx, np.int64)
        W = np.asarray(W, np.float64)
        if np.any(W < 0) or not np.all(np.isfinite(W)):
            raise ValueError("statistic weights must be finite and nonnegative")
        units = self._units(name, idx)
        res = self._dispatch("unit_counts", units,
                             lambda js: (params, name, [idx[units[j]] for j in js], [W[units[j]] for j in js]))
        total = [DiscCounts.zeros(self.structure) for _ in range(W.shape[1])]
        for part in res:
            if part is None:
                continue
            for r in range(len(total)):
                total[r].add_(part[r])
        _check_counts(total)
        return total

    def fused_counts(self, params_list, name, idx, spec, label, T):
        """Scores (n x M) and weighted counts per model from one forward-backward per model:
        ``spec`` (objective.WeightSpec) turns each unit's scores into its weights. Identical to
        scores() followed by counts() with spec's weights (same units, same kernels)."""
        idx = np.asarray(idx, np.int64)
        label = np.asarray(label, np.int64)
        T = np.asarray(T, np.float64)
        units = self._units(name, idx)
        res = self._dispatch("unit_fused", units, lambda js: (
            params_list, name, [idx[units[j]] for j in js], spec, [label[units[j]] for j in js],
            [T[units[j]] for j in js]))
        L = np.empty((len(idx), len(params_list)))
        total = [[DiscCounts.zeros(self.structure) for _ in range(2)] for _ in params_list]
        for u, (Lu, cnts) in zip(units, res):
            L[u] = Lu
            for k, ck in enumerate(cnts):
                for r in range(2):
                    total[k][r].add_(ck[r])
        for t in total:
            _check_counts(t)
        return L, total

    def close(self) -> None:
        for c in self.cores:
            if self.remote:
                c.close()
        self.cores = []


def _weighted_counts_device(torch, ker, st, Y, alpha, beta, gamma, scale, B, Bc, lengths, w) -> list[DiscCounts]:
    """R weighted raw counts of one padded unit, reduced on its device (w: (Bn, R))."""
    Bn, S, T = alpha.shape
    dt = alpha.dtype
    dev = alpha.device
    R = w.shape[1]
    t_idx = torch.arange(T, device=dev)
    vf = (t_idx[None, :] < lengths[:, None]).to(dt)
    pi = w.T @ gamma[:, :, 0]                                            # (R, S)
    if T > 1:
        vt = (t_idx[:-1][None, :] < (lengths[:, None] - 1)).to(dt)
        abar = alpha[:, :, :-1] * scale[:, :, :-1] * vt[:, None, :]
        xi = torch.einsum("br,bit,bjt->rij", w, abar, beta[:, :, 1:]) * ker.W
    else:
        xi = torch.zeros(R, S, S, dtype=dt, device=dev)
    theta = torch.einsum("ic,rij,jz->rcz", ker.E, xi, ker.E)
    occ, nn, ss, rr = [], [], [], []
    for c in range(st.C):
        sl = st.state_slice(c)
        K = st.num_gmm_component[c]
        Yc = Y[:, st.dim_slice(c), :]
        o = ~torch.isnan(Yc)
        of = o.to(dt)
        Y0 = torch.where(o, Yc, torch.zeros_like(Yc))
        Bs = B[:, sl, :]
        ratio = Bc[:, sl, :K, :] / torch.where(Bs != 0, Bs, torch.ones_like(Bs))[:, :, None, :]
        ratio = torch.where((Bs != 0)[:, :, None, :], ratio, torch.full_like(ratio, float("nan")))
        g = gamma[:, sl, None, :] * ratio
        g = torch.where(torch.isnan(g), torch.zeros_like(g), g) * vf[:, None, None, :]
        occ.append(torch.einsum("br,bmkt->rmk", w, g))
        wg = torch.einsum("br,bmkt->rbmkt", w, g) if R > 1 else (w[:, 0, None, None, None] * g)[None]
        nn.append(torch.einsum("rbmkt,bdt->rdmk", wg, of))
        ss.append(torch.einsum("rbmkt,bdt->rdmk", wg, Y0))
        rr.append(torch.einsum("rbmkt,bdt->rdmk", wg, Y0 * Y0))

    def np_(x):
        return x.detach().to("cpu").numpy()

    pi, xi, theta = np_(pi), np_(xi), np_(theta)
    occ, nn, ss, rr = ([np_(x) for x in lst] for lst in (occ, nn, ss, rr))
    return [DiscCounts(pi=pi[r], A=xi[r], theta=theta[r], occ=[o[r] for o in occ], n=[x[r] for x in nn],
                       s=[x[r] for x in ss], r=[x[r] for x in rr]) for r in range(R)]


def make_engine(structure: LSIMStructure, opts) -> "NumpyEngine | TorchEngine":
    if opts.engine == "torch":
        return TorchEngine(structure, opts.fb_constants, devices=opts.devices, chunk_size=opts.chunk_size)
    return NumpyEngine(structure, opts.fb_constants, n_jobs=opts.n_jobs, backend=opts.numpy_backend,
                       chunk_size=opts.chunk_size)


def available_cuda_devices() -> list[str]:
    try:
        import torch
    except ImportError:
        return []
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []

