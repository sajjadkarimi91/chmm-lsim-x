"""P10e item 8: explicit multiprocessing start methods for EM process pools.

EM pools (e_step, forward-only coherent scoring, both mini-batch drivers) use
an explicit context whose default is "spawn" everywhere. On the development
cluster (system libffi 3.1) a ctypes-callback workload segfaults under "fork"
on Python 3.12-3.14 and under "forkserver" on 3.14 -- numba/llvmlite create
such callbacks -- while "spawn" is safe. These tests pin the default, prove
every pool uses the requested context, run the default path after a fully
warmed parent, and run the libffi-closure reproducer under the default.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import chmm_lsim.models.lsim.em as em_mod
from chmm_lsim import ChannelGMM, FbConstants, LSIMParams, LSIMStructure, e_step, em_lsim_minibatch, em_lsim_minibatch_train


def _problem(seed=0):
    structure = LSIMStructure(channel_state_num=(2, 3), channel_obs_dim=(1, 1), num_gmm_component=(1, 1))
    rng = np.random.default_rng(seed)
    pi_0 = np.concatenate([rng.dirichlet(np.ones(m)) for m in structure.channel_state_num])
    transition = np.zeros((structure.S, structure.S))
    for z in range(structure.C):
        sl = structure.state_slice(z)
        transition[:, sl] = rng.dirichlet(np.ones(sl.stop - sl.start), size=structure.S)
    coupling = rng.dirichlet(np.ones(structure.C), size=structure.C).T
    emissions = [ChannelGMM(np.ones((m, 1)), rng.normal(size=(1, m, 1)), np.ones((1, m, 1)), diag=True) for m in structure.channel_state_num]
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=emissions)
    params.validate()
    return structure, params, [rng.normal(size=(2, T)) for T in (30, 12, 45, 20)]


def test_default_start_method_is_spawn_on_every_platform(monkeypatch):
    assert em_mod._resolve_start_method(None, "process") == "spawn"
    monkeypatch.setattr(em_mod.multiprocessing, "get_all_start_methods", lambda: ["fork", "spawn", "forkserver"])
    assert em_mod._resolve_start_method(None, "process") == "spawn"
    assert em_mod._resolve_start_method("forkserver", "process") == "forkserver"  # explicit opt-in still allowed
    assert em_mod._resolve_start_method(None, "thread") is None and em_mod._resolve_start_method(None, None) is None


@pytest.mark.parametrize("executor", [None, "thread"])
def test_start_method_without_process_executor_is_rejected(executor):
    structure, params, Y = _problem()
    n_jobs = 1 if executor is None else 2
    with pytest.raises(ValueError, match="only applies to executor='process'"):
        e_step(structure, params, Y, FbConstants(), n_jobs=n_jobs, executor=executor, start_method="spawn")
    with pytest.raises(ValueError, match="only applies to executor='process'"):
        em_lsim_minibatch_train(structure, params, Y, max_iterations=1, num_minibatches=1, shuffle=False, n_jobs=n_jobs,
                                executor=executor, start_method="spawn")


def test_unknown_start_method_is_rejected():
    structure, params, Y = _problem()
    with pytest.raises(ValueError, match="start_method must be one of"):
        e_step(structure, params, Y, FbConstants(), n_jobs=2, executor="process", start_method="teleport")
    with pytest.raises(ValueError, match="start_method must be one of"):
        em_lsim_minibatch(structure, params, Y, max_iterations=1, num_minibatches=1, shuffle=False, n_jobs=2,
                          executor="process", start_method="teleport")


class _RecordingPPE(concurrent.futures.ProcessPoolExecutor):
    methods: list[str] = []

    def __init__(self, *args, mp_context=None, **kwargs):
        type(self).methods.append(mp_context.get_start_method() if mp_context is not None else "platform-default")
        super().__init__(*args, mp_context=mp_context, **kwargs)


@pytest.mark.parametrize("start_method", [None, "spawn"])
def test_every_em_process_pool_uses_the_requested_context(monkeypatch, start_method):
    structure, params, Y = _problem(seed=1)
    _RecordingPPE.methods = []
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _RecordingPPE)
    expected = em_mod._resolve_start_method(start_method, "process")
    kw = dict(n_jobs=2, executor="process", start_method=start_method)
    ref = e_step(structure, params, Y, FbConstants()).loglik
    assert e_step(structure, params, Y, FbConstants(), **kw).loglik == pytest.approx(ref, rel=1e-12)
    assert em_mod._forward_only_loglik(structure, params, Y, FbConstants(), backend="numpy", validate="params", **kw) == ref
    em_lsim_minibatch(structure, params, Y, max_iterations=1, num_minibatches=2, shuffle=False, **kw)
    r = em_lsim_minibatch_train(structure, params, Y, max_iterations=2, num_minibatches=2, shuffle=False, convergence="off", **kw)
    # 1 e_step + 1 forward-only + 2 minibatch updates + (2 updates x 2 epochs + 2 coherent + 1 pi_steady)
    assert len(_RecordingPPE.methods) == 1 + 1 + 2 + 4 + 2 + 1
    assert set(_RecordingPPE.methods) == {expected}
    assert r.n_epochs == 2


_WARM_PARENT_SCRIPT = textwrap.dedent('''
    import sys, warnings
    import numpy as np

    def main():
        import chmm_lsim.models.lsim.em as em
        from chmm_lsim import FbConstants, LSIMStructure, e_step, em_lsim_minibatch_train, initialize_lsim_params
        rng = np.random.default_rng(0)
        st = LSIMStructure(channel_state_num=(4,) * 4, channel_obs_dim=(1,) * 4, num_gmm_component=(1,) * 4)
        Y = [rng.normal(size=(4, int(t))) for t in rng.integers(100, 600, size=10)]
        _ = np.linalg.svd(rng.normal(size=(400, 400)))                       # multithreaded BLAS in the parent
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = initialize_lsim_params(st, Y, sigma_diag=True, seed=0)   # serial sklearn fits: OpenMP in the parent
            from sklearn.mixture import GaussianMixture
            GaussianMixture(6, covariance_type="full").fit(rng.normal(size=(50000, 3)))
        fbc = FbConstants()
        serial = e_step(st, params, Y, fbc, backend="numba").loglik             # numba kernels in the parent
        for _ in range(2):
            par = e_step(st, params, Y, fbc, backend="numba", n_jobs=3, executor="process").loglik
            fwd = em._forward_only_loglik(st, params, Y, fbc, backend="numba", validate="params", n_jobs=3, executor="process")
            assert abs(par - serial) <= 1e-9 * abs(serial) and fwd == serial, (par, fwd, serial)
        bad = [y.copy() for y in Y]
        bad[7][:, 0] = 1e6   # finite, passes coordinator validation; every emission underflows -> the worker raises
        try:
            em_lsim_minibatch_train(st, params, bad, max_iterations=1, num_minibatches=2, shuffle=False, convergence="off",
                                    backend="numba", n_jobs=3, executor="process")
        except em.TrialExecutionError as exc:
            print("TRIAL_INDEX", exc.trial_index)
        print("DEFAULT_METHOD", em._resolve_start_method(None, "process"))
        print("OK")

    if __name__ == "__main__":
        main()
''')


def test_default_process_pools_work_after_a_fully_warmed_parent(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("numba")
    script = tmp_path / "warm_parent.py"
    script.write_text(_WARM_PARENT_SCRIPT)
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout
    assert "TRIAL_INDEX 7" in proc.stdout  # the GLOBAL trial index, through a mini-batch update
    assert f"DEFAULT_METHOD {em_mod._resolve_start_method(None, 'process')}" in proc.stdout



_LIBFFI_CLOSURE_SCRIPT = textwrap.dedent("""
    import ctypes, concurrent.futures as cf, multiprocessing as mp
    import chmm_lsim.models.lsim.em as em

    def work(n):
        libc = ctypes.CDLL(None)
        CMP = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
        total = 0
        for _ in range(200):
            arr = (ctypes.c_int * 64)(*range(64, 0, -1))
            cb = CMP(lambda a, b: a[0] - b[0])
            libc.qsort(arr, 64, ctypes.sizeof(ctypes.c_int), cb)
            total += arr[0]
        return total

    if __name__ == "__main__":
        keep = [ctypes.CFUNCTYPE(ctypes.c_int)(lambda: 0) for _ in range(1000)]
        method = em._resolve_start_method(None, "process")
        with cf.ProcessPoolExecutor(3, mp_context=mp.get_context(method)) as pool:
            print("OK", method, sum(pool.map(work, range(12))))
""")


def test_default_start_method_survives_libffi_closures_in_workers(tmp_path):
    """Regression for the P10e closure finding: libffi closures created in pool
    workers (as numba/llvmlite do) segfault under fork (3.12-3.14) and forkserver
    (3.14) with the cluster's system libffi 3.1. The default must not."""
    import os

    script = tmp_path / "libffi_closures.py"
    script.write_text(_LIBFFI_CLOSURE_SCRIPT)
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300,
                          env={**os.environ, "PYTHONFAULTHANDLER": "1"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.startswith("OK spawn 2400"), proc.stdout
