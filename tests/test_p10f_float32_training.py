"""P10f item 3: CUDA/Torch float32 EM training is explicitly unsupported.

The float32 gate study (`benchmarks/p8b/float32_training_gate_study.py`) found that a bounded float64
renormalization repairs the M-step's stochastic invariants, but that `variant="sed"` float32 statistics are
themselves wrong (beta_floor saturation in the float32 SED backward recursion), so the gates do not all pass.
These tests pin the resulting contract: the three EM drivers reject `dtype="float32"` up front, float64 and the
CPU backends are unaffected, and float32 forward-only scoring is still available.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import chmm_lsim.emissions.gmm as gmm_mod  # noqa: E402
import chmm_lsim.models.lsim.em as em  # noqa: E402
from chmm_lsim.backends.base import UnsupportedCombination  # noqa: E402
from chmm_lsim.backends.torch_batched import forward_backward_batch  # noqa: E402
from chmm_lsim.core.constants import FbConstants  # noqa: E402
from chmm_lsim.emissions.gmm import ChannelGMM  # noqa: E402
from chmm_lsim.models.lsim.params import LSIMParams  # noqa: E402
from chmm_lsim.models.lsim.simulate import simulate_lsim  # noqa: E402
from chmm_lsim.core.structure import LSIMStructure  # noqa: E402
from chmm_lsim.core.validate import ValidationError  # noqa: E402


def _workload(C=6, M=5, n=400, T=10, seed=0):
    st = LSIMStructure(channel_state_num=(M,) * C, channel_obs_dim=(1,) * C, num_gmm_component=(1,) * C)
    rng = np.random.default_rng(seed)
    pi_0 = np.concatenate([rng.dirichlet(np.ones(M)) for _ in range(C)])
    tr = np.zeros((st.S, st.S))
    for z in range(C):
        tr[:, st.state_slice(z)] = rng.dirichlet(np.ones(M) * 2, size=st.S)
    coupling = rng.dirichlet(np.ones(C), size=C).T
    emissions = [ChannelGMM(np.ones((M, 1)), np.linspace(-4, 4, M)[None, :, None] + rng.normal(0, 0.2, (1, M, 1)), np.full((1, M, 1), 0.8), diag=True)
                 for _ in range(C)]
    p = LSIMParams(structure=st, pi_0=pi_0, transition=tr, coupling=coupling, emissions=emissions)
    p.validate()
    Y = [simulate_lsim(st, p, T, seed=i, validate=False)[0] for i in range(n)]
    return st, p, Y


DRIVERS = {
    "em_lsim": lambda st, p, Y, **kw: em.em_lsim(st, p, Y, max_iterations=2, check_convergence=False, **kw),
    "em_lsim_minibatch": lambda st, p, Y, **kw: em.em_lsim_minibatch(st, p, Y, max_iterations=2, num_minibatches=2, shuffle=False, **kw),
    "em_lsim_minibatch_train": lambda st, p, Y, **kw: em.em_lsim_minibatch_train(st, p, Y, max_iterations=2, num_minibatches=2, shuffle=False,
                                                                                 convergence="off", **kw),
}


@pytest.mark.parametrize("driver", sorted(DRIVERS))
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))])
def test_float32_training_is_rejected_before_any_work(driver, device, monkeypatch):
    st, p, Y = _workload(n=20)

    def boom(*a, **k):
        raise AssertionError("no E-step/validation may run before the float32 rejection")

    monkeypatch.setattr(em, "e_step", boom)
    monkeypatch.setattr(gmm_mod, "compute_global_nanvar", boom)
    with pytest.raises(UnsupportedCombination, match="float32"):
        DRIVERS[driver](st, p, Y, backend="torch", backend_options={"device": device, "dtype": "float32"})


@pytest.mark.parametrize("driver", sorted(DRIVERS))
def test_float64_and_default_dtype_training_still_run(driver):
    st, p, Y = _workload(n=20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for opts in ({"dtype": "float64"}, {}, None):
            DRIVERS[driver](st, p.copy(), Y, backend="torch", backend_options=opts)
        DRIVERS[driver](st, p.copy(), Y, backend="numpy")


def test_float32_forward_only_scoring_is_still_available():
    st, p, Y = _workload(n=50)
    ll32 = forward_backward_batch(st, p, Y, FbConstants(variant="sed"), mode="loglik", dtype="float32")
    ll64 = forward_backward_batch(st, p, Y, FbConstants(variant="sed"), mode="loglik", dtype="float64")
    np.testing.assert_allclose(ll32, ll64, rtol=1e-5)


def test_reproduces_the_float32_pi0_block_sum_failure_this_guard_replaces():
    """The original symptom, reproduced directly (Torch CPU float32; the CUDA study shows the same): an
    M-step from float32 statistics leaves a `pi_0` block summing to 1 +/- a few 1e-8, which the next
    E-step's `params.validate()` rejects (tolerance 1e-8)."""
    st, p, Y = _workload(n=400)
    raw = em.e_step(st, p, Y, FbConstants(variant="sed"), backend="torch", backend_options={"dtype": "float32"})
    out = em.m_step(st, em.finalize_expected_stats(raw, sigma_diag=True), sigma_diag=True)
    drift = max(abs(out.pi_0[st.state_slice(c)].sum() - 1.0) for c in range(st.C))
    assert drift > 1e-8
    with pytest.raises(ValidationError, match="pi_0 block"):
        out.validate()
