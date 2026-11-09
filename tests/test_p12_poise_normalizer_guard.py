"""P12 correction round: POISE's per-channel normalizer guard.

MATLAB's `forward_backward_lsim_poise.m` has always refused to divide by a
nonpositive or non-finite channel normalizer (`normalize_blocks`, lines 180-188);
the *unified* kernel's POISE branch divided unguarded, and so did this port. The
result was silent corruption rather than an error, and it is **reachable with a
perfectly valid model whose forward log-likelihood is finite** -- which is what
this module pins, on every backend and through every public entry point.

**The construction** (`_zero_normalizer_model`, C=2, verified against MATLAB:
`tools/matlab/check_em_lsim_poise.m` builds the same model and asserts the same
two failures, and both languages report the same channel and timestep).

- `theta = [[0, 0.5], [1, 0.5]]` -- column-stochastic, so the model is valid.
  Channel A's outgoing row is concentrated on B, hence `rho[A] = [0, 1]`:
  A's backward message reads block B and nothing else.
- Every state of A transitions into B deterministically at B-state 0.
- B-state 0's emission density underflows to exactly 0, while B-state 1's does
  not -- so `beta[B-state 0]` is 0, A's mixture `q` is exactly 0, and A's total
  smoothed mass is 0.
- The forward is unaffected: B's own self-coupling (`theta[B,B] = 0.5`) keeps
  mass on the on-support state, so the log-likelihood is finite (it equals
  SED's on the same model, to 12 digits, in both languages).

Without the guard, `validate="params"` returned `beta`/`gamma` full of `NaN`
from that finite-likelihood model, and with the refinement on the `NaN` spread
through `propagated` to **every** channel and timestep -- so a training run
consumed all-NaN statistics instead of failing.

The guard is deliberately **unconditional**: it is not gated on `validate=`,
because MATLAB's is not, and because silently returning NaN posteriors is not a
debugging aid the caller opted out of. `validate="off"` must still raise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from chmm_lsim.backends.base import run_backend  # noqa: E402
from chmm_lsim.compat import FbConstants  # noqa: E402
from chmm_lsim.em import e_step, em_lsim, em_lsim_minibatch_train  # noqa: E402
from chmm_lsim.inference import forward_backward  # noqa: E402
from chmm_lsim.params import ChannelGMM, LSIMParams  # noqa: E402
from chmm_lsim.structure import LSIMStructure  # noqa: E402
from chmm_lsim.validate import ValidationError  # noqa: E402

REFINEMENTS = [False, True]
CPU_BACKENDS = ["numpy", "numba", "torch"]


def _gmm(mu0: float, mu1: float) -> ChannelGMM:
    """A 2-state, 1-D, single-component channel with the two means given."""
    return ChannelGMM(np.ones((2, 1)), np.array([[[mu0], [mu1]]], dtype=float), np.ones((1, 2, 1)), diag=True)


def _zero_normalizer_model():
    """The C=2 construction described in the module docstring. Returns
    `(structure, params, Y)`; `params.validate()` passes."""
    structure = LSIMStructure(channel_state_num=(2, 2), channel_obs_dim=(1, 1), num_gmm_component=(1, 1))
    coupling = np.array([[0.0, 0.5], [1.0, 0.5]])  # columns sum to 1
    transition = np.zeros((4, 4))
    transition[:, 0:2] = 0.5  # into block A: uniform from every parent state
    transition[0:2, 2] = 1.0  # A -> B: all mass on B-state 0 (the off-support one)
    transition[2:4, 2] = 0.5  # B -> B: keeps reachable mass on the on-support state
    transition[2:4, 3] = 0.5
    params = LSIMParams(
        structure=structure,
        pi_0=np.full(4, 0.5),
        transition=transition,
        coupling=coupling,
        emissions=[_gmm(0.0, 0.0), _gmm(300.0, 0.0)],  # B-state 0's density underflows to exactly 0
    )
    params.validate()
    return structure, params, np.zeros((2, 3))


def _healthy_model():
    """The same shape, but with every channel's outgoing mixture reachable --
    used to show the guard does not fire on ordinary models."""
    structure, params, Y = _zero_normalizer_model()
    params.transition[0:2, 2] = 0.5  # A -> B no longer avoids the on-support state
    params.transition[0:2, 3] = 0.5
    params.validate()
    return structure, params, Y


def _poise(refinement: bool) -> FbConstants:
    return FbConstants(variant="poise", poise_refinement=refinement)


def _requires(backend: str, device: str = "cpu"):
    if backend == "numba":
        pytest.importorskip("numba")
    if backend == "torch":
        torch = pytest.importorskip("torch")
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")


def _opts(backend: str, device: str):
    return {"device": device, "dtype": "float64"} if backend == "torch" else None


# --------------------------------------------------------------------------
# The premise: the model is valid and its forward log-likelihood is finite.
# --------------------------------------------------------------------------


def test_the_construction_is_a_valid_model_with_a_finite_loglik():
    """If this ever stops holding, the guard tests below are vacuous: they would
    be rejecting a model that should have been rejected earlier, for a different
    reason. The value is MATLAB's own (`check_em_lsim_poise.m` prints it)."""
    structure, params, Y = _zero_normalizer_model()
    params.validate()
    np.testing.assert_allclose(params.coupling.sum(axis=0), 1.0, rtol=0, atol=1e-15)
    for block in (slice(0, 2), slice(2, 4)):
        np.testing.assert_allclose(params.transition[:, block].sum(axis=1), 1.0, rtol=0, atol=1e-15)

    sed = forward_backward(structure, params, Y, FbConstants(variant="sed"))
    assert np.isfinite(sed.loglik)
    np.testing.assert_allclose(sed.loglik, -8.979367102027762, rtol=1e-12, atol=0)
    assert not np.isnan(sed.gamma).any(), "SED is unaffected: only POISE's mixture collapses"


# --------------------------------------------------------------------------
# Single-sequence inference, every backend, both modes, every validate level.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", CPU_BACKENDS)
@pytest.mark.parametrize("refinement", REFINEMENTS)
@pytest.mark.parametrize("validate", ["params", "all", "off"])
def test_zero_channel_normalizer_raises_on_every_backend(backend, refinement, validate):
    """Unconditional: `validate="off"` raises too. The message names the stage,
    the channel and the timestep."""
    _requires(backend)
    structure, params, Y = _zero_normalizer_model()
    kwargs = {"device": "cpu"} if backend == "torch" else {}
    with pytest.raises(ValidationError, match=r"POISE backward recursion.*channel 0.*t=1"):
        run_backend(backend, "poise", structure, params, Y, _poise(refinement), validate=validate, **kwargs)


@pytest.mark.parametrize("refinement", REFINEMENTS)
def test_public_forward_backward_raises_too(refinement):
    """Not only the backend runner: the documented single-sequence entry point."""
    structure, params, Y = _zero_normalizer_model()
    with pytest.raises(ValidationError, match="nonpositive or non-finite"):
        forward_backward(structure, params, Y, _poise(refinement))


@pytest.mark.parametrize("backend", CPU_BACKENDS)
@pytest.mark.parametrize("refinement", REFINEMENTS)
def test_no_nan_posteriors_are_ever_returned(backend, refinement):
    """The property that actually matters, stated directly: from this model the
    call either returns finite posteriors or raises -- never NaN."""
    _requires(backend)
    structure, params, Y = _zero_normalizer_model()
    kwargs = {"device": "cpu"} if backend == "torch" else {}
    try:
        post = run_backend(backend, "poise", structure, params, Y, _poise(refinement), **kwargs)
    except ValidationError:
        return
    for name in ("beta", "gamma"):
        assert np.isfinite(getattr(post, name)).all(), f"{name} returned NaN/Inf instead of raising"


@pytest.mark.parametrize("backend", CPU_BACKENDS)
@pytest.mark.parametrize("refinement", REFINEMENTS)
def test_guard_does_not_fire_on_a_healthy_model(backend, refinement):
    """The other half of the contract: an ordinary model still runs, and its
    posteriors are finite and properly normalized."""
    _requires(backend)
    structure, params, Y = _healthy_model()
    kwargs = {"device": "cpu"} if backend == "torch" else {}
    post = run_backend(backend, "poise", structure, params, Y, _poise(refinement), validate="all", **kwargs)
    assert np.isfinite(post.gamma).all()
    for c, (lo, hi) in enumerate([(0, 2), (2, 4)]):
        np.testing.assert_allclose(post.gamma[lo:hi].sum(axis=0), 1.0, rtol=0, atol=1e-12, err_msg=f"channel {c}")


# --------------------------------------------------------------------------
# Batched Torch: ragged batches, chunking, and padded-timestep masking.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("refinement", REFINEMENTS)
@pytest.mark.parametrize("chunk_size", [1, 2, 3])
def test_ragged_batch_raises_and_names_the_offending_sequence(refinement, chunk_size):
    """A bad sequence batched beside unaffected ones must fail the call loudly and
    identify itself by its ORIGINAL dataset index, not its position in whatever
    chunk the scheduler happened to put it in.

    What makes a sequence trip the guard here is the *parameters*, not its data,
    so every sequence long enough to take a backward step trips it. The
    unaffected companions are therefore the `T=1` ones, whose backward loop is
    empty (`range(T-2,-1,-1)`) and which skip the refinement entirely."""
    pytest.importorskip("torch")
    from chmm_lsim.backends.torch_batched import forward_backward_batch

    structure, params, Y_bad = _zero_normalizer_model()
    data = [Y_bad[:, :1], Y_bad, Y_bad[:, :1]]  # only index 1 takes a backward step
    with pytest.raises(ValidationError, match=r"sequence 1\b"):
        list(forward_backward_batch(structure, params, data, _poise(refinement),
                                    mode="posterior", chunk_size=chunk_size))


@pytest.mark.parametrize("refinement", REFINEMENTS)
def test_padded_timesteps_do_not_trigger_a_false_positive(refinement):
    """The guard is masked to each sequence's own real timesteps. A short
    sequence padded out beside a much longer one must not be reported."""
    pytest.importorskip("torch")
    from chmm_lsim.backends.torch_batched import forward_backward_batch

    structure, params, Y = _healthy_model()
    data = [np.tile(Y, (1, 20)), Y[:, :1], Y[:, :2]]
    chunks = list(forward_backward_batch(structure, params, data, _poise(refinement),
                                         mode="posterior", chunk_size=3))
    seen = {i for chunk in chunks for i in chunk.sequence_indices}
    assert seen == {0, 1, 2}


@pytest.mark.parametrize("refinement", REFINEMENTS)
def test_cuda_raises_exactly_like_cpu(refinement):
    """Same failure, same message, on the device where the check had to be
    written differently (accumulated on-device, inspected once)."""
    _requires("torch", "cuda")
    structure, params, Y = _zero_normalizer_model()
    with pytest.raises(ValidationError, match=r"POISE backward recursion.*channel 0.*t=1"):
        run_backend("torch", "poise", structure, params, Y, _poise(refinement), device="cuda")


@pytest.mark.parametrize("refinement", REFINEMENTS)
def test_cuda_healthy_model_still_runs(refinement):
    _requires("torch", "cuda")
    structure, params, Y = _healthy_model()
    post = run_backend("torch", "poise", structure, params, Y, _poise(refinement), device="cuda")
    assert np.isfinite(post.gamma).all()


# --------------------------------------------------------------------------
# Statistics and training: nothing NaN may reach the M-step.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", CPU_BACKENDS)
@pytest.mark.parametrize("refinement", REFINEMENTS)
def test_e_step_raises_so_no_nan_statistics_reach_the_m_step(backend, refinement):
    """The consequence the guard exists for. Before it, `e_step` returned
    `ExpectedStats` whose fields were NaN and `m_step` consumed them."""
    _requires(backend)
    structure, params, Y = _zero_normalizer_model()
    with pytest.raises(ValidationError, match="nonpositive or non-finite"):
        e_step(structure, params, [Y], _poise(refinement), backend=backend,
               backend_options=_opts(backend, "cpu"), validate="off")


@pytest.mark.parametrize("refinement", REFINEMENTS)
def test_e_step_statistics_are_finite_on_a_healthy_model(refinement):
    """Stated as its own contract so the test above cannot pass by rejecting
    everything."""
    structure, params, Y = _healthy_model()
    stats = e_step(structure, params, [Y], _poise(refinement), backend="numpy")
    for name in ("zeta", "gamma_first", "gamma_sum"):
        assert np.isfinite(getattr(stats, name)).all(), name
    assert np.isfinite(stats.loglik)


@pytest.mark.parametrize("refinement", REFINEMENTS)
def test_em_lsim_training_fails_loudly(refinement):
    structure, params, Y = _zero_normalizer_model()
    with pytest.raises(ValidationError, match="nonpositive or non-finite"):
        em_lsim(structure, params.copy(), [Y], max_iterations=2, fb_constants=_poise(refinement),
                check_convergence=False, validate="off")


@pytest.mark.parametrize("refinement", REFINEMENTS)
def test_minibatch_training_fails_loudly(refinement):
    structure, params, Y = _zero_normalizer_model()
    with pytest.raises(ValidationError, match="nonpositive or non-finite"):
        em_lsim_minibatch_train(structure, params.copy(), [Y, Y], max_iterations=1, num_minibatches=1,
                                fb_constants=_poise(refinement), convergence="off", backend="numpy", n_jobs=1)


# --------------------------------------------------------------------------
# The refinement stage has its own guard.
# --------------------------------------------------------------------------


def test_refinement_stage_guard_is_reachable_and_reports_its_own_stage(monkeypatch):
    """The raw recursion and the refinement both normalize per channel, so both
    are guarded. The raw stage is what the construction above trips; this pins
    the refinement stage's own message by feeding `_refine` a propagated slice
    that collapses, which no raw-stage failure can mask."""
    import chmm_lsim.inference.poise as poise_mod

    structure, params, Y = _healthy_model()
    fbc = _poise(True)
    real_check = poise_mod.check_channel_normalizer
    seen_stages = []

    def spy(normalizer_channel, t, *, stage, sequence=None):
        seen_stages.append(stage)
        if stage == "refinement":
            normalizer_channel = np.zeros_like(normalizer_channel)  # collapse it
        return real_check(normalizer_channel, t, stage=stage, sequence=sequence)

    monkeypatch.setattr(poise_mod, "check_channel_normalizer", spy)
    with pytest.raises(ValidationError, match="POISE refinement.*channel 0"):
        poise_mod.forward_backward_poise(structure, params, Y, fbc)
    assert "backward recursion" in seen_stages and "refinement" in seen_stages


def test_guard_helper_accepts_finite_positive_and_rejects_each_bad_value():
    """The helper's own contract, independent of any recursion."""
    from chmm_lsim.inference.poise import check_channel_normalizer

    check_channel_normalizer(np.array([1.0, 1e-300]), 0, stage="unit")
    for bad in (0.0, -1.0, np.nan, np.inf, -np.inf):
        with pytest.raises(ValidationError, match="channel 1"):
            check_channel_normalizer(np.array([1.0, bad]), 7, stage="unit")


def test_guard_message_names_the_sequence_when_one_is_given():
    from chmm_lsim.inference.poise import check_channel_normalizer

    with pytest.raises(ValidationError, match="sequence 4, channel 0"):
        check_channel_normalizer(np.array([0.0]), 2, stage="unit", sequence=4)
