"""P10c gate: the backend-independent incremental mini-batch EM scheduling/
update driver (`em.em_lsim_minibatch`, docs/PYTHON_PORT_PLAN.md §2.6).

Verified here, matching the P10c task's own explicit list:

1. Exact P10a schedule replay across all six scenarios and every update,
   through the REAL production driver -- no test-only global-nanvar patch
   (contrast the P10b correction round's own replay tests, which needed
   one; §2.6's covariance-ridge policy is now real production code).
2. `num_minibatches=1` full P4 non-regression (byte-identical to `em_lsim`,
   not merely within tolerance, given matching `shuffle=False`/`backend=
   "numpy"`/`n_jobs=1`).
3. Seed reproducibility and different-seed schedule divergence.
4. Uneven partitions and `num_minibatches=N` (one trial per mini-batch).
5. Serial/parallel NumPy and Numba parity (the driver's own schedule/update
   loop, on top of P10b's already-verified per-call `e_step` parity).
6. Torch CPU float64 parity against NumPy, same schedule.
7. The zero-transition structural-preservation policy, including a mixed
   epoch (one all-`T=1` mini-batch alongside an ordinary one).
8. Large-offset and missing-data covariance stability under mini-batch
   training specifically (not just single-call `e_step`, already covered
   by P10b's own tests).
9. Worker errors retain the GLOBAL trial index (into the whole dataset),
   not the mini-batch-local one `e_step`'s own `TrialExecutionError` uses.
10. Schedule validation/generation contract: explicit `np.random.Generator`
    only (never global NumPy randomness), the `num_minibatches>N`
    warn+clamp policy, and every documented validation failure mode.

§P10c correction round additions:

11. MATLAB-compatible (half-away-from-zero) edge rounding in `generate_
    minibatch_schedule`, exact batch sizes and ordered membership.
12. The hardened schedule contract: bool/float trial-index rejection, deep
    list-to-tuple canonicalization, a constant per-epoch mini-batch count
    enforced by `validate_minibatch_schedule`, empty-`Y_trials` rejection,
    and `schedule=` raising (not silently ignoring) if `num_minibatches`/
    `shuffle`/`random_state` are also explicitly given.
13. `compute_global_nanvar`'s inference-free rewrite: parity against the
    prior E-step-derived value on all six P10a fixtures, missing-data/
    singleton/fully-missing/large-offset coverage, and a monkeypatch proof
    that no inference/backend call happens at all.
14. `training_batch_loglik_by_epoch`/`training_batch_loglik_total` (renamed
    from the ambiguous `training_batch_loglik_sum`), checked per-epoch.
15. `m_step`'s zero-transition `params_in` structure-compatibility check
    (same S/C, different state partition -- a real regression case mere
    array-shape compatibility would miss).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import P10A_SCENARIO_IDS, load_em_trace_fixture, load_p10a_scheduling_fixture  # noqa: E402

from chmm_lsim.core.constants import FbConstants
from chmm_lsim.learning.minibatch import MinibatchSchedule, generate_minibatch_schedule, validate_minibatch_schedule
from chmm_lsim.learning.em_loop import TrialExecutionError, finalize_expected_stats
from chmm_lsim.emissions.gmm import compute_global_nanvar
from chmm_lsim.models.lsim.em import e_step, em_lsim, em_lsim_minibatch, m_step
from chmm_lsim.emissions.gmm import ChannelGMM
from chmm_lsim.models.lsim.params import LSIMParams
from chmm_lsim.core.structure import LSIMStructure

EXECUTORS = ("thread", "process")


def _toy_model(rng, *, C=2, M=(2, 3), d=(1, 2), K=(1, 2), sigma_diag=True):
    structure = LSIMStructure(channel_state_num=M, channel_obs_dim=d, num_gmm_component=K)
    S = structure.S
    pi_0 = np.zeros(S)
    transition = np.zeros((S, S))
    for c in range(C):
        sl = structure.state_slice(c)
        width = sl.stop - sl.start
        pi_0[sl] = 1.0 / width
        transition[:, sl] = 1.0 / width
    coupling = np.full((C, C), 1.0 / C)
    emissions = []
    for c in range(C):
        M_c, K_c, d_c = structure.channel_state_num[c], structure.num_gmm_component[c], structure.channel_obs_dim[c]
        weights = np.full((M_c, K_c), 1.0 / K_c)
        mu = rng.normal(size=(d_c, M_c, K_c))
        if sigma_diag:
            sigma = np.ones((d_c, M_c, K_c))
        else:
            sigma = np.tile(np.eye(d_c)[:, :, None, None], (1, 1, M_c, K_c))
        emissions.append(ChannelGMM(weights=weights, mu=mu, sigma=sigma, diag=sigma_diag))
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=emissions)
    return structure, params


def _toy_trials(rng, structure, lengths):
    return [rng.normal(size=(structure.D, T)) for T in lengths]


def _packed_close(a: LSIMParams, b: LSIMParams, *, rtol, atol) -> None:
    pa, pb = a.to_packed(), b.to_packed()
    for key in ("pi_0", "transitions_matrices", "coupling_tetha_IM", "P_all", "mu_all", "sigma_all"):
        np.testing.assert_allclose(pb[key], pa[key], rtol=rtol, atol=atol, err_msg=key)


# ==========================================================================
# 1. Exact P10a golden-schedule replay -- real production code, no patch
# ==========================================================================


def _schedule_from_p10a_fixture(fx) -> MinibatchSchedule:
    epochs = []
    for itr in range(fx.max_itration):
        mbs = []
        for mb in range(fx.mini_batch):
            u = fx.update_index(itr + 1, mb + 1)
            n_tr = int(fx.updates_n_trials[u])
            idx0 = tuple(int(i) - 1 for i in fx.updates_trial_indices[u, :n_tr])
            mbs.append(idx0)
        epochs.append(tuple(mbs))
    return MinibatchSchedule(epochs=tuple(epochs))


@pytest.mark.parametrize("scenario_id", P10A_SCENARIO_IDS)
def test_golden_schedule_replay_matches_updates_out_no_patch(fixtures_dir, scenario_id):
    """The P10b correction round's own replay needed a test-only global-
    nanvar substitution to match MATLAB's `updates_*_out`; this uses the
    REAL `em_lsim_minibatch` driver, whose covariance-ridge policy is
    production code (`compute_global_nanvar`/`_substitute_global_nanvar`,
    §2.6) -- no monkeypatch, no test-side substitution, just the driver as
    a caller would use it, replaying the fixture's own exact schedule via
    `schedule=`."""
    fx = load_p10a_scheduling_fixture(fixtures_dir, scenario_id)
    structure = LSIMStructure(
        channel_state_num=tuple(fx.channel_num_states.tolist()),
        channel_obs_dim=tuple(fx.dim_observation.tolist()),
        num_gmm_component=tuple(fx.num_gmm_component.tolist()),
    )
    sigma_diag = bool(fx.sigma_diag)
    fbc = FbConstants(variant=fx.variant)
    Y_trials = fx.Y_trials()
    schedule = _schedule_from_p10a_fixture(fx)

    params0 = LSIMParams.from_packed(
        structure, pi_0=fx.updates_pi_0_in[0], transition=fx.updates_transition_in[0], coupling=fx.updates_coupling_in[0],
        P_all=fx.updates_P_all_in[0], mu_all=fx.updates_mu_all_in[0], sigma_all=fx.updates_sigma_all_in[0], sigma_diag=sigma_diag,
    )

    result = em_lsim_minibatch(structure, params0, Y_trials, fb_constants=fbc, schedule=schedule, backend="numpy", validate="off")

    assert len(result.updates) == fx.n_updates
    _PACKED_KEY_OF = {
        "pi_0": "pi_0", "transition": "transitions_matrices", "coupling": "coupling_tetha_IM",
        "P_all": "P_all", "mu_all": "mu_all", "sigma_all": "sigma_all",
    }
    for u_idx, update in enumerate(result.updates):
        packed_out = update.params_out.to_packed()
        for field, key in _PACKED_KEY_OF.items():
            got = packed_out[key]
            expected = getattr(fx, f"updates_{field}_out")[u_idx]
            np.testing.assert_allclose(
                got, expected, rtol=1e-6, atol=1e-8,
                err_msg=f"{scenario_id} update {u_idx}: {field} != stored updates_{field}_out",
            )
        # every update's own params_in must chain from the PREVIOUS update's
        # own params_out exactly -- the driver commits atomically before the
        # next mini-batch, so this is a property of the driver's own loop,
        # not just of the stored fixture.
        if u_idx > 0:
            _packed_close(update.params_in, result.updates[u_idx - 1].params_out, rtol=0, atol=0)

    # the driver's own final params must equal the LAST update's own params_out
    _packed_close(result.params, result.updates[-1].params_out, rtol=0, atol=0)


@pytest.mark.parametrize("scenario_id", ["diag", "uneven", "fast"])
def test_golden_schedule_replay_training_batch_loglik_matches_pretrial_loglik(fixtures_dir, scenario_id):
    """training_batch_loglik (the pre-update, params_in-scored likelihood
    for that update's own trials) must match the fixture's own stored
    updates_pretrial_loglik sum exactly -- both are the same quantity.
    training_batch_loglik_by_epoch (ONE value per epoch, §P10c correction
    round) must match fx.log_likelyhood_trace value-for-value, per epoch --
    not merely in total -- and training_batch_loglik_total must match its
    whole-run sum."""
    fx = load_p10a_scheduling_fixture(fixtures_dir, scenario_id)
    structure = LSIMStructure(
        channel_state_num=tuple(fx.channel_num_states.tolist()), channel_obs_dim=tuple(fx.dim_observation.tolist()),
        num_gmm_component=tuple(fx.num_gmm_component.tolist()),
    )
    sigma_diag = bool(fx.sigma_diag)
    fbc = FbConstants(variant=fx.variant)
    Y_trials = fx.Y_trials()
    schedule = _schedule_from_p10a_fixture(fx)
    params0 = LSIMParams.from_packed(
        structure, pi_0=fx.updates_pi_0_in[0], transition=fx.updates_transition_in[0], coupling=fx.updates_coupling_in[0],
        P_all=fx.updates_P_all_in[0], mu_all=fx.updates_mu_all_in[0], sigma_all=fx.updates_sigma_all_in[0], sigma_diag=sigma_diag,
    )
    result = em_lsim_minibatch(structure, params0, Y_trials, fb_constants=fbc, schedule=schedule, backend="numpy", validate="off")

    for u_idx, update in enumerate(result.updates):
        n_tr = int(fx.updates_n_trials[u_idx])
        expected = float(np.sum(fx.updates_pretrial_loglik[u_idx, :n_tr]))
        np.testing.assert_allclose(update.training_batch_loglik, expected, rtol=1e-9, atol=1e-9)

    assert len(result.training_batch_loglik_by_epoch) == fx.max_itration
    np.testing.assert_allclose(result.training_batch_loglik_by_epoch, fx.log_likelyhood_trace, rtol=1e-9, atol=1e-9)

    expected_total = float(fx.log_likelyhood_trace.sum())
    np.testing.assert_allclose(result.training_batch_loglik_total, expected_total, rtol=1e-9, atol=1e-9)


def test_training_batch_loglik_by_epoch_matches_independently_recomputed_per_epoch_sums():
    """Synthetic, non-P10a check with >=2 epochs and >1 mini-batch per
    epoch: training_batch_loglik_by_epoch[e] must equal the independently-
    recomputed sum of training_batch_loglik over exactly that epoch's own
    updates -- not the whole run's total (the bug this correction fixes:
    the prior training_batch_loglik_sum property was documented as a
    per-epoch analogue while actually summing across every epoch)."""
    rng = np.random.default_rng(60)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15, 9, 11])
    max_iterations = 3
    num_minibatches = 2

    result = em_lsim_minibatch(
        structure, params, Y_trials, max_iterations=max_iterations, num_minibatches=num_minibatches, shuffle=True,
        random_state=5, backend="numpy", validate="off",
    )

    assert len(result.updates) == max_iterations * num_minibatches
    by_epoch = result.training_batch_loglik_by_epoch
    assert len(by_epoch) == max_iterations

    for e in range(1, max_iterations + 1):
        independently_recomputed = sum(u.training_batch_loglik for u in result.updates if u.epoch_idx == e)
        np.testing.assert_allclose(by_epoch[e - 1], independently_recomputed, rtol=0, atol=0)

    # the per-epoch values are NOT all equal to the whole-run total (a
    # trivial/degenerate check would pass even with the OLD, buggy
    # semantics if every epoch happened to contribute equally)
    total = result.training_batch_loglik_total
    np.testing.assert_allclose(total, sum(by_epoch), rtol=1e-12, atol=1e-12)
    assert not all(np.isclose(v, total) for v in by_epoch)


# ==========================================================================
# 2. num_minibatches=1 update-SEQUENCE non-regression against em_lsim
#
# Narrower claim than "reproduces em_lsim" (§P10c correction round): with
# check_convergence=False, num_minibatches=1/shuffle=False/backend='numpy'/
# n_jobs=1 reproduces the exact PER-ITERATION E-step/M-step params_out
# em_lsim's own iter_hook exposes -- confirmed against every intermediate
# iteration, not just the last. This does NOT claim to reproduce em_lsim's
# own RETURNED EmResult.params in general: that is a best-iterate snapshot
# (can be an earlier iteration than the last) and em_lsim can stop early
# via its own convergence check -- neither exists in em_lsim_minibatch at
# all (P10d's job). The two happen to coincide here only because
# check_convergence=False and the toy model's own loglik keeps improving
# through the last iteration, making it the best one too.
# ==========================================================================


def test_num_minibatches_1_matches_em_lsim_update_sequence_byte_identical():
    """With shuffle=False, backend='numpy', n_jobs=1, and em_lsim's own
    check_convergence=False: num_minibatches=1 reduces every epoch to one
    mini-batch containing every trial in dataset order -- the exact same
    e_step/m_step calls em_lsim itself makes each iteration. Compared
    against every intermediate iteration's own params_out (via em_lsim's
    iter_hook), not merely em_lsim's own returned best-iterate EmResult."""
    rng = np.random.default_rng(42)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15, 9, 11, 7])
    fbc = FbConstants(variant="sed")
    max_iterations = 5

    captured: list[LSIMParams] = []
    em_lsim(
        structure, params.copy(), Y_trials, max_iterations=max_iterations, fb_constants=fbc, backend="numpy",
        check_convergence=False, validate="off", iter_hook=lambda itr, p, ll, aic, bic: captured.append(p.copy()),
    )

    result = em_lsim_minibatch(
        structure, params.copy(), Y_trials, fb_constants=fbc, max_iterations=max_iterations, num_minibatches=1,
        shuffle=False, backend="numpy", n_jobs=1, validate="off",
    )

    assert len(result.updates) == max_iterations
    for itr in range(max_iterations):
        _packed_close(result.updates[itr].params_out, captured[itr], rtol=0, atol=0)
    _packed_close(result.params, captured[-1], rtol=0, atol=0)


def test_num_minibatches_1_shuffle_true_matches_em_lsim_update_sequence_within_p4_tolerance():
    """Same as above but with shuffle=True (the driver's own default) --
    reordering the single mini-batch's own trials changes floating-point
    summation order slightly, so this is checked at P4's own established
    tolerance (rtol=1e-8,atol=1e-9), not exact equality."""
    rng = np.random.default_rng(43)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15, 9, 11, 7])
    fbc = FbConstants(variant="sed")
    max_iterations = 5

    captured: list[LSIMParams] = []
    em_lsim(
        structure, params.copy(), Y_trials, max_iterations=max_iterations, fb_constants=fbc, backend="numpy",
        check_convergence=False, validate="off", iter_hook=lambda itr, p, ll, aic, bic: captured.append(p.copy()),
    )

    result = em_lsim_minibatch(
        structure, params.copy(), Y_trials, fb_constants=fbc, max_iterations=max_iterations, num_minibatches=1,
        shuffle=True, random_state=0, backend="numpy", n_jobs=1, validate="off",
    )

    _packed_close(result.params, captured[-1], rtol=1e-8, atol=1e-9)


# ==========================================================================
# 3. Seed reproducibility and different-seed schedule divergence
# ==========================================================================


def test_same_seed_reproduces_identical_schedule_and_result():
    rng = np.random.default_rng(1)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15, 9, 11])

    r1 = em_lsim_minibatch(structure, params, Y_trials, max_iterations=3, num_minibatches=2, random_state=7, backend="numpy", validate="off")
    r2 = em_lsim_minibatch(structure, params, Y_trials, max_iterations=3, num_minibatches=2, random_state=7, backend="numpy", validate="off")

    assert r1.schedule.epochs == r2.schedule.epochs
    _packed_close(r1.params, r2.params, rtol=0, atol=0)


def test_different_seeds_produce_different_schedules():
    rng = np.random.default_rng(1)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15, 9, 11])

    r1 = em_lsim_minibatch(structure, params, Y_trials, max_iterations=3, num_minibatches=2, random_state=1, backend="numpy", validate="off")
    r2 = em_lsim_minibatch(structure, params, Y_trials, max_iterations=3, num_minibatches=2, random_state=2, backend="numpy", validate="off")

    assert r1.schedule.epochs != r2.schedule.epochs


def test_random_state_accepts_generator_instance_directly():
    """random_state may be an already-constructed np.random.Generator, not
    only an int seed -- and using the SAME Generator instance twice must
    NOT reproduce the same schedule (the generator's own internal state
    advances), unlike passing the same int seed twice."""
    rng = np.random.default_rng(1)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15, 9])

    shared_gen = np.random.default_rng(99)
    r1 = em_lsim_minibatch(structure, params, Y_trials, max_iterations=2, num_minibatches=2, random_state=shared_gen, backend="numpy", validate="off")
    r2 = em_lsim_minibatch(structure, params, Y_trials, max_iterations=2, num_minibatches=2, random_state=shared_gen, backend="numpy", validate="off")
    assert r1.schedule.epochs != r2.schedule.epochs  # same Generator object, but its state advanced between calls


def test_never_touches_global_numpy_random_state():
    """Whitebox: monkeypatch np.random's global legacy functions to raise if
    ever called -- em_lsim_minibatch must use only the explicit Generator
    it is given (or constructs itself), never global randomness."""
    rng = np.random.default_rng(2)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15])

    import unittest.mock

    def _boom(*a, **k):
        raise AssertionError("must not touch numpy's global random state")

    with unittest.mock.patch.object(np.random, "seed", _boom), unittest.mock.patch.object(np.random, "permutation", _boom), unittest.mock.patch.object(np.random, "shuffle", _boom):
        em_lsim_minibatch(structure, params, Y_trials, max_iterations=2, num_minibatches=2, random_state=3, backend="numpy", validate="off")


# ==========================================================================
# 4. Uneven partitions and num_minibatches=N
# ==========================================================================


def test_uneven_partition_covers_every_trial_exactly_once_per_epoch():
    rng = np.random.default_rng(4)
    num_trials = 7
    sched = generate_minibatch_schedule(num_trials, max_iterations=3, num_minibatches=3, shuffle=True, rng=rng)
    validate_minibatch_schedule(sched, num_trials=num_trials, max_iterations=3)
    for epoch in sched.epochs:
        sizes = sorted(len(mb) for mb in epoch)
        assert sizes == [2, 2, 3]  # 7 trials / 3 minibatches -- MATLAB round(linspace(...))'s own split
        assert sorted(i for mb in epoch for i in mb) == list(range(num_trials))


@pytest.mark.parametrize(
    "num_trials,num_minibatches,expected_sizes",
    [
        (5, 2, [3, 2]),  # MATLAB round(linspace(0,5,3)) = [0,3,5] -- half-tie at 2.5 breaks AWAY from zero, to 3
        (3, 2, [2, 1]),  # linspace(0,3,3) = [0,1.5,3] -- half-tie at 1.5 -> 2
        (9, 2, [5, 4]),  # linspace(0,9,3) = [0,4.5,9] -- half-tie at 4.5 -> 5
    ],
)
def test_generate_schedule_matlab_half_tie_rounding(num_trials, num_minibatches, expected_sizes):
    """np.round breaks ties to EVEN (banker's rounding): np.round(2.5)==2,
    which would give a [2,3] split here, not MATLAB's own [3,2] -- a real,
    measured divergence (§P10c correction round), not a theoretical one.
    Every edge in this computation is non-negative, so MATLAB's own
    half-away-from-zero tie-break is exactly floor(x+0.5); checked here via
    the exact resulting batch SIZES (derived from MATLAB's own edges, not
    merely "some valid partition") and, with shuffle=False, exact ORDERED
    membership (dataset order, unshuffled) for each mini-batch."""
    rng = np.random.default_rng(0)
    sched = generate_minibatch_schedule(num_trials, max_iterations=1, num_minibatches=num_minibatches, shuffle=False, rng=rng)
    sizes = [len(mb) for mb in sched.epochs[0]]
    assert sizes == expected_sizes

    # ordered membership, unshuffled: mini-batch b is exactly
    # range(sum(expected_sizes[:b]), sum(expected_sizes[:b+1]))
    cursor = 0
    for mb, size in zip(sched.epochs[0], expected_sizes):
        assert mb == tuple(range(cursor, cursor + size))
        cursor += size
    assert cursor == num_trials


def test_num_minibatches_equals_N_gives_one_trial_per_minibatch():
    rng = np.random.default_rng(5)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15, 9, 11])
    result = em_lsim_minibatch(
        structure, params, Y_trials, max_iterations=2, num_minibatches=len(Y_trials), shuffle=True, random_state=0,
        backend="numpy", validate="off",
    )
    for epoch in result.schedule.epochs:
        assert all(len(mb) == 1 for mb in epoch)
        assert len(epoch) == len(Y_trials)
    assert len(result.updates) == 2 * len(Y_trials)


@pytest.mark.parametrize("num_minibatches", [50, 1000])
def test_num_minibatches_greater_than_N_warns_and_clamps(num_minibatches):
    rng = np.random.default_rng(6)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15, 9, 11])
    with pytest.warns(UserWarning, match="clamping"):
        result = em_lsim_minibatch(
            structure, params, Y_trials, max_iterations=1, num_minibatches=num_minibatches, shuffle=False,
            backend="numpy", validate="off",
        )
    assert len(result.schedule.epochs[0]) == len(Y_trials)


def test_num_minibatches_below_one_raises():
    rng = np.random.default_rng(6)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12])
    with pytest.raises(ValueError, match="num_minibatches"):
        em_lsim_minibatch(structure, params, Y_trials, max_iterations=1, num_minibatches=0, backend="numpy", validate="off")


# ==========================================================================
# Schedule validation contract
# ==========================================================================


def test_schedule_wrong_epoch_count_raises():
    sched = MinibatchSchedule(epochs=((tuple(range(4)),),))
    with pytest.raises(ValueError, match="epochs"):
        validate_minibatch_schedule(sched, num_trials=4, max_iterations=2)


def test_schedule_out_of_range_index_raises():
    sched = MinibatchSchedule(epochs=(((0, 1, 2, 99),),))
    with pytest.raises(ValueError, match="out of range"):
        validate_minibatch_schedule(sched, num_trials=4)


def test_schedule_negative_index_raises():
    sched = MinibatchSchedule(epochs=(((0, 1, -1, 3),),))
    with pytest.raises(ValueError, match="out of range"):
        validate_minibatch_schedule(sched, num_trials=4)


def test_schedule_zero_epochs_raises():
    with pytest.raises(ValueError, match="zero epochs"):
        validate_minibatch_schedule(MinibatchSchedule(epochs=()), num_trials=4)


def test_explicit_schedule_conflicting_max_iterations_raises():
    rng = np.random.default_rng(0)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15])
    sched = MinibatchSchedule(epochs=((tuple(range(4)),),))
    with pytest.raises(ValueError, match="max_iterations"):
        em_lsim_minibatch(structure, params, Y_trials, schedule=sched, max_iterations=5, backend="numpy", validate="off")


def test_no_schedule_and_no_max_iterations_raises():
    rng = np.random.default_rng(0)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12])
    with pytest.raises(ValueError, match="max_iterations"):
        em_lsim_minibatch(structure, params, Y_trials, backend="numpy", validate="off")


# ==========================================================================
# Hardened schedule contract (§P10c correction round)
# ==========================================================================


@pytest.mark.parametrize("bad_index", [0.0, 1.0, True, False])
def test_schedule_rejects_float_and_bool_trial_indices_at_construction(bad_index):
    """0.0 == 0 and False == 0 in Python, so a value-only range/membership
    check would silently accept these -- MinibatchSchedule.__post_init__
    must reject the TYPE outright, at construction, before any value-level
    validate_minibatch_schedule check even runs."""
    with pytest.raises(ValueError, match="int"):
        MinibatchSchedule(epochs=((tuple([bad_index, 1, 2]),),))


def test_schedule_accepts_and_canonicalizes_np_integer_indices():
    idx = np.int64(3)
    sched = MinibatchSchedule(epochs=(((idx, np.int32(0)),),))
    assert all(type(i) is int for epoch in sched.epochs for mb in epoch for i in mb)
    assert sched.epochs == (((3, 0),),)


def test_schedule_deeply_canonicalizes_nested_lists_to_tuples():
    """A frozen dataclass only freezes its own top-level field assignment --
    it does nothing to stop a caller's own mutable nested list from being
    mutated later out from under an already-constructed schedule. Passing
    plain lists must be accepted (not merely tolerated) and converted to a
    genuine, deeply-immutable tuple-of-tuples-of-tuples."""
    mutable_source = [[0, 1], [2, 3]]
    sched = MinibatchSchedule(epochs=[mutable_source])
    assert isinstance(sched.epochs, tuple)
    assert all(isinstance(mb, tuple) for mb in sched.epochs[0])
    mutable_source[0].append(99)  # mutating the ORIGINAL list after construction
    assert sched.epochs == (((0, 1), (2, 3)),)  # ...must not affect the already-constructed schedule


def test_schedule_num_minibatches_property_enforced_constant_by_validation():
    """A schedule with a genuinely varying per-epoch mini-batch count is
    still representable (construction doesn't reject it -- only VALUE-level
    coverage/count checks belong in validate_minibatch_schedule), but must
    be rejected by validate_minibatch_schedule specifically because it would
    make .num_minibatches's single-number summary misleading."""
    varying = MinibatchSchedule(epochs=(((0, 1), (2, 3)), ((0, 1, 2, 3),)))
    assert varying.num_minibatches == 2  # first epoch's own count -- NOT validated yet
    with pytest.raises(ValueError, match="varies across epochs"):
        validate_minibatch_schedule(varying, num_trials=4)


# --------------------------------------------------------------------------
# validate_minibatch_schedule's OWN argument validation (§P10c final
# correction round): schedule must genuinely be a MinibatchSchedule,
# num_trials/max_iterations must be positive non-bool ints -- a bool or
# float that happens to compare equal to the real value (e.g. max_
# iterations=True or 1.0 against a genuine one-epoch schedule) must not
# silently pass a bare `!=` comparison.
# --------------------------------------------------------------------------


def test_validate_schedule_rejects_non_schedule_object():
    with pytest.raises(ValueError, match="MinibatchSchedule"):
        validate_minibatch_schedule([[[0]]], num_trials=1)


@pytest.mark.parametrize("bad_num_trials", [0, -1, 1.5, "1", True, False])
def test_validate_schedule_rejects_bad_num_trials(bad_num_trials):
    sched = MinibatchSchedule(epochs=(((0,),),))
    with pytest.raises(ValueError, match="num_trials"):
        validate_minibatch_schedule(sched, num_trials=bad_num_trials)


@pytest.mark.parametrize("bad_max_iterations", [0, -1, 1.5, "1", True, False])
def test_validate_schedule_rejects_bad_max_iterations(bad_max_iterations):
    """The concrete failure this guards against: with a genuine one-epoch
    schedule, max_iterations=True or max_iterations=1.0 both compare equal
    to the int 1 via a bare `!=` check and would otherwise silently pass."""
    sched = MinibatchSchedule(epochs=(((0,),),))
    with pytest.raises(ValueError, match="max_iterations"):
        validate_minibatch_schedule(sched, num_trials=1, max_iterations=bad_max_iterations)


def test_validate_schedule_accepts_genuine_max_iterations_one():
    """Sanity: the real, valid case (max_iterations=1, a genuine int) must
    still pass -- the new type check must not reject legitimate input."""
    sched = MinibatchSchedule(epochs=(((0,),),))
    validate_minibatch_schedule(sched, num_trials=1, max_iterations=1)  # must not raise


# --------------------------------------------------------------------------
# _resolve_rng's strict typing (§P10c final correction round): None, a
# genuine non-bool int/np.integer seed, or an np.random.Generator only.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad_random_state", [True, False, 1.5, "seed", object()])
def test_random_state_rejects_unsupported_types(bad_random_state):
    rng = np.random.default_rng(0)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12])
    with pytest.raises(ValueError, match="random_state"):
        em_lsim_minibatch(
            structure, params, Y_trials, max_iterations=1, num_minibatches=1, random_state=bad_random_state,
            backend="numpy", validate="off",
        )


def test_random_state_rejects_legacy_RandomState():
    rng = np.random.default_rng(0)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12])
    legacy = np.random.RandomState(0)
    with pytest.raises(ValueError, match="random_state"):
        em_lsim_minibatch(
            structure, params, Y_trials, max_iterations=1, num_minibatches=1, random_state=legacy,
            backend="numpy", validate="off",
        )


def test_random_state_accepts_none_int_and_generator():
    rng = np.random.default_rng(0)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12])
    for valid in (None, 0, np.int64(3), np.random.default_rng(1)):
        result = em_lsim_minibatch(
            structure, params, Y_trials, max_iterations=1, num_minibatches=1, random_state=valid,
            backend="numpy", validate="off",
        )
        assert len(result.updates) == 1


@pytest.mark.parametrize("bad", [0, -1, 1.5, "2", True])
def test_generate_schedule_rejects_non_positive_int_counts(bad):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="max_iterations|num_minibatches"):
        generate_minibatch_schedule(6, max_iterations=bad, num_minibatches=1, shuffle=True, rng=rng)
    with pytest.raises(ValueError, match="max_iterations|num_minibatches"):
        generate_minibatch_schedule(6, max_iterations=1, num_minibatches=bad, shuffle=True, rng=rng)


def test_generate_schedule_rejects_non_positive_num_trials():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="num_trials"):
        generate_minibatch_schedule(0, max_iterations=1, num_minibatches=1, shuffle=True, rng=rng)


def test_generate_schedule_rejects_non_bool_shuffle():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="shuffle"):
        generate_minibatch_schedule(4, max_iterations=1, num_minibatches=1, shuffle="yes", rng=rng)


def test_generate_schedule_rejects_non_generator_rng():
    with pytest.raises(ValueError, match="rng"):
        generate_minibatch_schedule(4, max_iterations=1, num_minibatches=1, shuffle=True, rng=0)


def test_empty_Y_trials_rejected_cleanly():
    rng = np.random.default_rng(0)
    structure, params = _toy_model(rng)
    with pytest.raises(ValueError, match="at least one trial"):
        em_lsim_minibatch(structure, params, [], max_iterations=1, num_minibatches=1, backend="numpy", validate="off")


@pytest.mark.parametrize("conflicting_kwarg", ["num_minibatches", "shuffle", "random_state"])
def test_schedule_with_generation_controls_raises_instead_of_ignoring(conflicting_kwarg):
    """num_minibatches/shuffle/random_state control schedule GENERATION --
    explicitly supplying one of them together with schedule= must raise,
    not be silently ignored, since the caller's own explicit request would
    otherwise have no effect at all with no indication why."""
    rng = np.random.default_rng(0)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15])
    sched = MinibatchSchedule(epochs=((tuple(range(4)),),))
    kwargs = {"num_minibatches": 2, "shuffle": False, "random_state": 0}
    with pytest.raises(ValueError, match=conflicting_kwarg):
        em_lsim_minibatch(structure, params, Y_trials, schedule=sched, backend="numpy", validate="off", **{conflicting_kwarg: kwargs[conflicting_kwarg]})


def test_schedule_without_generation_controls_does_not_raise():
    """The mirror image of the test above: schedule= ALONE (no num_
    minibatches/shuffle/random_state at all) must work exactly as before --
    the sentinel-default mechanism must not accidentally treat "omitted" as
    "conflicting"."""
    rng = np.random.default_rng(0)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15])
    sched = MinibatchSchedule(epochs=((tuple(range(4)),),))
    result = em_lsim_minibatch(structure, params, Y_trials, schedule=sched, backend="numpy", validate="off")
    assert len(result.updates) == 1


# ==========================================================================
# 5. Serial/parallel NumPy and Numba parity (the driver's own loop)
# ==========================================================================


@pytest.mark.parametrize("executor", EXECUTORS)
def test_driver_serial_vs_parallel_numpy_matches(executor):
    rng = np.random.default_rng(10)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15, 9, 11, 7, 13])

    serial = em_lsim_minibatch(
        structure, params, Y_trials, max_iterations=3, num_minibatches=2, shuffle=True, random_state=0,
        backend="numpy", n_jobs=1, validate="off",
    )
    parallel = em_lsim_minibatch(
        structure, params, Y_trials, max_iterations=3, num_minibatches=2, shuffle=True, random_state=0,
        backend="numpy", n_jobs=3, executor=executor, validate="off",
    )
    assert serial.schedule.epochs == parallel.schedule.epochs  # same seed -- same schedule regardless of executor
    _packed_close(serial.params, parallel.params, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("executor", EXECUTORS)
def test_driver_serial_vs_parallel_numba_matches(executor):
    pytest.importorskip("numba")
    rng = np.random.default_rng(11)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15, 9, 11, 7, 13])

    serial = em_lsim_minibatch(
        structure, params, Y_trials, max_iterations=3, num_minibatches=2, shuffle=True, random_state=0,
        backend="numba", n_jobs=1, validate="off",
    )
    parallel = em_lsim_minibatch(
        structure, params, Y_trials, max_iterations=3, num_minibatches=2, shuffle=True, random_state=0,
        backend="numba", n_jobs=3, executor=executor, validate="off",
    )
    _packed_close(serial.params, parallel.params, rtol=1e-8, atol=1e-10)


# ==========================================================================
# 6. Torch CPU float64 parity
# ==========================================================================


def test_torch_matches_numpy_same_schedule():
    pytest.importorskip("torch")
    rng = np.random.default_rng(12)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15, 9, 11])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        numpy_result = em_lsim_minibatch(
            structure, params, Y_trials, max_iterations=3, num_minibatches=2, shuffle=True, random_state=0,
            backend="numpy", validate="off",
        )
        torch_result = em_lsim_minibatch(
            structure, params, Y_trials, max_iterations=3, num_minibatches=2, shuffle=True, random_state=0,
            backend="torch", validate="off",
        )

    assert numpy_result.schedule.epochs == torch_result.schedule.epochs
    assert len(numpy_result.updates) == len(torch_result.updates)
    for u_np, u_t in zip(numpy_result.updates, torch_result.updates):
        np.testing.assert_allclose(u_t.training_batch_loglik, u_np.training_batch_loglik, rtol=1e-6, atol=1e-6)
        _packed_close(u_t.params_out, u_np.params_out, rtol=1e-6, atol=1e-8)
    _packed_close(torch_result.params, numpy_result.params, rtol=1e-6, atol=1e-8)


def test_torch_rejects_outer_executor():
    pytest.importorskip("torch")
    rng = np.random.default_rng(13)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8])
    from chmm_lsim.backends.base import UnsupportedCombination

    with pytest.raises(UnsupportedCombination, match="torch"):
        em_lsim_minibatch(
            structure, params, Y_trials, max_iterations=1, num_minibatches=1, backend="torch", n_jobs=2,
            executor="thread", validate="off",
        )


def test_torch_chunk_size_does_not_change_update_count_or_placement():
    """backend_options' chunk_size/max_memory_bytes are execution-memory
    controls only -- must never create extra M-steps: the same schedule
    run with two very different chunk_size values must produce the exact
    same number of updates, at the exact same schedule."""
    pytest.importorskip("torch")
    rng = np.random.default_rng(14)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15, 9, 11])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        small_chunk = em_lsim_minibatch(
            structure, params, Y_trials, max_iterations=2, num_minibatches=3, shuffle=False, backend="torch",
            backend_options={"chunk_size": 1}, validate="off",
        )
        big_chunk = em_lsim_minibatch(
            structure, params, Y_trials, max_iterations=2, num_minibatches=3, shuffle=False, backend="torch",
            backend_options={"chunk_size": 100}, validate="off",
        )
    assert len(small_chunk.updates) == len(big_chunk.updates) == 6
    _packed_close(small_chunk.params, big_chunk.params, rtol=1e-6, atol=1e-8)


# ==========================================================================
# 7. Zero-transition structural-preservation policy
# ==========================================================================


@pytest.mark.parametrize("sigma_diag", [True, False])
def test_zero_transition_minibatch_preserves_transition_and_coupling(sigma_diag):
    rng = np.random.default_rng(20)
    structure, params = _toy_model(rng, sigma_diag=sigma_diag)
    Y_trials = _toy_trials(rng, structure, [1, 1, 1, 1])
    schedule = MinibatchSchedule(epochs=((tuple(range(4)),),))

    result = em_lsim_minibatch(structure, params, Y_trials, schedule=schedule, backend="numpy", validate="off")
    update = result.updates[0]
    assert update.n_transitions == 0
    np.testing.assert_array_equal(update.params_out.transition, update.params_in.transition)
    np.testing.assert_array_equal(update.params_out.coupling, update.params_in.coupling)
    assert not np.array_equal(update.params_out.pi_0, update.params_in.pi_0)  # pi_0 IS still updated
    assert np.all(np.isfinite(update.params_out.pi_0))
    assert np.all(np.isfinite(update.params_out.transition))
    assert np.all(np.isfinite(update.params_out.coupling))
    for gmm in update.params_out.emissions:
        assert np.all(np.isfinite(gmm.mu))
        assert np.all(np.isfinite(gmm.sigma))
        assert np.all(np.isfinite(gmm.weights))


def test_mixed_epoch_one_zero_transition_minibatch_one_ordinary():
    """One epoch, two mini-batches: the first is entirely T=1 trials (zero
    transitions), the second has ordinary multi-timestep trials -- the
    zero-transition mini-batch preserves transition/coupling from ITS OWN
    params_in (the epoch's starting snapshot), while the ordinary one
    updates them normally from what the first mini-batch already committed."""
    rng = np.random.default_rng(21)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [1, 1, 1]) + _toy_trials(rng, structure, [10, 12, 9])
    schedule = MinibatchSchedule(epochs=(((0, 1, 2), (3, 4, 5)),))

    result = em_lsim_minibatch(structure, params, Y_trials, schedule=schedule, backend="numpy", validate="off")
    zero_tr_update, ordinary_update = result.updates

    assert zero_tr_update.n_transitions == 0
    np.testing.assert_array_equal(zero_tr_update.params_out.transition, params.transition)
    np.testing.assert_array_equal(zero_tr_update.params_out.coupling, params.coupling)

    assert ordinary_update.n_transitions > 0
    # the ordinary update's own params_in IS the zero-transition update's
    # own params_out (committed atomically before this mini-batch) -- so
    # its transition/coupling should have visibly changed from the ORIGINAL
    # params (since it had real transition evidence to learn from), even
    # though the zero-transition step in between changed nothing structural.
    assert not np.array_equal(ordinary_update.params_out.transition, params.transition)
    assert np.all(np.isfinite(result.params.transition))
    assert np.all(np.isfinite(result.params.coupling))


def test_ordinary_full_batch_m_step_unaffected_by_zero_transition_policy(fixtures_dir):
    """m_step's own new params_in= parameter must not change ANY existing
    caller's behavior when n_transitions>0 -- confirmed by re-running the
    existing P4 em-trace gate with and without an (irrelevant, on this
    path) params_in argument and getting byte-identical results."""
    from chmm_lsim.models.lsim.em import e_step, m_step
    from chmm_lsim.learning.em_loop import finalize_expected_stats

    fx = load_em_trace_fixture(fixtures_dir, "c3_het_diag")
    structure = LSIMStructure(
        tuple(fx.channel_num_states.tolist()), tuple(fx.dim_observation.tolist()), tuple(fx.num_gmm_component.tolist())
    )
    sigma_diag = bool(fx.sigma_diag)
    packed = fx.params_in(1)
    params = LSIMParams.from_packed(
        structure, pi_0=packed["pi_0"], transition=packed["transition_matrices_IM"], coupling=packed["coupling_tetha_IM"],
        P_all=packed["P_all"], mu_all=packed["mu_all"], sigma_all=packed["sigma_all"], sigma_diag=sigma_diag,
    )
    stats = finalize_expected_stats(e_step(structure, params, fx.Y_trials, validate="off"), sigma_diag=sigma_diag)
    assert stats.n_transitions > 0

    without = m_step(structure, stats, sigma_diag=sigma_diag)
    with_unused_params_in = m_step(structure, stats, sigma_diag=sigma_diag, params_in=params)
    _packed_close(without, with_unused_params_in, rtol=0, atol=0)


def test_m_step_zero_transitions_without_params_in_raises():
    from chmm_lsim.models.lsim.em import e_step, m_step
    from chmm_lsim.learning.em_loop import finalize_expected_stats

    rng = np.random.default_rng(22)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [1, 1])
    stats = finalize_expected_stats(e_step(structure, params, Y_trials, validate="off"), sigma_diag=params.sigma_diag)
    assert stats.n_transitions == 0
    with pytest.raises(ValueError, match="n_transitions"):
        m_step(structure, stats, sigma_diag=params.sigma_diag)


def test_m_step_zero_transitions_rejects_mismatched_state_partition():
    """Regression case §P10c correction round explicitly calls out: two
    structures can share the same S (total states) and C (channel count)
    while partitioning states across channels differently -- M=(2,3) vs
    M=(3,2), both S=5,C=2 -- so params_in.transition/coupling (shape
    (S,S)/(C,C)) are ARRAY-SHAPE-compatible with the "wrong" structure too.
    m_step must reject this by comparing the actual LSIMStructure objects
    (channel_state_num etc, not just S/C), not merely check array shapes."""
    rng = np.random.default_rng(23)
    structure_a, params_a = _toy_model(rng, M=(2, 3))
    structure_b, params_b = _toy_model(rng, M=(3, 2))
    assert structure_a.S == structure_b.S and structure_a.C == structure_b.C
    assert structure_a != structure_b
    assert params_a.transition.shape == params_b.transition.shape  # shape-compatible, but WRONG structure

    Y_trials = _toy_trials(rng, structure_a, [1, 1])
    stats = finalize_expected_stats(e_step(structure_a, params_a, Y_trials, validate="off"), sigma_diag=params_a.sigma_diag)
    assert stats.n_transitions == 0

    with pytest.raises(ValueError, match="structure"):
        m_step(structure_a, stats, sigma_diag=params_a.sigma_diag, params_in=params_b)

    # sanity: the SAME structure's own params_in is accepted (no false positive)
    out = m_step(structure_a, stats, sigma_diag=params_a.sigma_diag, params_in=params_a)
    np.testing.assert_array_equal(out.transition, params_a.transition)


# ==========================================================================
# 8. Large-offset and missing-data covariance stability under mini-batch
#    training specifically
# ==========================================================================


def test_missing_data_covariance_stability_under_minibatch_training(fixtures_dir):
    """c3_het_full_nan (full covariance + missingness) run through 3 epochs
    of 2-minibatch training must stay finite and produce valid (positive-
    definite) covariance matrices throughout."""
    fx = load_em_trace_fixture(fixtures_dir, "c3_het_full_nan")
    structure = LSIMStructure(
        tuple(fx.channel_num_states.tolist()), tuple(fx.dim_observation.tolist()), tuple(fx.num_gmm_component.tolist())
    )
    sigma_diag = bool(fx.sigma_diag)
    packed = fx.params_in(1)
    params = LSIMParams.from_packed(
        structure, pi_0=packed["pi_0"], transition=packed["transition_matrices_IM"], coupling=packed["coupling_tetha_IM"],
        P_all=packed["P_all"], mu_all=packed["mu_all"], sigma_all=packed["sigma_all"], sigma_diag=sigma_diag,
    )
    Y_trials = fx.Y_trials * 3  # only 1 real trial in this fixture -- replicate for a >1-trial mini-batch schedule
    assert len(Y_trials) >= 2

    result = em_lsim_minibatch(
        structure, params, Y_trials, max_iterations=3, num_minibatches=min(2, len(Y_trials)), shuffle=True,
        random_state=0, backend="numpy", validate="off",
    )
    for update in result.updates:
        for gmm in update.params_out.emissions:
            assert np.all(np.isfinite(gmm.sigma))
            if not gmm.diag:
                for i in range(gmm.M_c):
                    for k in range(gmm.K_c):
                        eigvals = np.linalg.eigvalsh(gmm.sigma[:, :, i, k])
                        assert np.all(eigvals > 0), f"non-positive-definite covariance at state {i}, component {k}"
            else:
                assert np.all(gmm.sigma > 0)


def test_large_offset_stability_under_minibatch_training():
    """Same construction as test_em.py's large-offset stability tests, but
    run through actual mini-batch training (2 mini-batches x 2 epochs) --
    proving compute_global_nanvar's own whole-dataset ridge stays stable at
    a 1e10 offset too, not just a single e_step call."""
    offset = 1e10
    structure = LSIMStructure(channel_state_num=(1,), channel_obs_dim=(1,), num_gmm_component=(1,))
    gmm = ChannelGMM(weights=np.array([[1.0]]), mu=np.array([[[offset]]]), sigma=np.array([[[1e8]]]), diag=True)
    params = LSIMParams(structure=structure, pi_0=np.array([1.0]), transition=np.array([[1.0]]), coupling=np.ones((1, 1)), emissions=[gmm])

    x = np.arange(100.0)
    Y_trials = [(offset + x)[None, :], (offset + 50 + x)[None, :], (offset + x[::-1])[None, :], (offset - 20 + x)[None, :]]

    result = em_lsim_minibatch(
        structure, params, Y_trials, max_iterations=2, num_minibatches=2, shuffle=True, random_state=0,
        backend="numpy", validate="off",
    )
    for update in result.updates:
        sigma = update.params_out.emissions[0].sigma
        assert np.all(np.isfinite(sigma))
        assert np.all(sigma > 0)
        assert np.all(np.abs(update.params_out.emissions[0].mu - offset) < 1000)  # sane, not blown up


# ==========================================================================
# compute_global_nanvar: inference-free rewrite (§P10c correction round)
# ==========================================================================


def _e_step_derived_nanvar(structure, Y_trials):
    """The OLD (removed) implementation's own logic, reproduced here ONLY
    as an independent reference for parity testing -- runs a real E-step
    (any valid params work, since nanvar_y never depends on them) and reads
    finalize_expected_stats's own nanvar_y off the result. Production code
    no longer does this at all (that is exactly what this correction fixes)."""
    S = structure.S
    pi_0 = np.zeros(S)
    transition = np.zeros((S, S))
    for c in range(structure.C):
        sl = structure.state_slice(c)
        width = sl.stop - sl.start
        pi_0[sl] = 1.0 / width
        transition[:, sl] = 1.0 / width
    coupling = np.full((structure.C, structure.C), 1.0 / structure.C)
    emissions = []
    for c in range(structure.C):
        M_c, K_c, d_c = structure.channel_state_num[c], structure.num_gmm_component[c], structure.channel_obs_dim[c]
        emissions.append(ChannelGMM(weights=np.full((M_c, K_c), 1.0 / K_c), mu=np.zeros((d_c, M_c, K_c)), sigma=np.ones((d_c, M_c, K_c)), diag=True))
    params = LSIMParams(structure=structure, pi_0=pi_0, transition=transition, coupling=coupling, emissions=emissions)
    raw = e_step(structure, params, Y_trials, validate="off")
    finalized = finalize_expected_stats(raw, sigma_diag=True)
    return [c.nanvar_y for c in finalized.channels]


@pytest.mark.parametrize("scenario_id", P10A_SCENARIO_IDS)
def test_compute_global_nanvar_matches_e_step_derived_value_on_p10a_fixtures(fixtures_dir, scenario_id):
    """Parity against the prior (E-step-derived) implementation, across all
    six P10a fixtures -- the inference-free rewrite must produce the exact
    same numbers, since nanvar_y never depended on params/gamma in the
    first place; only HOW it's computed changed."""
    fx = load_p10a_scheduling_fixture(fixtures_dir, scenario_id)
    structure = LSIMStructure(
        channel_state_num=tuple(fx.channel_num_states.tolist()), channel_obs_dim=tuple(fx.dim_observation.tolist()),
        num_gmm_component=tuple(fx.num_gmm_component.tolist()),
    )
    Y_trials = fx.Y_trials()

    new = compute_global_nanvar(structure, Y_trials)
    old = _e_step_derived_nanvar(structure, Y_trials)
    assert len(new) == len(old) == structure.C
    for c in range(structure.C):
        np.testing.assert_allclose(new[c], old[c], rtol=1e-12, atol=1e-14, equal_nan=True)


def test_compute_global_nanvar_signature_takes_no_inference_arguments():
    """The simplified public signature: structure and Y_trials, plus (P10f)
    only the keyword-only memory bound `max_chunk_observations` -- no params,
    FbConstants, backend, n_jobs, executor, or backend_options."""
    import inspect

    sig = inspect.signature(compute_global_nanvar)
    assert list(sig.parameters) == ["structure", "Y_trials", "max_chunk_observations"]
    assert sig.parameters["max_chunk_observations"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["max_chunk_observations"].default is None


def test_compute_global_nanvar_never_calls_inference(monkeypatch):
    """Whitebox proof, not an inference from reading the code: monkeypatch
    every inference/backend entry point compute_global_nanvar could
    plausibly reach (e_step, run_backend, and the process/thread executors
    a parallel e_step would use) to raise if ever called, then confirm
    compute_global_nanvar still returns a correct, finite result."""
    import concurrent.futures as cf

    import chmm_lsim.models.lsim.em as em_mod
    from chmm_lsim.backends import base as backends_base

    def _boom(*a, **k):
        raise AssertionError("compute_global_nanvar must never perform inference")

    monkeypatch.setattr(em_mod, "e_step", _boom)
    monkeypatch.setattr(backends_base, "run_backend", _boom)
    monkeypatch.setattr(cf, "ThreadPoolExecutor", _boom)
    monkeypatch.setattr(cf, "ProcessPoolExecutor", _boom)

    rng = np.random.default_rng(50)
    structure, _ = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8])

    result = compute_global_nanvar(structure, Y_trials)
    assert len(result) == structure.C
    for arr in result:
        assert np.all(np.isfinite(arr))


def test_compute_global_nanvar_missing_data():
    rng = np.random.default_rng(51)
    structure = LSIMStructure(channel_state_num=(2,), channel_obs_dim=(1,), num_gmm_component=(1,))
    Y = rng.normal(size=(1, 50))
    Y[0, ::3] = np.nan  # scattered missing
    result = compute_global_nanvar(structure, [Y])
    assert np.all(np.isfinite(result[0]))
    n_observed = int((~np.isnan(Y[0])).sum())
    expected = np.nanvar(Y[0], ddof=1)
    np.testing.assert_allclose(result[0][0], expected, rtol=1e-12)
    assert n_observed > 1


def test_compute_global_nanvar_singleton_observation_is_zero_not_nan():
    """MATLAB's own convention: var([5],[],2,'omitnan')==0, not NaN."""
    structure = LSIMStructure(channel_state_num=(1,), channel_obs_dim=(1,), num_gmm_component=(1,))
    Y = np.array([[5.0]])
    result = compute_global_nanvar(structure, [Y])
    assert result[0][0] == 0.0


def test_compute_global_nanvar_fully_missing_channel_is_nan():
    structure = LSIMStructure(channel_state_num=(1,), channel_obs_dim=(1,), num_gmm_component=(1,))
    Y = np.full((1, 10), np.nan)
    result = compute_global_nanvar(structure, [Y])
    assert np.isnan(result[0][0])


def test_compute_global_nanvar_stable_at_large_offset():
    """Welford's algorithm never forms a raw sum(y**2) moment, so this stays
    accurate even far from zero -- same construction as test_em.py's own
    large-offset stability tests."""
    offset = 1e10
    structure = LSIMStructure(channel_state_num=(1,), channel_obs_dim=(1,), num_gmm_component=(1,))
    x = np.arange(100.0)
    Y_trials = [(offset + x)[None, :], (offset + 50 + x)[None, :]]
    result = compute_global_nanvar(structure, Y_trials)
    combined = np.concatenate([Y[0] for Y in Y_trials])
    expected = np.var(combined, ddof=1)
    np.testing.assert_allclose(result[0][0], expected, rtol=1e-9)


# --------------------------------------------------------------------------
# compute_global_nanvar: observation validation (§P10c final correction
# round) -- runs before the first E-step, so this is the first place a
# malformed Y_trials would otherwise surface only as a confusing downstream
# symptom (a raw indexing error, a silently-wrong slice, a silent contract
# violation, or a silently-computed NaN/Inf variance) rather than a clear,
# immediately-attributable error.
# --------------------------------------------------------------------------


def _tiny_structure():
    return LSIMStructure(channel_state_num=(2, 3), channel_obs_dim=(1, 2), num_gmm_component=(1, 2))


def test_compute_global_nanvar_rejects_empty_Y_trials():
    with pytest.raises(ValueError, match="at least one trial"):
        compute_global_nanvar(_tiny_structure(), [])


def test_compute_global_nanvar_rejects_scalar_Y():
    structure = _tiny_structure()
    with pytest.raises(ValueError, match="2-D"):
        compute_global_nanvar(structure, [np.array(5.0)])


def test_compute_global_nanvar_rejects_1d_Y():
    structure = _tiny_structure()
    with pytest.raises(ValueError, match="2-D"):
        compute_global_nanvar(structure, [np.zeros(10)])


def test_compute_global_nanvar_rejects_too_many_channels():
    structure = _tiny_structure()
    with pytest.raises(ValueError, match="structure.D"):
        compute_global_nanvar(structure, [np.zeros((structure.D + 1, 10))])


def test_compute_global_nanvar_rejects_too_few_channels():
    structure = _tiny_structure()
    with pytest.raises(ValueError, match="structure.D"):
        compute_global_nanvar(structure, [np.zeros((structure.D - 1, 10))])


def test_compute_global_nanvar_rejects_integer_dtype():
    structure = _tiny_structure()
    with pytest.raises(ValueError, match="floating dtype"):
        compute_global_nanvar(structure, [np.zeros((structure.D, 10), dtype=np.int64)])


def test_compute_global_nanvar_rejects_inf():
    structure = _tiny_structure()
    Y = np.zeros((structure.D, 10))
    Y[0, 0] = np.inf
    with pytest.raises(ValueError, match="Inf"):
        compute_global_nanvar(structure, [Y])


def test_compute_global_nanvar_rejects_T_zero():
    structure = _tiny_structure()
    with pytest.raises(ValueError, match="T=0"):
        compute_global_nanvar(structure, [np.zeros((structure.D, 0))])


def test_compute_global_nanvar_rejects_malformed_second_trial_naming_its_index():
    """The FIRST trial is valid; the SECOND is malformed -- the loop must
    not stop checking after trial 0, and the error must name index 1
    (`Y_trials[1]`), not merely fail somewhere unattributed."""
    structure = _tiny_structure()
    good = np.zeros((structure.D, 10))
    bad = np.zeros((structure.D, 10), dtype=np.int64)
    with pytest.raises(ValueError, match=r"Y_trials\[1\]"):
        compute_global_nanvar(structure, [good, bad])


def test_compute_global_nanvar_preserves_legal_nan_missingness():
    """validate_observations only rejects Inf, never NaN -- scattered
    missing data must still be accepted and contribute correctly."""
    structure = _tiny_structure()
    rng = np.random.default_rng(52)
    Y = rng.normal(size=(structure.D, 20))
    Y[0, ::3] = np.nan
    result = compute_global_nanvar(structure, [Y])  # must not raise
    assert len(result) == structure.C


# ==========================================================================
# 9. Worker errors retain the GLOBAL trial index
# ==========================================================================


@pytest.mark.parametrize("executor", EXECUTORS)
def test_worker_error_reports_global_trial_index_not_minibatch_local(executor):
    """A malformed trial at GLOBAL dataset position 5, placed at LOCAL
    position 1 within its own mini-batch, must surface as global index 5,
    not local index 1 -- e_step's own TrialExecutionError only knows the
    local position; em_lsim_minibatch must translate it.

    The fault is a shape/dtype-VALID trial with an extreme observation
    value (`1e6`, versus this toy model's own order-1 `mu`/`sigma`), which
    underflows the emission density to exactly `0.0` everywhere and trips
    `inference.sed`'s own "forward pass produced a NaN log-likelihood"
    check -- NOT a malformed shape/dtype/Inf array, which `compute_global_
    nanvar`'s own upfront `validate_observations` pass (§P10c final
    correction round) now catches BEFORE the mini-batch loop even starts,
    which would no longer exercise this translation logic at all."""
    rng = np.random.default_rng(30)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8, 15, 9, 11, 7, 13])
    bad_global_index = 5
    Y_trials[bad_global_index] = np.full((structure.D, 7), 1e6)

    # schedule: mini-batch 0 = [4,5,6,7] (bad trial at LOCAL position 1);
    # mini-batch 1 = [0,1,2,3]
    schedule = MinibatchSchedule(epochs=(((4, 5, 6, 7), (0, 1, 2, 3)),))

    with pytest.raises(TrialExecutionError) as exc_info:
        em_lsim_minibatch(
            structure, params, Y_trials, schedule=schedule, backend="numpy", n_jobs=2, executor=executor, validate="params",
        )
    assert exc_info.value.trial_index == bad_global_index


def test_worker_error_reports_global_trial_index_serial():
    """Same as above, serial (n_jobs=1) path -- e_step itself doesn't raise
    TrialExecutionError at n_jobs=1 (that's parallel-only machinery), so
    this confirms the ordinary exception still identifies the failing
    trial some other clear way rather than a bare unrelated traceback; not
    a TrialExecutionError requirement at n_jobs=1, just a sanity check that
    the failure is still attributable."""
    rng = np.random.default_rng(31)
    structure, params = _toy_model(rng)
    Y_trials = _toy_trials(rng, structure, [10, 12, 8])
    Y_trials[1] = rng.normal(size=(structure.D + 1, 7))
    schedule = MinibatchSchedule(epochs=(((0, 1, 2),),))
    with pytest.raises(Exception):  # noqa: B017 -- validate="params" raises chmm_lsim.validate.ValidationError directly at n_jobs=1
        em_lsim_minibatch(structure, params, Y_trials, schedule=schedule, backend="numpy", n_jobs=1, validate="params")


# ==========================================================================
# 10. Top-level package export contract
# ==========================================================================


def test_p10c_types_exported_from_top_level_package():
    """em_lsim_minibatch and its own dedicated result/trace/schedule types
    are public API -- must be reachable via `lsim.*`, not only `chmm_lsim.em.*`."""
    import chmm_lsim

    for name, obj in [
        ("em_lsim_minibatch", em_lsim_minibatch),
        ("MinibatchSchedule", MinibatchSchedule),
        ("generate_minibatch_schedule", generate_minibatch_schedule),
        ("validate_minibatch_schedule", validate_minibatch_schedule),
    ]:
        assert getattr(chmm_lsim, name) is obj
        assert name in chmm_lsim.__all__

    from chmm_lsim import MinibatchEmResult, MinibatchUpdate, compute_global_nanvar, em_lsim_minibatch as reimported

    assert reimported is em_lsim_minibatch
    assert MinibatchEmResult is not None and MinibatchUpdate is not None and compute_global_nanvar is not None
