"""Validates every exported fixture against tests/fixtures/SCHEMA.md.

This is P1's own test suite: it does not compare anything against a Python
implementation (there isn't one yet), it checks that what MATLAB exported and
what convert_fixtures.py wrote are internally consistent and match the schema
this document defines. P3 imports tests/fixtures/loader.py directly for the
real parity tests once src/lsim exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import load_em_trace_fixture, load_manifest, load_posterior_fixture  # noqa: E402

REQUIRED_SCENARIO_COUNT = 24  # the canonical P1 scenario grid

# P12 exported the POISE variants across that same grid, in both refinement modes, plus one
# extra scenario that exists only for POISE's exact-zero outgoing-strength branch (the
# canonical grid's `c3_coupling_zero_row` uses a 1e-12 row, built for sed's flag_nan).
CANONICAL_FIXTURE_VARIANTS = {"sed", "fast", "poise", "poise_refined"}
POISE_ONLY_SCENARIOS = {"c3_coupling_zero_row_exact"}
POISE_ONLY_VARIANTS = {"poise", "poise_refined"}

# A fixture-id label maps to the fb_constants it was exported with: the two POISE labels share
# one variant and differ by the refinement flag.
_LABEL_TO_CONSTANTS = {
    "sed": ("sed", False),
    "fast": ("fast", False),
    "poise": ("poise", False),
    "poise_refined": ("poise", True),
}
REQUIRED_AXES = [
    "C=1", "C=2", "C=3", "C=5",
    "M=homog2", "M=heterog",
    "d=all1", "d=all3", "d=mixed",
    "K=1", "K=2", "K=3", "K=mixed",
    "covariance=diag", "covariance=full",
    "NaN=none", "NaN=scattered", "NaN=whole-timestep-one-channel", "NaN=channel-fully-missing",
    "coupling=dense", "coupling=sparse-row", "coupling=near-degenerate-column",
    "transitions=dense", "transitions=sparse-l2r",
    "variant=sed", "variant=fast",
    "T=1", "T=2", "T=3", "T=50", "T=500",
    "trials=1", "trials=3-unequal",
    "low-mass", "sigma-near-singular",
]


def test_seeds_are_recorded_and_distinct(manifest):
    """Regression test for a real P1 bug: every posterior fixture's `seed` field
    was hard-coded to 0 (a comment claimed otherwise). Seeds must be positive
    and distinct per scenario -- distinct because export_fixtures.m assigns one
    literal rng() seed per scenario case (101, 102, 103, ...), so a collision
    or a run of zeros would mean the wiring broke again."""
    seeds = []
    for scenario in manifest["scenarios"]:
        seed = scenario["seed"]
        assert seed > 0, f"{scenario['scenario_id']}: seed={seed}, expected a positive literal seed"
        seeds.append(seed)
    assert len(set(seeds)) == len(seeds), f"duplicate seeds across scenarios: {seeds}"


def test_each_fixtures_seed_matches_its_manifest_entry(fixtures_dir, manifest):
    """The seed=0 bug would NOT have been caught by test_seeds_are_recorded_and_distinct
    alone if the manifest and the .npz files had simply drifted apart (e.g. the
    manifest fixed but the .npz regeneration skipped, or vice versa) -- that
    test only ever looked at the manifest. Cross-check every fixture's own
    stored seed against what the manifest claims for it."""
    for scenario in manifest["scenarios"]:
        for variant, entry in scenario["files"].items():
            fixture_id = Path(entry["path"]).stem
            fx = load_posterior_fixture(fixtures_dir, fixture_id)
            assert fx.seed == scenario["seed"], (
                f"{fixture_id}: file seed={fx.seed} != manifest seed={scenario['seed']}"
            )


def test_manifest_metadata_matches_each_file(fixtures_dir, manifest):
    """The manifest is a separate, hand-enriched summary of what's in each
    .npz (added in the P1 correction pass so SCHEMA.md's example manifest
    structure would actually be true) -- nothing enforces the two stay in
    sync except a test like this one. Checks every metadata field the
    manifest duplicates from the file, not just seed."""
    for scenario in manifest["scenarios"]:
        for variant, entry in scenario["files"].items():
            fixture_id = Path(entry["path"]).stem
            fx = load_posterior_fixture(fixtures_dir, fixture_id)

            assert fx.C == scenario["C"], f"{fixture_id}: C mismatch"
            assert fx.S == scenario["S"], f"{fixture_id}: S mismatch"
            assert fx.D == scenario["D"], f"{fixture_id}: D mismatch"
            assert fx.sigma_diag == scenario["sigma_diag"], f"{fixture_id}: sigma_diag mismatch"
            assert fx.num_trials == scenario["num_trials"], f"{fixture_id}: num_trials mismatch"
            np.testing.assert_array_equal(
                fx.channel_num_states, scenario["channel_num_states"],
                err_msg=f"{fixture_id}: channel_num_states mismatch",
            )
            np.testing.assert_array_equal(
                fx.dim_observation, scenario["dim_observation"],
                err_msg=f"{fixture_id}: dim_observation mismatch",
            )
            np.testing.assert_array_equal(
                fx.num_gmm_component, scenario["num_gmm_component"],
                err_msg=f"{fixture_id}: num_gmm_component mismatch",
            )
            np.testing.assert_array_equal(
                fx.length_observation, scenario["length_observation"],
                err_msg=f"{fixture_id}: length_observation mismatch",
            )

            manifest_fbc = entry["fb_constants"]
            assert fx.fb_constants.variant == manifest_fbc["variant"], f"{fixture_id}: fbc.variant mismatch"
            assert fx.fb_constants.beta_floor == manifest_fbc["beta_floor"], f"{fixture_id}: fbc.beta_floor mismatch"
            assert fx.fb_constants.pi0_reg == manifest_fbc["pi0_reg"], f"{fixture_id}: fbc.pi0_reg mismatch"
            assert fx.fb_constants.f_reg == manifest_fbc["f_reg"], f"{fixture_id}: fbc.f_reg mismatch"
            assert fx.fb_constants.svi_eps == manifest_fbc["svi_eps"], f"{fixture_id}: fbc.svi_eps mismatch"
            assert (fx.sed_diagnostics is not None) == entry["has_sed_diagnostics"], (
                f"{fixture_id}: has_sed_diagnostics mismatch"
            )


def test_manifest_loads(manifest):
    assert manifest["schema_version"].startswith("1.")
    assert "kernel_sha256" in manifest
    assert "constants_sha256" in manifest


def test_coverage_checklist(manifest):
    n = len(manifest["scenarios"])
    if n < REQUIRED_SCENARIO_COUNT:
        pytest.skip(f"only {n}/{REQUIRED_SCENARIO_COUNT} scenarios exported so far "
                    "(canary or partial run) -- full coverage check applies once "
                    "the complete suite is generated")
    canonical = [sc for sc in manifest["scenarios"] if sc["scenario_id"] not in POISE_ONLY_SCENARIOS]
    assert len(canonical) == REQUIRED_SCENARIO_COUNT, (
        f"expected exactly {REQUIRED_SCENARIO_COUNT} canonical scenarios, no historical tuples, "
        f"got {len(canonical)} (P12's poise-only scenarios are counted separately)"
    )
    all_axes = set()
    for scenario in manifest["scenarios"]:
        all_axes.update(scenario["axes_covered"])
    missing = [a for a in REQUIRED_AXES if a not in all_axes]
    assert not missing, f"coverage checklist missing axes: {missing}"


def test_expected_posterior_fixture_grid_when_complete(manifest):
    """Each canonical scenario carries all four variant files (sed, fast and the two POISE
    modes, 96 in total); each P12 poise-only scenario carries exactly the two POISE modes.
    Was `test_exactly_48_posterior_fixtures_when_complete` before P12 added the variants."""
    n_scenarios = len(manifest["scenarios"])
    if n_scenarios < REQUIRED_SCENARIO_COUNT:
        pytest.skip(f"only {n_scenarios}/{REQUIRED_SCENARIO_COUNT} scenarios exported so far")
    canonical = [s for s in manifest["scenarios"] if s["scenario_id"] not in POISE_ONLY_SCENARIOS]
    poise_only = [s for s in manifest["scenarios"] if s["scenario_id"] in POISE_ONLY_SCENARIOS]
    assert len(canonical) == REQUIRED_SCENARIO_COUNT
    n_canonical_files = sum(len(s["files"]) for s in canonical)
    assert n_canonical_files == 4 * REQUIRED_SCENARIO_COUNT == 96, (
        f"expected exactly 96 canonical fixture/variant combinations, got {n_canonical_files}"
    )
    for scenario in canonical:
        assert set(scenario["files"].keys()) == CANONICAL_FIXTURE_VARIANTS, (
            f"{scenario['scenario_id']}: expected {sorted(CANONICAL_FIXTURE_VARIANTS)}, "
            f"got {sorted(scenario['files'].keys())}"
        )
    for scenario in poise_only:
        assert set(scenario["files"].keys()) == POISE_ONLY_VARIANTS, (
            f"{scenario['scenario_id']}: a poise-only scenario must carry exactly "
            f"{sorted(POISE_ONLY_VARIANTS)}, got {sorted(scenario['files'].keys())}"
        )


class TestPosteriorFixture:
    """Parametrized over every posterior_fixture_id found in manifest.json."""

    def test_shapes_and_offsets_consistent(self, fixtures_dir, posterior_fixture_id):
        fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)

        assert fx.state_numbers_index.shape == (fx.C + 1,)
        assert fx.dimension_numbers_index.shape == (fx.C + 1,)
        assert int(fx.state_numbers_index[-1]) == fx.S
        assert int(fx.dimension_numbers_index[-1]) == fx.D
        assert fx.channel_num_states.shape == (fx.C,)
        assert fx.dim_observation.shape == (fx.C,)
        assert int(fx.channel_num_states.sum()) == fx.S
        assert int(fx.dim_observation.sum()) == fx.D

        assert fx.pi_0.shape == (fx.S,)
        assert fx.transitions_matrices.shape == (fx.S, fx.S)
        assert fx.coupling_tetha_IM.shape == (fx.C, fx.C)

        max_M = int(fx.channel_num_states.max())
        max_K = int(fx.num_gmm_component.max())
        assert fx.P_all.shape == (fx.C, max_M, max_K)
        assert fx.mu_all.shape == (fx.D, max_M, max_K)
        if fx.sigma_diag:
            assert fx.sigma_all.shape == (fx.D, max_M, max_K)
        else:
            max_d = int(fx.dim_observation.max())
            assert fx.sigma_all.shape == (fx.D, max_d, max_M, max_K)

        assert fx.length_observation.shape == (fx.num_trials,)
        assert len(fx.trials) == fx.num_trials
        for i, trial in enumerate(fx.trials):
            T_i = int(fx.length_observation[i])
            assert trial.Y.shape == (fx.D, T_i)
            assert trial.alpha.shape == (fx.S, T_i)
            assert trial.beta.shape == (fx.S, T_i)
            assert trial.alpha_T.shape == (fx.S, T_i)
            assert trial.b_c_ot_nc.shape == (fx.S, T_i)
            assert trial.B.shape == (fx.S, T_i)
            assert trial.B_comp.shape == (fx.S, max_K, T_i)

    def test_no_unexpected_nan_or_inf(self, fixtures_dir, posterior_fixture_id):
        fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
        for i, trial in enumerate(fx.trials):
            for name in ("alpha", "beta", "alpha_T", "b_c_ot_nc", "B", "B_comp"):
                arr = getattr(trial, name)
                assert not np.any(np.isinf(arr)), f"{posterior_fixture_id} trial{i}.{name} has Inf"
                assert not np.any(np.isnan(arr)), (
                    f"{posterior_fixture_id} trial{i}.{name} has NaN -- only Y may contain NaN "
                    "(missing data); an alpha/beta/B array with NaN means the P0b exact-zero "
                    "fix did not take effect for this fixture's constants"
                )
            assert np.isfinite(trial.P_O_model)

    def test_fbc_recorded_and_reloadable(self, fixtures_dir, posterior_fixture_id):
        fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
        # The fixture LABEL and the recorded variant coincide for sed/fast/poise, but
        # `poise_refined` is the same `poise` kernel with the refinement flag on -- the label
        # carries the mode, the constants carry the variant.
        expected_variant, expected_refinement = _LABEL_TO_CONSTANTS[fx.variant]
        assert fx.fb_constants.variant == expected_variant
        assert fx.fb_constants.poise_refinement == expected_refinement
        assert np.isfinite(fx.fb_constants.pi0_reg)

    def test_sed_diagnostics_only_where_flagged(self, fixtures_dir, posterior_fixture_id):
        fx = load_posterior_fixture(fixtures_dir, posterior_fixture_id)
        if fx.variant != "sed":
            assert fx.sed_diagnostics is None, (
                f"{posterior_fixture_id}: only the sed variant has delta-solver diagnostics"
            )
        if fx.sed_diagnostics is not None:
            n = fx.sed_diagnostics.t_samples.shape[0]
            assert fx.sed_diagnostics.f.shape == (n, fx.C, fx.C)
            assert fx.sed_diagnostics.h.shape == (n, fx.C)
            assert fx.sed_diagnostics.delta.shape == (n, fx.C, fx.C)
            assert fx.sed_diagnostics.Gz.shape == (n, fx.C, fx.S)
            assert not np.any(np.isnan(fx.sed_diagnostics.f))
            assert not np.any(np.isnan(fx.sed_diagnostics.delta))


def test_c3_sparse_l2r_records_pi0_reg_override(fixtures_dir, manifest):
    scenario = next((s for s in manifest["scenarios"] if s["scenario_id"] == "c3_sparse_l2r"), None)
    if scenario is None:
        pytest.skip("c3_sparse_l2r not exported in this run")
    for variant in sorted(CANONICAL_FIXTURE_VARIANTS):
        fx = load_posterior_fixture(fixtures_dir, f"c3_sparse_l2r__{variant}")
        assert fx.fb_constants.pi0_reg == 0.0, (
            f"c3_sparse_l2r__{variant}: expected the documented pi0_reg=0 override, "
            f"got {fx.fb_constants.pi0_reg} -- the loader must read each fixture's "
            "own constants, never assume the canonical default"
        )
    canonical = load_posterior_fixture(fixtures_dir, "c1_minimal__sed")
    assert canonical.fb_constants.pi0_reg != 0.0, (
        "sanity check: a non-overridden fixture must show the canonical pi0_reg, "
        "otherwise this test can't tell an override from a loader bug"
    )


def test_c3_sparse_l2r_has_exact_zero_alpha_and_no_nan(fixtures_dir, manifest):
    scenario = next((s for s in manifest["scenarios"] if s["scenario_id"] == "c3_sparse_l2r"), None)
    if scenario is None:
        pytest.skip("c3_sparse_l2r not exported in this run")
    fx = load_posterior_fixture(fixtures_dir, "c3_sparse_l2r__sed")
    alpha = fx.trials[0].alpha
    assert np.any(alpha == 0.0), (
        "c3_sparse_l2r is supposed to exercise alpha_next==0 -- if no entry is "
        "exactly zero, the scenario isn't testing what its name claims"
    )
    assert not np.any(np.isnan(fx.trials[0].beta))
    assert not np.any(np.isnan(fx.trials[0].alpha_T))


def test_c3_channel_all_nan_fully_missing_emission_equals_one(fixtures_dir, manifest):
    scenario = next((s for s in manifest["scenarios"] if s["scenario_id"] == "c3_channel_all_nan"), None)
    if scenario is None:
        pytest.skip("c3_channel_all_nan not exported in this run")
    for variant in sorted(CANONICAL_FIXTURE_VARIANTS):
        fx = load_posterior_fixture(fixtures_dir, f"c3_channel_all_nan__{variant}")
        trial = fx.trials[0]
        assert np.any(np.isnan(trial.Y)), f"{variant}: expected Y to contain NaN for this scenario"
        # Channel 0 (per the exporter's scenario definition) is NaN for every t;
        # its emission likelihood block in B must equal exactly 1 at every t.
        c = 0
        lo, hi = int(fx.state_numbers_index[c]), int(fx.state_numbers_index[c + 1])
        block = trial.B[lo:hi, :]
        np.testing.assert_array_equal(block, np.ones_like(block))


class TestEmTraceFixture:
    def test_manifest_metadata_matches_file(self, fixtures_dir, manifest, em_trace_scenario_id):
        entry = next(t for t in manifest["em_traces"] if t["scenario_id"] == em_trace_scenario_id)
        fx = load_em_trace_fixture(fixtures_dir, em_trace_scenario_id)
        assert fx.seed == entry["seed"], f"em_trace__{em_trace_scenario_id}: seed mismatch"
        assert fx.C == entry["C"], f"em_trace__{em_trace_scenario_id}: C mismatch"
        assert len(fx.iterations) == entry["num_iters"], (
            f"em_trace__{em_trace_scenario_id}: num_iters mismatch"
        )

    def test_field_names_are_params_out_and_params_in_qualified(self, fixtures_dir, em_trace_scenario_id):
        npz_path = fixtures_dir / f"em_trace__{em_trace_scenario_id}.npz"
        d = np.load(npz_path, allow_pickle=False)
        num_iters = int(d["num_iters"])
        for k in range(1, num_iters + 1):
            assert f"em_params_out_pi0_iter{k}" in d.files
            assert f"em_loglik_params_in_iter{k}" in d.files
            assert f"em_AIC_params_in_iter{k}" in d.files
            assert f"em_BIC_params_in_iter{k}" in d.files
            # the field must never be named as if it were the likelihood of
            # params_out at the same iteration -- that pairing is exactly what
            # the P1 sign-off round corrected
            assert f"em_loglik_iter{k}" not in d.files
            assert f"em_loglik_params_out_iter{k}" not in d.files

    def test_params_in_reconstructs_without_extra_storage(self, fixtures_dir, em_trace_scenario_id):
        fx = load_em_trace_fixture(fixtures_dir, em_trace_scenario_id)
        p_in_1 = fx.params_in(1)
        # Regression test for a real P1 bug: params_in(1) used to return the
        # raw init_model pi_0, which does NOT reproduce em_lsim's recorded
        # iter-1 likelihood (measured off by ~2e-4 relative). It must return
        # the effective (1e-4-regularized) pi_0 instead.
        np.testing.assert_array_equal(p_in_1["pi_0"], fx.effective_pi_0_iter1)
        assert not np.array_equal(fx.effective_pi_0_iter1, fx.init_pi_0), (
            "effective_pi_0_iter1 is bit-identical to the raw init_pi_0 -- either "
            "left2right ended up nonzero (regularization skipped) or the regularization "
            "step silently became a no-op; either way this fixture no longer demonstrates "
            "the distinction it exists to test"
        )

        if len(fx.iterations) >= 2:
            p_in_2 = fx.params_in(2)
            np.testing.assert_array_equal(p_in_2["pi_0"], fx.iterations[0].params_out_pi0)
            np.testing.assert_array_equal(
                p_in_2["transition_matrices_IM"], fx.iterations[0].params_out_transition
            )

    def test_effective_iter1_params_are_replayable(self, fixtures_dir, em_trace_scenario_id):
        """The exporter itself re-verifies this numerically at generation time
        (recomputes the E-step likelihood from effective_pi_0_iter1 and asserts
        it matches em_lsim's own recorded em_loglik_params_in_iter1 to 1e-8
        relative tolerance, erroring the whole export if not) -- this asserts
        the fixture carries everything a P4 test needs to redo it: Y (the
        concatenation of every trial, SCHEMA.md 1.1.0), the effective params,
        and the recorded likelihood to compare against."""
        fx = load_em_trace_fixture(fixtures_dir, em_trace_scenario_id)
        assert fx.Y.shape == (fx.D, int(fx.length_observation.sum()))
        assert np.isfinite(fx.iterations[0].loglik_params_in)
        assert fx.effective_pi_0_iter1.shape == fx.init_pi_0.shape

    def test_all_iterations_finite(self, fixtures_dir, em_trace_scenario_id):
        fx = load_em_trace_fixture(fixtures_dir, em_trace_scenario_id)
        assert len(fx.iterations) >= 1
        for it in fx.iterations:
            assert np.isfinite(it.loglik_params_in)
            assert np.isfinite(it.AIC_params_in)
            assert np.isfinite(it.BIC_params_in)
            assert not np.any(np.isnan(it.params_out_pi0))
