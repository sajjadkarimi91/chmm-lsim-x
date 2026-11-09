"""P12 correction round: the fixture contract that `fbc_poise_refinement` and the
per-file provenance introduced (SCHEMA.md `1.6.0`).

Two things are pinned here that no other module covers.

**The refinement flag is load-bearing.** It is the only thing distinguishing a
`__poise` fixture from a `__poise_refined` one -- same scenario, same seed, same
parameters, different algorithm and different expected posteriors. A silently
mis-read flag would not fail loudly; it would compare POISE against the refined
ground truth and report a numerical mismatch, or worse, pass. So the loader
reads it strictly (exact integer scalar, value in `{0,1}`, consistent with the
file's own variant label) and these tests plant each corruption in turn.

**Provenance is per file, not per manifest.** The grid is no longer the product
of a single export: the `sed`/`fast` family is R2026a and stays stamped `1.1.0`,
while the P12 POISE family is R2025b at `1.6.0`, exported from the current kernel
-- the one that carries the normalizer guard, whose `kernel_sha256` those entries
record. One top-level stamp cannot describe both families, so every entry carries
its own and these tests refuse to let that regress.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from loader import (  # noqa: E402
    FixtureValidationError,
    load_em_trace_fixture,
    load_posterior_fixture,
)

PROVENANCE_FIELDS = ("matlab_release", "kernel_sha256", "constants_sha256", "generated_date")
POISE_LABELS = ("poise", "poise_refined")


def _entries(manifest):
    """Every manifest entry that names a fixture file, with a readable id."""
    out = []
    for scenario in manifest["scenarios"]:
        for variant, entry in scenario["files"].items():
            out.append((f"{scenario['scenario_id']}__{variant}", variant, entry))
    for trace in manifest["em_traces"]:
        out.append((Path(trace["path"]).stem, None, trace))
    return out


# --------------------------------------------------------------------------
# Provenance: every entry states which kernel produced it.
# --------------------------------------------------------------------------


def test_every_manifest_entry_carries_its_own_provenance(manifest):
    missing = [fid for fid, _, e in _entries(manifest)
               if any(f not in e or not str(e[f]).strip() for f in PROVENANCE_FIELDS)]
    assert not missing, f"entries without complete per-file provenance: {missing[:5]}"


def _committed_grid_only():
    from conftest import FIXTURES_DIR_IS_OVERRIDDEN

    if FIXTURES_DIR_IS_OVERRIDDEN:
        pytest.skip("states a property of the COMMITTED grid, not of an arbitrary regeneration")


def test_provenance_splits_exactly_along_the_two_exports(manifest):
    """A positive statement of what the grid is, so a future partial export that
    forgets to stamp itself shows up here as a third group rather than silently
    inheriting someone else's hash."""
    _committed_grid_only()
    groups: dict[tuple[str, str], list[str]] = {}
    for fid, _, entry in _entries(manifest):
        groups.setdefault((entry["matlab_release"], entry["kernel_sha256"]), []).append(fid)
    assert len(groups) == 2, f"expected exactly two exports, got {len(groups)}: {list(groups)}"

    by_size = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    poise_group = next(ids for _, ids in by_size if all(
        i.endswith(POISE_LABELS) or "__poise" in i for i in ids))
    other_group = next(ids for key, ids in groups.items() if ids is not poise_group)
    assert all("__poise" in i for i in poise_group), "the P12 export must contain only POISE fixtures"
    assert not any("__poise" in i for i in other_group), "the original export must contain no POISE fixtures"


def test_manifest_top_level_stamp_is_not_taken_as_describing_every_entry(manifest):
    """The bug this fixes: the top-level hash described the *first* export only, yet
    later entries were added underneath it. It is still recorded (it describes that
    original export) but it must no longer be the only answer available."""
    _committed_grid_only()
    top = manifest.get("kernel_sha256")
    assert top, "the original export's stamp is still recorded"
    per_entry = {e["kernel_sha256"] for _, _, e in _entries(manifest)}
    assert len(per_entry) > 1 and top in per_entry


def test_file_level_provenance_matches_the_manifest_entry(fixtures_dir, manifest):
    """The manifest and the `.npz` must not drift apart -- the same failure mode
    `test_each_fixtures_seed_matches_its_manifest_entry` exists for."""
    checked = 0
    for fid, _, entry in _entries(manifest):
        d = np.load(Path(fixtures_dir) / f"{fid}.npz", allow_pickle=False)
        if "kernel_sha256" not in d:
            continue  # pre-1.6.0 family: provenance lives in the manifest entry only
        checked += 1
        for field in PROVENANCE_FIELDS:
            assert str(d[field]) == str(entry[field]), f"{fid}: {field} differs from the manifest"
    assert checked >= 58, f"expected the whole 1.6.0 family to be checked, got {checked}"


def test_poise_family_is_schema_1_6_0_and_the_older_family_is_untouched(fixtures_dir, manifest):
    _committed_grid_only()
    for fid, variant, _ in _entries(manifest):
        d = np.load(Path(fixtures_dir) / f"{fid}.npz", allow_pickle=False)
        version = str(d["schema_version"])
        assert version.split(".")[0] == "1", f"{fid}: unexpected MAJOR"
        if "__poise" in fid:
            assert version == "1.6.0", f"{fid}: the P12 family is exported at 1.6.0, got {version}"
        else:
            assert version == "1.1.0", (
                f"{fid}: the pre-P12 family must NOT have been regenerated for an additive "
                f"bump (SCHEMA.md's policy), got {version}"
            )


# --------------------------------------------------------------------------
# The refinement flag: strict reading, and planted corruptions.
# --------------------------------------------------------------------------


def test_flag_matches_the_label_for_every_poise_fixture(fixtures_dir, manifest):
    seen = 0
    for fid, variant, _ in _entries(manifest):
        if variant not in POISE_LABELS:
            continue
        seen += 1
        fx = load_posterior_fixture(fixtures_dir, fid)
        assert fx.fb_constants.variant == "poise"
        assert fx.fb_constants.poise_refinement is (variant == "poise_refined")
    assert seen == 50, f"expected 50 POISE posterior fixtures, saw {seen}"


def test_em_trace_flag_comes_from_the_file_name_not_the_variant(fixtures_dir):
    """An EM trace records `fbc_variant='poise'` in BOTH modes, so its mode lives
    only in its name -- the loader must not read the mode off the variant."""
    raw = load_em_trace_fixture(fixtures_dir, "c3_het_diag__poise")
    refined = load_em_trace_fixture(fixtures_dir, "c3_het_diag__poise_refined")
    assert raw.fb_constants.variant == refined.fb_constants.variant == "poise"
    assert raw.fb_constants.poise_refinement is False
    assert refined.fb_constants.poise_refinement is True


def test_unused_field_policy_non_poise_fixtures_report_false(fixtures_dir, manifest):
    for fid, variant, _ in _entries(manifest):
        if variant in POISE_LABELS or variant is None:
            continue
        fx = load_posterior_fixture(fixtures_dir, fid)
        assert fx.fb_constants.poise_refinement is False, (
            f"{fid}: the flag is meaningless for {variant} and must read as False"
        )


@pytest.fixture
def planted(tmp_path, fixtures_dir):
    """Copies one fixture into a temp dir so a single field can be corrupted."""

    def _plant(fixture_id: str, **overrides):
        src = Path(fixtures_dir) / f"{fixture_id}.npz"
        data = dict(np.load(src, allow_pickle=False))
        for key, value in overrides.items():
            if value is None:
                data.pop(key, None)
            else:
                data[key] = value
        dst = tmp_path / f"{fixture_id}.npz"
        np.savez(dst, **data)
        shutil.copy(Path(fixtures_dir) / "manifest.json", tmp_path / "manifest.json")
        return tmp_path

    return _plant


def test_planted_corruption_flag_stored_as_float(planted):
    d = planted("c3_het_diag__poise", fbc_poise_refinement=np.float64(0.0))
    with pytest.raises(FixtureValidationError, match="exact integer"):
        load_posterior_fixture(d, "c3_het_diag__poise")


def test_planted_corruption_flag_stored_as_an_array(planted):
    d = planted("c3_het_diag__poise", fbc_poise_refinement=np.array([0], dtype=np.int64))
    with pytest.raises(FixtureValidationError, match="must be a scalar"):
        load_posterior_fixture(d, "c3_het_diag__poise")


def test_planted_corruption_flag_out_of_range(planted):
    d = planted("c3_het_diag__poise", fbc_poise_refinement=np.int64(2))
    with pytest.raises(FixtureValidationError, match="must be 0 or 1"):
        load_posterior_fixture(d, "c3_het_diag__poise")


def test_planted_corruption_flag_disagrees_with_the_label(planted):
    """The corruption that would otherwise be invisible: a `__poise` file claiming
    the refinement ran. Silently trusting either side would validate POISE against
    the wrong ground truth."""
    d = planted("c3_het_diag__poise", fbc_poise_refinement=np.int64(1))
    with pytest.raises(FixtureValidationError, match="implies poise_refinement=False"):
        load_posterior_fixture(d, "c3_het_diag__poise")


def test_planted_corruption_refined_label_claiming_raw(planted):
    d = planted("c3_het_diag__poise_refined", fbc_poise_refinement=np.int64(0))
    with pytest.raises(FixtureValidationError, match="implies poise_refinement=True"):
        load_posterior_fixture(d, "c3_het_diag__poise_refined")


def test_planted_corruption_flag_missing_from_a_poise_fixture(planted):
    """Absent means False for the pre-1.6.0 families, but a POISE fixture that does
    not state its own mode is corrupt, not a default."""
    d = planted("c3_het_diag__poise_refined", fbc_poise_refinement=None)
    with pytest.raises(FixtureValidationError, match="must record fbc_poise_refinement"):
        load_posterior_fixture(d, "c3_het_diag__poise_refined")


def test_planted_corruption_flag_set_on_a_non_poise_fixture(planted):
    d = planted("c3_het_diag__sed", fbc_poise_refinement=np.int64(1))
    with pytest.raises(FixtureValidationError, match="has no meaning"):
        load_posterior_fixture(d, "c3_het_diag__sed")


def test_planted_fixtures_load_cleanly_when_nothing_is_corrupted(planted):
    """Guards the planting machinery itself: if a copied fixture failed to load for
    unrelated reasons, every test above would pass for the wrong reason."""
    d = planted("c3_het_diag__poise")
    fx = load_posterior_fixture(d, "c3_het_diag__poise")
    assert fx.fb_constants.poise_refinement is False
