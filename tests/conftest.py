"""Session-wide pytest configuration: locates tests/fixtures/ and exposes the
list of available fixture/EM-trace IDs for parametrization, read fresh from
manifest.json so P1's exporter is the only place fixture identity is defined.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# The committed grid, unless LSIM_FIXTURES_DIR points elsewhere. The override exists so a
# freshly regenerated grid can be run against the whole suite before anyone decides whether
# to adopt it (P12's clean-regeneration check) -- the fixture *loader* is the thing under
# test there, so it must read the alternative directory, not a copy of the committed one.
_OVERRIDE = os.environ.get("LSIM_FIXTURES_DIR")
FIXTURES_DIR = Path(_OVERRIDE) if _OVERRIDE else Path(__file__).parent / "fixtures"
CANONICAL_FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR_IS_OVERRIDDEN = _OVERRIDE is not None
sys.path.insert(0, str(CANONICAL_FIXTURES_DIR))

from loader import (  # noqa: E402
    iter_em_trace_scenario_ids,
    iter_posterior_fixture_ids,
    load_manifest,
)


def _manifest_present() -> bool:
    return (FIXTURES_DIR / "manifest.json").exists()


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def manifest() -> dict:
    return load_manifest(FIXTURES_DIR)


def pytest_generate_tests(metafunc):
    if "posterior_fixture_id" in metafunc.fixturenames:
        ids = iter_posterior_fixture_ids(FIXTURES_DIR) if _manifest_present() else []
        metafunc.parametrize("posterior_fixture_id", ids)
    if "em_trace_scenario_id" in metafunc.fixturenames:
        ids = iter_em_trace_scenario_ids(FIXTURES_DIR) if _manifest_present() else []
        metafunc.parametrize("em_trace_scenario_id", ids)
