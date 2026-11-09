"""Unit tests for chmm_lsim.compat.FbConstants (P3a)."""

from __future__ import annotations

import pytest

from chmm_lsim.compat import FbConstants


def test_python_default_is_raw_poise():
    fbc = FbConstants()
    assert fbc.variant == "poise"
    assert fbc.poise_refinement is False
    assert fbc.beta_floor == 1e-9
    assert fbc.pi0_reg == 1e-10
    assert fbc.f_reg == 1e-6
    assert fbc.svi_eps == 1e-8


def test_fast_variant():
    fbc = FbConstants(variant="fast")
    assert fbc.variant == "fast"
    # every numerical constant remains shared with the canonical profiles (§3bis.4)
    assert fbc.beta_floor == 1e-9
    assert fbc.pi0_reg == 1e-10


def test_invalid_variant_raises():
    with pytest.raises(ValueError, match="variant"):
        FbConstants(variant="legacy_sed")


def test_fast_remains_in_the_top_level_public_api():
    import chmm_lsim

    assert "forward_backward_fast" in chmm_lsim.__all__
    assert hasattr(chmm_lsim, "forward_backward_fast")


def test_frozen():
    fbc = FbConstants()
    with pytest.raises(Exception):
        fbc.beta_floor = 1.0
