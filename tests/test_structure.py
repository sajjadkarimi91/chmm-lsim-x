"""Unit tests for chmm_lsim.structure.LSIMStructure (P2)."""

from __future__ import annotations

import numpy as np
import pytest

from chmm_lsim.core.structure import LSIMStructure


def test_derived_arrays_correct():
    s = LSIMStructure(channel_state_num=(2, 3, 2), channel_obs_dim=(1, 3, 2), num_gmm_component=(1, 2, 3))
    assert s.C == 3
    assert s.S == 7
    assert s.D == 6
    np.testing.assert_array_equal(s.state_offsets, [0, 2, 5, 7])
    np.testing.assert_array_equal(s.dim_offsets, [0, 1, 4, 6])
    np.testing.assert_array_equal(s.block_of_state, [0, 0, 1, 1, 1, 2, 2])
    np.testing.assert_array_equal(s.block_of_dim, [0, 1, 1, 1, 2, 2])
    assert s.max_M == 3
    assert s.max_K == 3
    assert s.max_d == 3
    assert s.state_slice(1) == slice(2, 5)
    assert s.dim_slice(2) == slice(4, 6)


def test_single_channel():
    s = LSIMStructure((2,), (1,), (1,))
    assert (s.C, s.S, s.D) == (1, 2, 1)


def test_derived_arrays_are_read_only():
    s = LSIMStructure((2, 3), (1, 1), (1, 1))
    with pytest.raises(ValueError):
        s.state_offsets[0] = 99
    with pytest.raises(ValueError):
        s.block_of_state[0] = 99


def test_frozen_rejects_attribute_rebinding():
    s = LSIMStructure((2, 3), (1, 1), (1, 1))
    with pytest.raises(Exception):
        s.channel_state_num = (5, 5)


def test_equality_and_hash_by_tuples_only():
    a = LSIMStructure((2, 3), (1, 1), (1, 1))
    b = LSIMStructure((2, 3), (1, 1), (1, 1))
    c = LSIMStructure((3, 2), (1, 1), (1, 1))
    assert a == b
    assert hash(a) == hash(b)
    assert a != c


def test_zero_channels_raises():
    with pytest.raises(ValueError, match="C >= 1"):
        LSIMStructure((), (), ())


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="length C"):
        LSIMStructure((2, 3), (1,), (1, 1))


@pytest.mark.parametrize("field", ["channel_state_num", "channel_obs_dim", "num_gmm_component"])
def test_non_positive_entry_raises(field):
    kwargs = {"channel_state_num": (2, 3), "channel_obs_dim": (1, 1), "num_gmm_component": (1, 1)}
    kwargs[field] = (0, 1)
    with pytest.raises(ValueError):
        LSIMStructure(**kwargs)
