"""Unit tests for chmm_lsim.backends.numpy_'s execution primitives (P3a)."""

from __future__ import annotations

import numpy as np

from chmm_lsim.backends.numpy_ import block_sum, broadcast_block, build_coupling_gather, build_W
from chmm_lsim.core.structure import LSIMStructure


def test_block_sum_1d():
    s = LSIMStructure((2, 3), (1, 1), (1, 1))
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(block_sum(x, s.state_offsets), [3.0, 12.0])


def test_block_sum_2d_axis0_and_axis1():
    s = LSIMStructure((2, 3), (1, 1), (1, 1))
    x = np.arange(25.0).reshape(5, 5)
    summed0 = block_sum(x, s.state_offsets, axis=0)
    assert summed0.shape == (2, 5)
    np.testing.assert_array_equal(summed0, np.stack([x[0:2].sum(axis=0), x[2:5].sum(axis=0)]))
    summed1 = block_sum(x, s.state_offsets, axis=1)
    assert summed1.shape == (5, 2)
    np.testing.assert_array_equal(summed1, np.stack([x[:, 0:2].sum(axis=1), x[:, 2:5].sum(axis=1)], axis=1))


def test_broadcast_block():
    s = LSIMStructure((2, 3), (1, 1), (1, 1))
    v = np.array([10.0, 20.0])
    np.testing.assert_array_equal(broadcast_block(v, s.block_of_state), [10.0, 10.0, 20.0, 20.0, 20.0])


def test_broadcast_block_is_inverse_of_block_sum_for_constant_blocks():
    s = LSIMStructure((2, 3), (1, 1), (1, 1))
    v = np.array([1.5, -2.0])
    broadcasted = broadcast_block(v, s.block_of_state)
    np.testing.assert_array_equal(block_sum(broadcasted, s.state_offsets) / np.array([2, 3]), v)


def test_build_coupling_gather():
    s = LSIMStructure((2, 2), (1, 1), (1, 1))
    coupling = np.array([[0.7, 0.4], [0.3, 0.6]])
    gather = build_coupling_gather(coupling, s.block_of_state)
    assert gather.shape == (4, 4)
    # rows 0,1 -> block 0; cols 0,1 -> block 0; so gather[0,0] = coupling[0,0]
    np.testing.assert_array_equal(gather[0:2, 0:2], np.full((2, 2), coupling[0, 0]))
    np.testing.assert_array_equal(gather[0:2, 2:4], np.full((2, 2), coupling[0, 1]))
    np.testing.assert_array_equal(gather[2:4, 0:2], np.full((2, 2), coupling[1, 0]))


def test_build_W_is_transition_times_gather():
    s = LSIMStructure((2, 2), (1, 1), (1, 1))
    transition = np.full((4, 4), 0.25)
    coupling = np.array([[0.7, 0.4], [0.3, 0.6]])
    W = build_W(transition, coupling, s.block_of_state)
    expected = transition * build_coupling_gather(coupling, s.block_of_state)
    np.testing.assert_array_equal(W, expected)
