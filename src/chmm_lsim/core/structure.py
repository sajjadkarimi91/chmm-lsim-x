"""LSIMStructure: the flat packed layout shared by every LSIM array. Channel counts are the only thing that defines the
layout; everything else here is derived from them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class LSIMStructure:
    """Per-channel sizes plus the derived block layout over S states / D dims.

    The three defining fields are plain int tuples -- not arrays -- so the
    dataclass stays hashable and comparable by value; a NumPy array field
    would break both (`__hash__` on an unhashable ndarray, `__eq__` returning
    an array instead of a bool). Every derived array is computed once here and
    marked read-only, since this object is a cache key elsewhere (torch.compile
    shape buckets, W-matrix caches) and must never be mutated after construction.
    """

    channel_state_num: tuple[int, ...]
    channel_obs_dim: tuple[int, ...]
    num_gmm_component: tuple[int, ...]

    C: int = field(init=False, compare=False, repr=False)
    S: int = field(init=False, compare=False, repr=False)
    D: int = field(init=False, compare=False, repr=False)
    state_offsets: NDArray[np.int64] = field(init=False, compare=False, repr=False)
    dim_offsets: NDArray[np.int64] = field(init=False, compare=False, repr=False)
    block_of_state: NDArray[np.int64] = field(init=False, compare=False, repr=False)
    block_of_dim: NDArray[np.int64] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        channel_state_num = tuple(int(m) for m in self.channel_state_num)
        channel_obs_dim = tuple(int(d) for d in self.channel_obs_dim)
        num_gmm_component = tuple(int(k) for k in self.num_gmm_component)
        object.__setattr__(self, "channel_state_num", channel_state_num)
        object.__setattr__(self, "channel_obs_dim", channel_obs_dim)
        object.__setattr__(self, "num_gmm_component", num_gmm_component)

        C = len(channel_state_num)
        if C == 0:
            raise ValueError("LSIMStructure requires at least one channel (C >= 1)")
        if len(channel_obs_dim) != C or len(num_gmm_component) != C:
            raise ValueError(
                f"channel_state_num, channel_obs_dim, num_gmm_component must all have "
                f"length C={C}, got {len(channel_obs_dim)}, {len(num_gmm_component)}"
            )
        if any(m <= 0 for m in channel_state_num):
            raise ValueError(f"channel_state_num must be strictly positive, got {channel_state_num}")
        if any(d <= 0 for d in channel_obs_dim):
            raise ValueError(f"channel_obs_dim must be strictly positive, got {channel_obs_dim}")
        if any(k <= 0 for k in num_gmm_component):
            raise ValueError(f"num_gmm_component must be strictly positive, got {num_gmm_component}")

        state_offsets = np.concatenate([[0], np.cumsum(channel_state_num)]).astype(np.int64)
        dim_offsets = np.concatenate([[0], np.cumsum(channel_obs_dim)]).astype(np.int64)
        state_offsets.flags.writeable = False
        dim_offsets.flags.writeable = False

        block_of_state = np.repeat(np.arange(C, dtype=np.int64), channel_state_num)
        block_of_dim = np.repeat(np.arange(C, dtype=np.int64), channel_obs_dim)
        block_of_state.flags.writeable = False
        block_of_dim.flags.writeable = False

        object.__setattr__(self, "C", C)
        object.__setattr__(self, "S", int(state_offsets[-1]))
        object.__setattr__(self, "D", int(dim_offsets[-1]))
        object.__setattr__(self, "state_offsets", state_offsets)
        object.__setattr__(self, "dim_offsets", dim_offsets)
        object.__setattr__(self, "block_of_state", block_of_state)
        object.__setattr__(self, "block_of_dim", block_of_dim)

    def state_slice(self, c: int) -> slice:
        return slice(int(self.state_offsets[c]), int(self.state_offsets[c + 1]))

    def dim_slice(self, c: int) -> slice:
        return slice(int(self.dim_offsets[c]), int(self.dim_offsets[c + 1]))

    @property
    def max_M(self) -> int:
        return max(self.channel_state_num)

    @property
    def max_K(self) -> int:
        return max(self.num_gmm_component)

    @property
    def max_d(self) -> int:
        return max(self.channel_obs_dim)
