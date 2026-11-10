"""The `EmissionModel` interface and the emission-model registry.

An emission model owns everything that depends on the form of the
observation densities ``p(y | state)``: the per-channel parameter objects
(``params.emissions[c]``), their validation, density evaluation, sampling,
the EM sufficient statistics and M-step, initialization, the free-parameter
count and the array layout used by the `.npz` formats. Transition models
(`chmm_lsim.models.*`) only talk to emissions through this interface, so a
new emission family is one module that subclasses `EmissionModel` and calls
`register_emission_model`.

Models are looked up by name (`get_emission_model`), or from a parameter
object by the class of its per-channel emission objects
(`emission_model_for`), or from E-step statistics by the class of their
per-channel statistics objects (`emission_model_for_stats`).

Densities and sampling receive whole sequences / the full observation
history rather than one sample, so models whose density at time ``t``
depends on earlier observations (autoregressive emissions) fit the same
interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar

import numpy as np
from numpy.typing import NDArray


class EmissionModel(ABC):
    """Interface of an emission family. One instance per family is registered.

    Methods taking ``**options`` receive model-specific keyword options (for
    the Gaussian-mixture model: ``sigma_diag``, ``matlab_compat``, ...); a
    model ignores options it does not use. ``emissions`` is always the list
    of per-channel parameter objects of one model, ``structure`` its
    `LSIMStructure` and ``Y`` a full ``(D, T)`` observation sequence (NaN =
    missing).
    """

    #: registry name, e.g. ``"gmm"``
    name: ClassVar[str]
    #: class of the per-channel parameter objects (``params.emissions[c]``)
    channel_type: ClassVar[type]
    #: class of the per-channel E-step statistics objects
    stats_type: ClassVar[type]
    #: model-wide arrays and per-channel array fields of the `.npz` formats
    serial_global_keys: ClassVar[tuple[str, ...]] = ()
    serial_channel_fields: ClassVar[tuple[str, ...]] = ()

    # -- parameters --------------------------------------------------------------------------------
    @abstractmethod
    def check_shapes(self, structure, emissions: list) -> None:
        """Raise `ValueError` unless `emissions` has one object per channel with
        the channel's own dimensions."""

    @abstractmethod
    def validate(self, emissions: list, *, matlab_compat: bool = True) -> None:
        """Raise `ValidationError` unless every value is a valid parameter."""

    @abstractmethod
    def copy(self, emission):
        """Deep copy of one channel's parameters."""

    @abstractmethod
    def n_free_parameters(self, structure, **options) -> int:
        """Number of free emission parameters (for AIC/BIC)."""

    # -- densities and sampling --------------------------------------------------------------------
    @abstractmethod
    def densities(self, structure, emissions: list, Y: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
        """``(B, B_comp)`` for the whole sequence `Y`: ``B[s, t] = p(y_t | state s)``
        with states stacked channel by channel, shape ``(S, T)``, and its
        per-component breakdown ``B_comp`` of shape ``(S, max_K, T)`` (a
        model without mixture components uses ``B_comp[:, 0, :] = B``)."""

    @abstractmethod
    def sample(self, rng: np.random.Generator, emission, state: int, *, past: NDArray[np.float64]) -> NDArray[np.float64]:
        """One draw of channel observations ``(d_c,)`` in hidden `state`, given the
        observations of all channels before this time step ``past`` ``(D, t)``."""

    # -- EM ----------------------------------------------------------------------------------------
    @abstractmethod
    def new_stats(self, structure, c: int, emission):
        """Empty (zero) sufficient statistics for channel `c`."""

    @abstractmethod
    def accumulate(self, stats, structure, c: int, posterior, Y: NDArray[np.float64], emission) -> None:
        """Add one sequence's contribution to channel `c`'s `stats`, in place.
        `posterior` has ``gamma`` ``(S, T)`` and the ``B``/``B_comp`` returned by
        `densities` for `Y`."""

    @abstractmethod
    def merge_stats(self, chunks: list):
        """Combine raw statistics of one channel from disjoint chunks of sequences."""

    @abstractmethod
    def finalize_stats(self, stats, **options):
        """Derived quantities, computed once after all merging; returns a new object."""

    @abstractmethod
    def m_step(self, structure, stats: list, *, T_all: int, **options) -> list:
        """New per-channel parameters from every channel's finalized statistics."""

    def global_statistics(self, structure, Y_trials, **options):
        """Whole-dataset quantities the mini-batch M-step should use instead of the
        mini-batch's own (computed once per training run; validates every
        observation). Default: none."""
        from chmm_lsim.core.validate import validate_observation_list

        validate_observation_list(structure, Y_trials, **options)
        return None

    def apply_global_statistics(self, stats, global_stats):
        """Substitute `global_statistics` into finalized `ExpectedStats`. Default: no-op."""
        return stats

    # -- initialization ----------------------------------------------------------------------------
    def check_init_options(self, structure, **options) -> dict:
        """Validate and normalize the initialization options (called before any
        observation is touched)."""
        return options

    @abstractmethod
    def initialize(self, structure, Y_trials: list, channel_seeds: list[int], **options) -> list:
        """Initial per-channel parameters from validated observations;
        ``channel_seeds[c]`` seeds channel `c`."""

    # -- serialization -----------------------------------------------------------------------------
    @abstractmethod
    def as_float64(self, emissions: list) -> list:
        """Copies of `emissions` with every array converted to float64."""

    @abstractmethod
    def to_arrays(self, emissions: list) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]]]:
        """``(model_wide_arrays, per_channel_arrays)`` keyed by
        `serial_global_keys` / `serial_channel_fields`."""

    @abstractmethod
    def read_global_arrays(self, path, data: dict, fail: Callable[[Any, str], None]) -> dict:
        """Check the model-wide arrays in `data` (calling ``fail(path, message)``
        on a violation) and return the options they encode."""

    @abstractmethod
    def channel_array_shapes(self, M: int, d: int, K: int, options: dict) -> dict[str, tuple[int, ...]]:
        """Required float64 shape of every per-channel array of one channel."""

    @abstractmethod
    def from_arrays(self, channel_arrays: list[dict[str, np.ndarray]], options: dict) -> list:
        """Per-channel parameter objects from checked arrays (copies)."""


# ------------------------------------------------------------------------------------------- registry
_MODELS: dict[str, EmissionModel] = {}
_BY_CHANNEL_TYPE: dict[type, EmissionModel] = {}
_BY_STATS_TYPE: dict[type, EmissionModel] = {}


def register_emission_model(model: EmissionModel) -> EmissionModel:
    """Register an emission-model instance under ``model.name``. Raises
    `ValueError` if the name, or its parameter/statistics class, is already
    registered. Returns `model`."""
    if not isinstance(model, EmissionModel):
        raise TypeError(f"register_emission_model expects an EmissionModel instance, got {type(model).__name__}")
    name = model.name
    if name in _MODELS:
        raise ValueError(f"emission model {name!r} is already registered (registered emission models are {sorted(_MODELS)})")
    for table, cls, what in ((_BY_CHANNEL_TYPE, model.channel_type, "parameter"), (_BY_STATS_TYPE, model.stats_type, "statistics")):
        if cls in table:
            raise ValueError(f"{what} class {cls.__name__} already belongs to emission model {table[cls].name!r}")
    _MODELS[name] = model
    _BY_CHANNEL_TYPE[model.channel_type] = model
    _BY_STATS_TYPE[model.stats_type] = model
    return model


def unregister_emission_model(name: str) -> None:
    """Remove a registered emission model (for tests and interactive use)."""
    model = _MODELS.pop(name)
    _BY_CHANNEL_TYPE.pop(model.channel_type, None)
    _BY_STATS_TYPE.pop(model.stats_type, None)


def registered_emission_models() -> tuple[str, ...]:
    """Names of every registered emission model, in registration order."""
    return tuple(_MODELS)


def get_emission_model(name: str) -> EmissionModel:
    """The emission model registered as `name`; `ValueError` naming the
    registered models otherwise."""
    try:
        return _MODELS[name]
    except KeyError:
        raise ValueError(f"unknown emission model {name!r}; registered emission models are {sorted(_MODELS)}") from None


def _lookup(table: dict[type, EmissionModel], cls: type, what: str) -> EmissionModel:
    model = table.get(cls)
    if model is not None:
        return model
    for registered, model in table.items():
        if issubclass(cls, registered):
            return model
    raise ValueError(
        f"no registered emission model handles {what} of type {cls.__module__}.{cls.__qualname__}; "
        f"registered emission models are {sorted(_MODELS)}"
    )


def emission_model_for(obj) -> EmissionModel:
    """The emission model of a parameter object (anything with an ``emissions``
    list, e.g. `LSIMParams`/`CHMMParams`) or of a list of per-channel emission
    objects, found by the class of its per-channel objects. Every channel must
    use the same model."""
    emissions = obj.emissions if hasattr(obj, "emissions") else obj
    if len(emissions) == 0:
        raise ValueError("cannot determine the emission model of an empty emissions list")
    model = _lookup(_BY_CHANNEL_TYPE, type(emissions[0]), "per-channel emission parameters")
    for c, e in enumerate(emissions[1:], start=1):
        if not isinstance(e, model.channel_type):
            raise ValueError(f"emissions[{c}] is a {type(e).__name__}, but emissions[0] uses the {model.name!r} emission model")
    return model


def emission_model_for_stats(channel_stats: list) -> EmissionModel:
    """The emission model that produced a list of per-channel E-step statistics."""
    if len(channel_stats) == 0:
        raise ValueError("cannot determine the emission model of an empty statistics list")
    return _lookup(_BY_STATS_TYPE, type(channel_stats[0]), "per-channel emission statistics")


# ------------------------------------------------------------------------------- dispatching helpers
def compute_emissions(structure, params, Y: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
    """``(B (S, T), B_comp (S, max_K, T))``: emission densities of every state
    for the sequence `Y`, computed by the emission model of `params` (for
    Gaussian mixtures: `gmm_pdf_diag` or `gmm_pdf_full`, chosen by
    ``params.sigma_diag``)."""
    return emission_model_for(params.emissions).densities(structure, params.emissions, Y)


def check_emissions_match(structure, emissions: list) -> None:
    """Raises `ValueError` unless `emissions` has one object per channel, each
    with the channel's own dimensions (checked by its emission model)."""
    if len(emissions) != structure.C:
        raise ValueError(f"len(emissions)={len(emissions)} != C={structure.C}")
    emission_model_for(emissions).check_shapes(structure, emissions)


def validate_emissions(emissions: list, *, matlab_compat: bool = True) -> None:
    """Parameter-value checks of `emissions` by their emission model
    (shared by LSIM and CHMM parameters)."""
    emission_model_for(emissions).validate(emissions, matlab_compat=matlab_compat)


def new_emission_stats(structure, emissions: list, model: EmissionModel | None = None) -> list:
    """One empty statistics object per channel."""
    model = model or emission_model_for(emissions)
    return [model.new_stats(structure, c, emissions[c]) for c in range(structure.C)]


def accumulate_emission_stats(stats: list, structure, posterior, Y, emissions: list, model: EmissionModel | None = None) -> None:
    """Add one sequence to every channel's statistics, in place."""
    model = model or emission_model_for(emissions)
    for c in range(structure.C):
        model.accumulate(stats[c], structure, c, posterior, Y, emissions[c])


def m_step_emissions(structure, stats, **options) -> list:
    """Every channel's emission update from finalized `ExpectedStats` (the
    emission half of the M-step, shared by every transition model)."""
    model = emission_model_for_stats(stats.channels)
    return model.m_step(structure, stats.channels, T_all=stats.T_all, **options)


def initialize_emissions(structure, Y_trials, *, label: str, seed: int = 0, model: str = "gmm", **options) -> list:
    """Initial emissions for every channel: checks the model's options, that
    `Y_trials` is non-empty and every trial valid, derives one seed per channel
    from `seed` (`numpy.random.SeedSequence(seed).spawn(C)`), then calls the
    model's `initialize`. `label` names the calling function in errors."""
    from chmm_lsim.core.validate import ValidationError, validate_observations

    emission_model = get_emission_model(model)
    options = emission_model.check_init_options(structure, **options)
    if len(Y_trials) == 0:
        raise ValidationError(f"{label} requires at least one trial in Y_trials, got an empty list")
    for Y in Y_trials:
        validate_observations(structure, Y)
    seed_seq = np.random.SeedSequence(seed)
    channel_seeds = [int(child.generate_state(1)[0]) for child in seed_seq.spawn(structure.C)]
    return emission_model.initialize(structure, Y_trials, channel_seeds, **options)
