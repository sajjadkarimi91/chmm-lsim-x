"""Model-independent mini-batch EM scheduling: the trial-to-mini-batch
schedule (`MinibatchSchedule`, `generate_minibatch_schedule`,
`validate_minibatch_schedule`), the per-update and per-run result records,
and the convergence policies of the mini-batch training driver
(`RobustConvergence`)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from chmm_lsim.models.lsim.params import LSIMParams


# ==========================================================================
# P10c: backend-independent incremental mini-batch EM scheduling/update
# driver. Reuses P10b's `e_step` (and
# therefore its `n_jobs`/`executor` axis, and its `backend="torch"` ->
# `forward_backward_batch(..., mode="stats")` dispatch) unchanged for every
# mini-batch's own E-step -- this module adds only the SCHEDULING (which
# trials go in which mini-batch of which epoch) and the incremental UPDATE
# LOOP (freeze params_in, E-step, merge, finalize once, M-step, commit)
# around it. Deliberately NOT `em_lsim`/`EmResult`: no coherent end-of-epoch
# re-score, no `loglik_trace`/`aic_trace`/`bic_trace`, no convergence check,
# no best-iterate selection -- every one of those requires evaluating every
# trial under one single coherent parameter snapshot once per epoch, which
# is P10d's own job (§2.6), not this mechanical driver's.
# ==========================================================================


class _Unset:
    """A dedicated sentinel type, not `None` -- `em_lsim_minibatch`'s own
    `num_minibatches`/`shuffle`/`random_state` need to tell "the caller
    explicitly passed this" apart from "the caller left it at its default",
    which a plain default value can never do (an explicit `num_minibatches=1`
    is indistinguishable from omitting it if `1` is also the default) --
    see `em_lsim_minibatch`'s own docstring for why that distinction
    matters (schedule-generation-only controls given alongside an explicit
    `schedule=` must raise, not be silently ignored)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<unset>"

_UNSET = _Unset()

def _canonical_trial_index(i) -> int:
    """A genuine, non-bool integer trial index, or raises. `bool` is a
    subclass of `int` in Python (`isinstance(True, int)` is `True`), so it
    is checked and rejected explicitly, before the general `int`/`np.
    integer` check would otherwise silently accept it as `0`/`1`; a `float`
    (even an integer-valued one like `0.0`) is rejected too -- both would
    otherwise pass a value-only range/membership check while being the
    wrong type."""
    if isinstance(i, bool):
        raise ValueError(f"trial index must be a plain int, got bool {i!r}")
    if isinstance(i, (int, np.integer)):
        return int(i)
    raise ValueError(f"trial index must be an int (or np.integer), got {type(i).__name__}: {i!r}")

@dataclass(frozen=True)
class MinibatchSchedule:
    """The exact, 0-based, ordered trial-index assignment for every mini-
    batch of every epoch in one P10c training run -- generated once up
    front (`generate_minibatch_schedule`) or supplied explicitly (e.g. to
    replay a P10a golden fixture's own MATLAB-exported schedule byte-for-
    byte via its `updates_trial_indices`/`updates_epoch_idx`/`updates_
    minibatch_idx` fields), so a run's own batch membership/order is fully
    reproducible and inspectable before any E-step ever executes it.

    `epochs[e][mb]` is a tuple of 0-based trial indices, in the exact order
    that mini-batch's own E-step will process them -- ordered, not merely a
    set: `em_lsim_par.m`'s own `batches{mb}` is `perm(edges(b)+1:edges(b+1))`,
    an ordered slice of that epoch's `randperm` output, not a resorted one,
    and P10a's fixture replay depends on that order being preserved exactly.

    **Deep canonicalization at construction (§P10c correction round).**
    `epochs` (and every nested epoch/mini-batch) is converted to a genuine
    tuple regardless of what was passed in (a list of lists is accepted and
    converted, not merely tolerated by duck-typing) -- a `frozen=True`
    dataclass only freezes its own top-level field *assignment*; it does
    nothing to prevent a caller's own mutable nested `list` from being
    mutated later out from under an already-constructed, supposedly-
    immutable schedule. Every trial index is independently canonicalized to
    a plain `int` (`_canonical_trial_index`): a `bool` is rejected outright
    (`isinstance(True, int)` is `True` in Python, so `0 <= False < num_
    trials` silently reads as a valid index `0` unless bool is excluded
    explicitly), and a `float` such as `0.0` is rejected even though it
    compares equal to the int `0` -- both would otherwise pass a value-only
    range/membership check while being the wrong TYPE. This construction-
    time canonicalization means `validate_minibatch_schedule` never needs
    to re-check index types -- only index VALUES -- for any `MinibatchSchedule`
    instance that already exists."""

    epochs: tuple[tuple[tuple[int, ...], ...], ...]

    def __post_init__(self) -> None:
        canonical = tuple(
            tuple(tuple(_canonical_trial_index(i) for i in mb) for mb in epoch) for epoch in self.epochs
        )
        object.__setattr__(self, "epochs", canonical)

    @property
    def max_iterations(self) -> int:
        return len(self.epochs)

    @property
    def num_minibatches(self) -> int:
        """The per-epoch mini-batch count -- GUARANTEED constant across
        every epoch for any schedule that has passed `validate_minibatch_
        schedule` (§P10c correction round: a varying per-epoch count now
        raises there, rather than being silently summarized by only the
        first epoch's own count). `em_lsim_minibatch` always validates
        before use, so this is safe to rely on for any schedule it actually
        runs; a schedule built and inspected WITHOUT validation first could
        still have a varying count, in which case this reports only the
        first epoch's own -- construct-then-validate (or use `generate_
        minibatch_schedule`, which validates its own result) if that
        matters to a caller."""
        return len(self.epochs[0]) if self.epochs else 0

def validate_minibatch_schedule(schedule: MinibatchSchedule, *, num_trials: int, max_iterations: int | None = None) -> None:
    """Raises `ValueError`, naming the exact problem, for any schedule that
    does not partition `range(num_trials)` exactly once per epoch: wrong
    epoch count (only checked if `max_iterations` is given), a mini-batch
    count that varies epoch to epoch (§P10c correction round: this is what
    makes `MinibatchSchedule.num_minibatches`'s single-number summary
    meaningful -- a schedule that fails this check is still representable
    as a `MinibatchSchedule`, just rejected here rather than silently
    misreported by that property), an out-of-range or negative 0-based
    index, a trial missing or duplicated within an epoch, or an empty
    mini-batch. Trial-index TYPE correctness (no bool/float indices) is
    `MinibatchSchedule.__post_init__`'s own job, not this function's --
    by the time an instance exists, its indices are already genuine `int`.

    **Its own arguments are validated too (§P10c final correction round).**
    `schedule` must genuinely be a `MinibatchSchedule` (not, say, a plain
    list of lists that happens to look like one -- `len(schedule.epochs)`
    below would raise a confusing `AttributeError` otherwise); `num_trials`
    must be a positive, non-bool int; `max_iterations`, when given, must be
    a positive, non-bool int too. The last one matters concretely: with a
    genuine one-epoch schedule, `len(schedule.epochs) != max_iterations`
    alone would silently ACCEPT `max_iterations=True` or `max_iterations=
    1.0` (both compare equal to the int `1` in Python), which is exactly
    the same bool/float-masquerading-as-int problem `_canonical_trial_
    index` already guards against for trial indices -- now guarded here
    for `max_iterations`/`num_trials` too."""
    if not isinstance(schedule, MinibatchSchedule):
        raise ValueError(f"schedule must be a MinibatchSchedule, got {type(schedule).__name__}: {schedule!r}")
    num_trials = _require_positive_int(num_trials, "num_trials")
    if max_iterations is not None:
        max_iterations = _require_positive_int(max_iterations, "max_iterations")
    if max_iterations is not None and len(schedule.epochs) != max_iterations:
        raise ValueError(f"schedule has {len(schedule.epochs)} epochs, expected max_iterations={max_iterations}")
    if not schedule.epochs:
        raise ValueError("schedule has zero epochs")
    minibatch_counts = {len(mbs) for mbs in schedule.epochs}
    if len(minibatch_counts) > 1:
        raise ValueError(
            f"schedule's own mini-batch count varies across epochs ({sorted(minibatch_counts)}) -- every epoch must "
            f"have the same number of mini-batches"
        )
    for e, mbs in enumerate(schedule.epochs):
        if not mbs:
            raise ValueError(f"epoch {e}: schedule has zero mini-batches")
        seen: list[int] = []
        for mb, idx in enumerate(mbs):
            if not idx:
                raise ValueError(f"epoch {e}, minibatch {mb}: empty mini-batch")
            for i in idx:
                if not (0 <= i < num_trials):
                    raise ValueError(f"epoch {e}, minibatch {mb}: trial index {i} out of range [0, {num_trials})")
            seen.extend(idx)
        if len(seen) != num_trials or sorted(seen) != list(range(num_trials)):
            counts: dict[int, int] = {}
            for i in seen:
                counts[i] = counts.get(i, 0) + 1
            duplicates = sorted(i for i, c in counts.items() if c > 1)
            missing = sorted(set(range(num_trials)) - set(seen))
            raise ValueError(
                f"epoch {e}: does not partition every trial in range(0,{num_trials}) exactly once "
                f"(duplicates={duplicates}, missing={missing})"
            )

def _resolve_num_minibatches(num_minibatches: int, num_trials: int) -> int:
    """`em_lsim_par.m`'s own policy exactly: `mini_batch > num_trails`
    warns and clamps to `num_trails` (`round(extra.mini_batch)` there is
    MATLAB's own int-rounding of a possibly-non-integer input; Python's
    caller is expected to already pass a genuine int, so this only clamps,
    it does not round)."""
    num_minibatches = _require_positive_int(num_minibatches, "num_minibatches")
    if num_minibatches > num_trials:
        warnings.warn(
            f"num_minibatches ({num_minibatches}) > number of trials ({num_trials}); clamping to {num_trials} "
            f"(matches em_lsim_par.m's own mini_batch>num_trails warning+clamp policy)",
            stacklevel=3,
        )
        return num_trials
    return int(num_minibatches)

def _resolve_rng(random_state: int | np.random.Generator | None) -> np.random.Generator:
    """Never touches NumPy's global random state (`np.random.seed`/the
    legacy global `RandomState`) -- an explicit `np.random.Generator`,
    either passed in directly or freshly constructed via `np.random.
    default_rng(random_state)` (which itself never reads or writes global
    state), is the only source of randomness this whole module ever uses.

    **Strictly typed (§P10c final correction round).** Accepts only `None`
    (fresh, still-non-global entropy), a genuine non-bool `int`/`np.integer`
    seed, or an already-constructed `np.random.Generator` -- `bool` (a
    subclass of `int` in Python, so `True`/`False` would otherwise silently
    seed with `1`/`0`), `float`, `str`, a legacy `np.random.RandomState`
    (NOT a `Generator`, and not usable as this module's own RNG), and any
    other type all raise a clear `ValueError` naming the actual type given,
    rather than being handed to `np.random.default_rng` and failing (or, for
    some inputs, silently "succeeding") with an unrelated, confusing NumPy-
    internal error."""
    if random_state is None:
        return np.random.default_rng(None)
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
        raise ValueError(
            f"random_state must be None, a non-bool int seed, or an np.random.Generator, got "
            f"{type(random_state).__name__}: {random_state!r}"
        )
    return np.random.default_rng(int(random_state))

def _require_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return int(value)

def generate_minibatch_schedule(
    num_trials: int, max_iterations: int, num_minibatches: int, *, shuffle: bool, rng: np.random.Generator
) -> MinibatchSchedule:
    """Mirrors `em_lsim_par.m`'s own per-epoch partitioning exactly --
    `edges = round(linspace(0, num_trials, num_minibatches+1))`, then
    `perm(edges(b)+1:edges(b+1))` per mini-batch -- matching MATLAB's own
    partition-SIZE behavior for an uneven split (`num_trials` not evenly
    divisible by `num_minibatches`).

    **MATLAB-compatible rounding, not NumPy's default (§P10c correction
    round).** MATLAB's `round` breaks ties away from zero (`round(2.5)==3`);
    `np.round` breaks ties to EVEN (`np.round(2.5)==2`, banker's rounding) --
    a real, measured divergence, not a theoretical one: `num_trials=5,
    num_minibatches=2` gives MATLAB edges `[0,3,5]` but `np.round` its own
    `edges=[0,2,5]`, a different (`3,2`) vs (`2,3`) mini-batch-size split for
    the exact same inputs. Every edge here is non-negative, so half-away-
    from-zero is exactly `floor(x+0.5)`, computed directly rather than via
    `np.round` at all.

    This does NOT reproduce MATLAB's own `randperm` SEQUENCE for a shared
    seed (`np.random.Generator.permutation` and MATLAB `randperm` are
    different algorithms) -- that is what the explicit `schedule=` override
    on `em_lsim_minibatch` exists for; this generator is for genuine
    Python-side training, not fixture replay.

    Validates every argument itself (never trusts a caller-supplied
    pre-check) and validates its OWN result via `validate_minibatch_
    schedule` before returning -- this function must never hand back an
    invalid schedule. `num_minibatches > num_trials` must already have been
    resolved by the caller (`_resolve_num_minibatches`) -- this raises
    rather than silently clamping again."""
    num_trials = _require_positive_int(num_trials, "num_trials")
    max_iterations = _require_positive_int(max_iterations, "max_iterations")
    num_minibatches = _require_positive_int(num_minibatches, "num_minibatches")
    if not isinstance(shuffle, (bool, np.bool_)):
        raise ValueError(f"shuffle must be a bool, got {shuffle!r}")
    if not isinstance(rng, np.random.Generator):
        raise ValueError(f"rng must be a np.random.Generator, got {type(rng).__name__}: {rng!r}")
    if num_minibatches > num_trials:
        raise ValueError(f"num_minibatches={num_minibatches} > num_trials={num_trials} -- caller must resolve/clamp first")

    edges = np.floor(np.linspace(0, num_trials, num_minibatches + 1) + 0.5).astype(int)
    epochs = []
    for _ in range(max_iterations):
        perm = rng.permutation(num_trials) if shuffle else np.arange(num_trials)
        mbs = tuple(tuple(int(i) for i in perm[edges[b] : edges[b + 1]]) for b in range(num_minibatches))
        epochs.append(mbs)
    schedule = MinibatchSchedule(epochs=tuple(epochs))
    validate_minibatch_schedule(schedule, num_trials=num_trials, max_iterations=max_iterations)
    return schedule

def _resolve_schedule(
    Y_trials: list[NDArray[np.float64]],
    *,
    max_iterations: int | None,
    num_minibatches,
    shuffle,
    random_state,
    schedule: MinibatchSchedule | None,
) -> MinibatchSchedule:
    """Resolves/validates one `MinibatchSchedule`, shared by `em_lsim_
    minibatch` and `em_lsim_minibatch_train` -- factored out (§P10c/P10d
    correction round) so this logic exists in exactly one place, never
    reimplemented (and never independently re-broken) in either caller.

    An earlier version of both callers reimplemented this branch
    separately and both reintroduced two already-fixed bugs: accessing
    `schedule.epochs` (via `len(schedule.epochs)`) before confirming
    `schedule` is actually a `MinibatchSchedule` (a non-`MinibatchSchedule`
    `schedule=` would raise a confusing `AttributeError` instead of a
    clear `ValueError`), and comparing `max_iterations` to that length with
    a bare `!=`, which `True`/`1.0` can silently pass (both compare equal
    to the int `1` in Python) exactly like the bool/float-masquerading-as-
    int problem `_canonical_trial_index`/`_require_positive_int` already
    guard against elsewhere. Both are fixed here by routing EVERY explicit-
    schedule call through `validate_minibatch_schedule(schedule, num_trials=
    ..., max_iterations=max_iterations)` -- which already performs both
    checks, strictly, before touching `schedule.epochs` at all -- instead
    of a separate, ad hoc pre-check in the caller."""
    num_trials = len(Y_trials)
    if schedule is not None:
        conflicting = [
            name
            for name, val in (("num_minibatches", num_minibatches), ("shuffle", shuffle), ("random_state", random_state))
            if val is not _UNSET
        ]
        if conflicting:
            raise ValueError(
                f"schedule= was given together with {conflicting} -- these control schedule GENERATION and would "
                f"be silently ignored (not merely inapplicable) when an explicit schedule is supplied instead; "
                f"drop them, or drop schedule= and let this function generate one"
            )
        validate_minibatch_schedule(schedule, num_trials=num_trials, max_iterations=max_iterations)
        return schedule

    if max_iterations is None:
        raise ValueError("max_iterations is required when no explicit schedule is given")
    resolved_num_minibatches = _resolve_num_minibatches(1 if num_minibatches is _UNSET else num_minibatches, num_trials)
    resolved_shuffle = True if shuffle is _UNSET else shuffle
    resolved_random_state = None if random_state is _UNSET else random_state
    rng = _resolve_rng(resolved_random_state)
    # generate_minibatch_schedule already validates its own result before
    # returning (§P10c correction round) -- no need to re-validate here.
    return generate_minibatch_schedule(num_trials, max_iterations, resolved_num_minibatches, shuffle=resolved_shuffle, rng=rng)


@dataclass
class MinibatchUpdate:
    """One optimization mini-batch's own complete trace record -- the
    Python-object counterpart of everything P10a's own fixture `updates_*`
    fields capture. `training_batch_loglik` is explicitly diagnostic (§2.6):
    it is this mini-batch's own trials' log-likelihood measured under
    `params_in` (the PRE-update snapshot) -- never pair it with `params_out`,
    and never confuse it with a coherent, single-model log-likelihood (that
    is P10d's own end-of-epoch re-score, not this)."""

    epoch_idx: int  # 1-based, matching P10a's own convention
    minibatch_idx: int  # 1-based
    trial_indices: tuple[int, ...]  # 0-based, into the Y_trials list given to em_lsim_minibatch, in E-step order
    params_in: LSIMParams
    params_out: LSIMParams
    n_trials: int
    T_all: int
    n_transitions: int
    training_batch_loglik: float

@dataclass
class MinibatchEmResult:
    """P10c's own driver result -- deliberately NOT `EmResult`: no
    `loglik_trace`/`aic_trace`/`bic_trace`, no `pi_steady`, no best-iterate,
    no convergence decision. Every one of those requires evaluating every
    trial under ONE single coherent parameter snapshot once per epoch --
    P10d's own end-of-epoch re-score (§2.6) -- which this mechanical
    scheduling/update driver does not compute. `params` is simply the LAST
    committed snapshot (the final mini-batch's own `params_out`), not a
    "best" one, since nothing here evaluates a coherent metric to select a
    best iterate by."""

    params: LSIMParams  # final committed snapshot, after the schedule's very last mini-batch update
    schedule: MinibatchSchedule
    updates: list[MinibatchUpdate] = field(default_factory=list)

    @property
    def training_batch_loglik_by_epoch(self) -> list[float]:
        """Per-epoch sum of `training_batch_loglik` across that epoch's own
        mini-batches -- ONE value per epoch, the direct analogue of
        `em_lsim_par.m`'s own per-epoch `epoch_P_O_model` sum (§2.6's "mixed
        pre-update mini-batch likelihood" diagnostic). **Named correction,
        §P10c correction round**: a prior version of this class exposed only
        `training_batch_loglik_sum` (the sum across EVERY epoch in the whole
        run) while its own docstring called it the analogue of MATLAB's
        PER-EPOCH `epoch_P_O_model` -- a real mismatch between the name/
        value and the claimed semantics. This property is the one that is
        actually per-epoch. Still a mixed-parameter diagnostic, not a
        coherent single-model log-likelihood (every mini-batch within an
        epoch scored its own trials under a DIFFERENT, successively-updated
        `params_in` snapshot) -- never use this, or `training_batch_loglik_
        total` below, for convergence or best-iterate selection (P10d's own
        job, not this mechanical driver's)."""
        by_epoch: dict[int, float] = {}
        for u in self.updates:
            by_epoch[u.epoch_idx] = by_epoch.get(u.epoch_idx, 0.0) + u.training_batch_loglik
        return [by_epoch[e] for e in sorted(by_epoch)]

    @property
    def training_batch_loglik_total(self) -> float:
        """Sum of `training_batch_loglik_by_epoch` across the WHOLE run --
        explicitly the whole-run total, not a per-epoch value (replaces the
        prior, ambiguously-named `training_batch_loglik_sum`). Same caveats
        as `training_batch_loglik_by_epoch`: a mixed-parameter diagnostic,
        never a coherent single-model log-likelihood, never used for
        convergence or best-iterate selection."""
        return sum(u.training_batch_loglik for u in self.updates)

@dataclass
class MinibatchTrainingResult:
    """P10d's own driver result: `EmResult`-shaped coherent metrics
    (`loglik_trace`/`aic_trace`/`bic_trace`/best-epoch/`pi_steady`), one
    entry per EPOCH -- not per raw mini-batch update, which is what `Minibatch
    EmResult`/`MinibatchUpdate` (P10c) already record in full and are
    referenced here via `minibatch`, not duplicated. `params`/`pi_steady`/
    `best_epoch` describe the coherent BEST-EPOCH snapshot, evaluated the
    same way `EmResult.params`/`pi_steady`/`best_iterate` describe `em_lsim`'s
    own best-ITERATE snapshot -- "epoch" here is the exact analogue of
    `em_lsim`'s own "iteration" (confirmed by `num_minibatches=1`'s own
    non-regression equivalence, P10c)."""

    params: LSIMParams  # coherent best-epoch snapshot (NOT necessarily the last epoch's own params)
    best_epoch: int  # 1-based epoch index the snapshot came from
    pi_steady: NDArray[np.float64] | None = None  # (S,) gamma-based, evaluated once under the best-epoch snapshot
    loglik_trace: list[float] = field(default_factory=list)  # coherent per-epoch loglik, one per COMPLETED epoch
    aic_trace: list[float] = field(default_factory=list)
    bic_trace: list[float] = field(default_factory=list)
    n_epochs: int = 0  # number of epochs actually run (may be < the requested max_iterations if convergence fired)
    minibatch: "MinibatchEmResult | None" = None  # the complete P10c-level trace: every mini-batch's own update
    # P10e: which epochs were coherently evaluated. loglik_trace/aic_trace/bic_trace have exactly one entry per
    # element here, in order -- with a sparse coherent_eval_interval they are NOT one entry per training epoch.
    evaluated_epoch_indices: list[int] = field(default_factory=list)
    convergence: str = "legacy"  # effective policy: "legacy" | "robust" | "off"
    coherent_eval_interval: "int | str" = 1
    stop_reason: str = "max_iterations"  # "max_iterations" | "converged"


@dataclass(frozen=True)
class RobustConvergence:
    """Settings for `em_lsim_minibatch_train(convergence="robust")` (P10e).

    A plateau rule on the magnitude of change, suited to a mini-batch coherent
    trace that oscillates. An evaluated epoch counts as a convergence event
    only if its change from the previous evaluated epoch is SMALL in either
    direction: `|L_e - L_prev| < rel_tol * |L_prev|`. Any larger change -- an
    improvement, or a decrease of any size -- resets the count. Training stops
    once `patience` consecutive evaluated epochs are convergence events and at
    least `min_epochs` epochs have run.

    Consequences: a genuine plateau stops; steady improvement, a single large
    drop, the recovery after it, and a zig-zag with large swings all continue;
    a sustained large decline is NOT treated as convergence either (training
    runs to `max_iterations`, and best-epoch selection still returns the best
    evaluated snapshot). `rel_tol=0` never stops."""

    rel_tol: float = 1e-5
    patience: int = 3
    min_epochs: int = 10

    def __post_init__(self) -> None:
        if not (isinstance(self.rel_tol, (int, float)) and not isinstance(self.rel_tol, bool) and np.isfinite(self.rel_tol) and self.rel_tol >= 0):
            raise ValueError(f"RobustConvergence.rel_tol must be a finite float >= 0, got {self.rel_tol!r}")
        for name in ("patience", "min_epochs"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"RobustConvergence.{name} must be a positive int, got {v!r}")

_CONVERGENCE_POLICIES = ("legacy", "robust", "off")

def _legacy_convergence_stop(loglik_trace: list[float], epoch_idx: int) -> bool:
    """em_lsim.m's rule, unchanged: after epoch 9, stop when at least 2 of the
    last 5 epoch-to-epoch differences are < 0.001 (this counts ANY decrease).
    Indexed by epoch, so it is only defined for every-epoch evaluation."""
    if epoch_idx > 9:
        recent = np.diff(loglik_trace[epoch_idx - 6 : epoch_idx])
        return bool(np.sum(recent < 0.001) >= 2)
    return False

class _RobustConvergenceMonitor:
    def __init__(self, cfg: RobustConvergence):
        self.cfg = cfg
        self.previous: float | None = None
        self.small_changes = 0

    def update(self, epoch_idx: int, loglik: float) -> bool:
        prev = self.previous
        if prev is not None and np.isfinite(loglik) and np.isfinite(prev) and abs(loglik - prev) < self.cfg.rel_tol * abs(prev):
            self.small_changes += 1
        else:
            self.small_changes = 0
        self.previous = loglik
        return epoch_idx >= self.cfg.min_epochs and self.small_changes >= self.cfg.patience

def _resolve_convergence(check_convergence, convergence, coherent_eval_interval):
    if convergence is not None and convergence not in _CONVERGENCE_POLICIES:
        raise ValueError(f"convergence must be one of {_CONVERGENCE_POLICIES} or None, got {convergence!r}")
    if check_convergence is not _UNSET and not isinstance(check_convergence, bool):
        raise ValueError(f"check_convergence must be a bool, got {check_convergence!r}")
    if convergence is None:
        policy = "off" if check_convergence is False else "legacy"
    else:
        policy = convergence
        if check_convergence is not _UNSET and (check_convergence != (policy != "off")):
            raise ValueError(
                f"check_convergence={check_convergence!r} contradicts convergence={policy!r}; pass only convergence="
            )
    if isinstance(coherent_eval_interval, str):
        if coherent_eval_interval != "final":
            raise ValueError(f"coherent_eval_interval must be a positive int or 'final', got {coherent_eval_interval!r}")
    elif isinstance(coherent_eval_interval, bool) or not isinstance(coherent_eval_interval, (int, np.integer)) or coherent_eval_interval < 1:
        raise ValueError(f"coherent_eval_interval must be a positive int or 'final', got {coherent_eval_interval!r}")
    else:
        coherent_eval_interval = int(coherent_eval_interval)
    if coherent_eval_interval != 1 and policy != "off":
        raise ValueError(
            f"coherent_eval_interval={coherent_eval_interval!r} evaluates only some epochs; convergence is only defined for "
            f"every-epoch evaluation, so pass convergence='off' (or check_convergence=False)"
        )
    return policy, coherent_eval_interval
