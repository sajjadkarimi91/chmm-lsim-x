# LSIM fixture schema — v1.5.0

Defines what every fixture file contains, before any exporter code is written. The exporter
(`tools/matlab/export_fixtures.m`) and the loader (`tools/convert_fixtures.py` + the pytest fixture
loader) both conform to this document; if they disagree, this document wins and one of them is wrong.

Versioning: `schema_version` is a plain `MAJOR.MINOR.PATCH` string, stored in every fixture file and in
`manifest.json`. MAJOR bumps on any field rename/removal/shape change (loader must reject a mismatched
MAJOR rather than guess); MINOR on additive fields; PATCH on documentation-only changes. Current: `1.6.0`
(MINOR bump from `1.5.0`: **P12**, two additive changes to the posterior/EM-trace families.
*(a)* `fbc_poise_refinement` — the POISE variant's one-shot-refinement switch, stored as an exact
integer `0`/`1`; see the `fb_constants` block below for the loader's strict reading of it.
*(b)* **per-file provenance**: every fixture file and every `manifest.json` entry now carries its own
`matlab_release`, `kernel_sha256`, `constants_sha256` and `generated_date`. Before this, those lived
only at the manifest's top level, which was correct only while the whole grid came from one export —
once a later, partial export added files, the single top-level stamp silently appeared to describe
files it had never touched. The manifest keeps the top-level fields (they describe the *original*
export) but **per-entry values govern**, and the converter backfills them onto pre-1.6.0 entries from
that original stamp, which is exactly what produced those files. Additive, and — following the same
policy as every earlier bump — the `sed`/`fast` family's `.npz` files were **not** regenerated: they
remain stamped `1.1.0` with their R2026a provenance, while the P12 POISE family carries `1.6.0` and its
own. A full clean regeneration was run twice to confirm the exporter is deterministic (byte-identical
across runs) and that a complete fresh export satisfies this document; see `docs/PYTHON_PORT_PLAN.md`'s
P12 record for the measured cross-release comparison. Earlier MINOR bump from `1.4.0`: a P10a fixture/instrumentation correction round replaced the P10a scheduling
fixture's own fields wholesale -- `batch_assignment` (membership only) dropped for the ordered
`epoch_permutation`/`batch_offsets`; `pi_0_in`/`pi_0_out`/etc. (one snapshot each) replaced by the
complete per-update `updates_*_in`/`updates_*_out` axis plus `raw_init_*`/`epoch_*_final`/
`returned_best_*`; two new scenarios (`uneven`, `fast`) added -- this is a MAJOR-shaped change in
substance, but since only this one, still-bespoke fixture kind is affected (no posterior/EM-trace/P5/P6
field touched) and this fixture kind's own consumers do not exist yet (P10b, not started), it is
recorded as this fixture kind's second MINOR revision rather than bumping the shared schema MAJOR for
every other fixture kind over a change none of them are affected by; see `docs/PYTHON_PORT_PLAN.md`'s
P10a section for the full record. Earlier MINOR bump from `1.3.0`: the P10a scheduling fixture added,
§P10a below -- another wholly new, separate fixture kind, same pattern as the P6/P5 bumps described
next: its own `schema_version`/`matlab_release`, its own exporter (`tools/matlab/
export_p10a_scheduling_fixtures.m`) and converter (`tools/convert_p10a_scheduling_fixtures.py`), no
existing field touched, no other family regenerated. Earlier MINOR bump from `1.2.0`: the P6 fixture
added, §P6 below -- another wholly new, separate fixture kind,
same pattern as the P5 bump described next: its own `schema_version`/`matlab_release`, its own exporter
(`tools/matlab/export_p6_fixtures.m`) and converter (`tools/convert_p6_fixtures.py`), no existing field
touched, no other family regenerated. Earlier MINOR bump from `1.1.0`: the init-reference fixture added,
§P5 below -- a wholly new, separate fixture kind with its own `schema_version`/`matlab_release` fields,
exported by its own script (`tools/matlab/export_init_reference.m`) and converted by its own script
(`tools/convert_init_reference.py`); no existing posterior/EM-trace field renamed, removed, or reshaped,
and neither of those families' own `.npz` files were regenerated for this bump -- the loader only
enforces MAJOR, so their still-`1.1.0`-stamped files remain valid and unaffected. Earlier MINOR bump
from `1.0.0`: EM-trace fixtures generalized to multi-trial, §P4 below -- additive, no existing field
renamed or reshaped for the single-trial case).

## Five fixture kinds

1. **Posterior fixtures** — one MATLAB *scenario* (a model + observations) exported under **both**
   kernel variants (`sed`, `fast`) as two separate files. This is the `24 × 2 = 48` suite that P3a/P3b
   parity-test against. File name: `<scenario_id>__<variant>.npz`.
2. **EM-trace fixtures** — one scenario run through `em_lsim` for a small, fixed number of iterations
   with `extra.iter_hook` attached, to validate the `params_out` / `loglik_params_in` field semantics
   (§3.A) end to end. Not part of the 24×2 count. File name: `em_trace__<scenario_id>.npz`. As of `1.1.0`,
   not necessarily single-trial -- see `num_trials`/`length_observation`/`Y` below.
3. **The init-reference fixture** — a single, bespoke file (`init_reference.npz`, no `manifest.json`
   entry, no `<scenario_id>` family) for P5's statistical gate specifically: `em_lsim.m`'s *no*-`init_model`
   branch (`fine_gmm_fitting`), not the `init_model`-given branch every EM-trace fixture above exercises.
   See its own section below.
4. **The P6 fixture** — another single, bespoke file (`p6_fixture.npz`, same non-manifest-driven pattern
   as the init-reference fixture), for P6's simulation/equivalent-HMM/Viterbi/diagnostics port: one
   hand-specified, heterogeneous small CHMM model run through `im_para_eqhmm.m` +
   `chmm_cartesian_product.m`, `viterbi_chmm.m`, and `compute_joint_state_observation_prob.m` as they
   actually exist. See its own section below.
5. **The P10a scheduling fixtures** — six single, bespoke files (`p10a_scheduling__<scenario_id>.npz`
   for `scenario_id` in `diag`/`full`/`missing`/`unequal`/`uneven`/`fast`, same non-manifest-driven
   pattern as the init-reference/P6 fixtures), for P10a's MATLAB audit/instrumentation deliverable: the
   audited, drift-settled `em_lsim_par.m` (see `docs/PYTHON_PORT_PLAN.md`'s P10a section) run with
   `mini_batch>1` for a few epochs on one small, heterogeneous model per scenario, capturing the
   COMPLETE incremental parameter path (every update's own ordered trial indices and `params_in`/
   `params_out`, not a schedule summary), the exact ordered mini-batch schedule, the legacy batch-
   training diagnostic, and the coherent end-of-epoch re-score (loglik and gamma-based `pi_steady` both)
   via its own
   `extra.minibatch_hook`/`extra.epoch_hook` instrumentation. See its own section below.

Both of the first two kinds share the same packed-parameter representation (`lsim_pack_para`'s output,
§0.1 of `docs/PYTHON_PORT_PLAN.md`), so the same loader code reads the "input parameters" section of
either. Neither the init-reference nor the P6 fixture does -- neither ever calls `lsim_pack_para`
(though the P6 fixture separately exports an equivalent packed representation of its own small model,
built by hand in its exporter, for convenience -- see its own section).

## `manifest.json` (top-level index)

```jsonc
{
  "schema_version": "1.1.0",
  "generated_date": "2026-08-06",          // wall-clock date of export, not a Date.now() call
  "matlab_release": "26.1.0.2528890 (R2026a)",
  "kernel_sha256": "<sha256 of forward_backward_lsim_kernel.m at export time>",
  "constants_sha256": "<sha256 of lsim_fb_constants.m at export time>",
  // ^ two hashes, not one, so a manifest can tell which of the two canonical-behavior files changed
  "scenarios": [
    {
      "scenario_id": "c1_minimal",
      "axes_covered": ["C=1", "M_c=homog2", ...],   // cross-referenced against the coverage checklist
      "C": 1, "S": 2, "D": 1,
      "channel_num_states": [2], "dim_observation": [1], "num_gmm_component": [1],
      "sigma_diag": true,
      "num_trials": 1, "length_observation": [30],
      "seed": 101,
      "files": {
        "sed":  {"path": "c1_minimal__sed.npz",  "fb_constants": {...}, "has_sed_diagnostics": false},
        "fast": {"path": "c1_minimal__fast.npz", "fb_constants": {...}, "has_sed_diagnostics": false}
      },
      "notes": "smallest legal model: C=1, M=2, d=1, K=1, T=30, no NaN"
    },
    ...
  ],
  "em_traces": [
    {"scenario_id": "c3_het_diag", "path": "em_trace__c3_het_diag.npz", "num_iters": 4, "seed": 301}
  ]
}
```

`fb_constants` inside each `files.<variant>` entry is the actual struct that scenario+variant was
generated with — **not** a reference to "the canonical set". Every fixture carries its own, even though
46 of 48 happen to equal the canonical values; `c3_sparse_l2r`'s override (`pi0_reg=0`) applies to
**both** its `sed` and `fast` files, since the override is about the model's `pi_0` staying exactly as
given, not about the SED-specific δ-solver — so it is 46/48, not 47/48. Nothing about a fixture's file
format marks it as non-default — the loader must read the stored value, always, for every fixture, and
must not have a code path that assumes any default.

## Per-fixture `.npz` contents

### Metadata (scalars / short vectors, always present)

Every string field is stored as a numpy fixed-width unicode array (`<U<N>`) whose width `N` is
whatever numpy assigns for that string's actual content length — **not** a schema-declared fixed
width. Two fixtures' `notes` field will not share a dtype if their note text differs in length; that is
normal `numpy.savez` behaviour, not a bug. Only the fact that it decodes to `str` matters.

| key | dtype | shape | meaning |
| --- | --- | --- | --- |
| `schema_version` | unicode `<U*` | scalar | the version **this file** was written at; the loader enforces MAJOR only, so families exported at different MINOR versions legitimately coexist (see Versioning above) |
| `kernel_sha256`, `constants_sha256` | unicode `<U*` | scalar | `1.6.0`+. SHA-256 of `forward_backward_lsim_kernel.m` and `lsim_fb_constants.m` as they were **when this file was exported** — so a fixture always states which kernel produced it, rather than inheriting a manifest-wide claim |
| `generated_date` | unicode `<U*` | scalar | `1.6.0`+. `yyyy-mm-dd` of this file's own export (was a hardcoded literal at the manifest's top level) |
| `scenario_id` | unicode `<U*` | scalar | e.g. `"c3_sparse_l2r"` |
| `variant` | unicode `<U*` | scalar | `"sed"` or `"fast"` |
| `fixture_id` | unicode `<U*` | scalar | `scenario_id + "__" + variant` |
| `seed` | `int64` | scalar | the literal value passed to `rng(seed,'twister')` — recorded via `rng().Seed` read back immediately after that call, not hand-copied |
| `matlab_release` | unicode `<U*` | scalar | `version()` output at export time |
| `C`, `S`, `D` | `int64` | scalar | channel count, total states, total obs dims |
| `channel_num_states` | `int64` | `(C,)` | `M_c` |
| `dim_observation` | `int64` | `(C,)` | `d_c` |
| `num_gmm_component` | `int64` | `(C,)` | `K_c` |
| `sigma_diag` | `int64` | scalar | `1` = diagonal covariance, `0` = full |
| `num_trials` | `int64` | scalar | number of trials in this scenario |
| `length_observation` | `int64` | `(num_trials,)` | `T_r` per trial |
| `fbc_variant` | unicode `<U*` | scalar | `fb_constants.variant` actually used |
| `fbc_beta_floor` | `float64` | scalar | the value `fb_constants.beta_floor` actually held at export time. `lsim_fb_constants(...)` always sets every one of these five fields regardless of `variant` (SED-only fields still get a value under `fast`, and vice versa) — the exporter stores **whatever value was actually there**, never a fabricated `nan` for "not read by this variant". A reader that wants "is this field semantically meaningful for this variant" answers that from `variant` itself, not from whether the stored number happens to be `nan` |
| `fbc_pi0_reg` | `float64` | scalar | actually used — **the field that matters for `c3_sparse_l2r`** |
| `fbc_f_reg` | `float64` | scalar | actually used |
| `fbc_svi_eps` | `float64` | scalar | actually used |
| `fbc_poise_refinement` | `int64` (exact integer; **not** float, **not** an array) | scalar | `1.6.0`+. `fb_constants.poise_refinement`: whether POISE's one-shot observation-conditioned refinement was applied. Meaningful **only** for `fbc_variant == 'poise'`; the unused-field policy records it as `0` for every other variant, and both MATLAB (`lsim_fb_constants`) and Python (`lsim.FbConstants`) default it to `false`, so a clean full export and a variant-filtered one agree. The loader reads it strictly: scalar shape, integer dtype, value in `{0,1}`, **and consistent with the fixture's own variant label** — `<scenario>__poise` must record `0` and `<scenario>__poise_refined` must record `1`. A file whose name and contents disagree is a corrupt export and is rejected, because this one flag decides which of two different algorithms the fixture is ground truth for. Absent (pre-`1.6.0` files) means `0` |
| `has_sed_diagnostics` | `int64` | scalar | `1` iff `diag_*` keys below are present — **never `1` when `variant=="fast"`**, since `fast` has no δ-solver and no `P_vz_tm1_vw_t_O` to reconstruct from |
| `notes` | unicode `<U*` | scalar | free text: what this fixture targets, any override |

### Packed model parameters (`lsim_pack_para` output — the input to the kernel under test)

| key | dtype | shape |
| --- | --- | --- |
| `pi_0` | `float64` | `(S,)` |
| `transitions_matrices` | `float64` | `(S,S)` |
| `coupling_tetha_IM` | `float64` | `(C,C)` |
| `P_all` | `float64` | `(C, max(M), max(K))` |
| `mu_all` | `float64` | `(D, max(M), max(K))` |
| `sigma_all` | `float64` | `(D, max(M), max(K))` if `sigma_diag==1`, else `(D, max(d), max(M), max(K))` |
| `state_numbers_index` | `int64` | `(C+1,)` |
| `dimension_numbers_index` | `int64` | `(C+1,)` |

### Per-trial data and outputs — **always trial-indexed, even when `num_trials==1`**

For `i` in `0 .. num_trials-1` (uniform naming removes a branch from every loader that would otherwise
need "is this the single-trial or multi-trial shape" logic):

| key pattern | dtype | shape | meaning |
| --- | --- | --- | --- |
| `Y_trial{i}` | `float64` | `(D, T_i)` | observations, `NaN` = missing — the actual input |
| `P_O_model_trial{i}` | `float64` | scalar | MATLAB's `P_O_model` (pseudo-loglik, §0.4.5) |
| `alpha_trial{i}` | `float64` | `(S, T_i)` | |
| `beta_trial{i}` | `float64` | `(S, T_i)` | |
| `alpha_T_trial{i}` | `float64` | `(S, T_i)` | i.e. `gamma` |
| `b_c_ot_nc_trial{i}` | `float64` | `(S, T_i)` | i.e. `scale` |
| `P_observ_cond_to_state_trial{i}` | `float64` | `(S, T_i)` | i.e. `B` |
| `P_observ_cond_to_state_comp_trial{i}` | `float64` | `(S, max(K), T_i)` | i.e. `B_comp` |

### SED diagnostics — **only present when `has_sed_diagnostics==1`, only ever for `variant=="sed"`**

A curated subset of scenarios (at minimum `c3_sparse_l2r`, plus one dense scenario as a contrast) carry
these, reconstructed by the exporter exactly as `tools/matlab/check_sed_invariants.m` does — same
exact-zero safe division, so a state with `alpha_next==0` gets `f`/`δ`/`Gz` entries of exactly `0`, not
`NaN` — from the *public* outputs of `forward_backward_lsim`, not from a kernel change:

| key | dtype | shape | meaning |
| --- | --- | --- | --- |
| `diag_t_samples` | `int64` | `(n,)` | which `t` indices (0-based, into trial 0 only) |
| `diag_f` | `float64` | `(n, C, C)` | `f[z,w]` at each sampled `t` |
| `diag_h` | `float64` | `(n, C)` | `h[z]` at each sampled `t` |
| `diag_delta` | `float64` | `(n, C, C)` | `δ[z,w]` at each sampled `t` |
| `diag_Gz` | `float64` | `(n, C, S)` | `Gz[z,j]` at each sampled `t` |

`fast` fixtures never carry these keys — `has_sed_diagnostics` is `0` and none of the `diag_*` keys
exist in the file. A loader that reads `diag_f` from a `fast` fixture is a loader bug, not a missing
feature; the schema makes that an `assert 'diag_f' not in npz` rather than a silent empty array.

## EM-trace fixture (`em_trace__<scenario_id>.npz`)

**Corrected from the first version of this document, which said "same metadata block as above" —
that was never quite true (there is no `variant`/`fixture_id`/`has_sed_diagnostics`, since an EM trace
has no SED/fast axis and no per-t diagnostics) and, more importantly, the exporter did not actually
deliver even the fields it does share until this was checked against a real replay attempt.** What an
EM-trace fixture actually contains:

| key | dtype | shape | meaning |
| --- | --- | --- | --- |
| `schema_version`, `matlab_release`, `scenario_id` | unicode `<U*` | scalar | as in the metadata block above |
| `seed`, `notes` | `int64` / unicode `<U*` | scalar | as above |
| `C`, `D`, `sigma_diag` | `int64` | scalar | as above (no separate `S` field — read `state_numbers_index[-1]`) |
| `num_trials` | `int64` | scalar | `1` for every P1 EM-trace fixture (single-trial only, §3.A); P4 added `c3_trials_unequal`, `num_trials=3` |
| `length_observation` | `int64` | `(num_trials,)` | `T_r` per trial, in the same order `Y` is concatenated in below |
| `fbc_variant`, `fbc_beta_floor`, `fbc_pi0_reg`, `fbc_f_reg`, `fbc_svi_eps` | as above | the `FbConstants` actually passed to `em_lsim` via `extra.fb_constants` — **the exporter sets this explicitly rather than relying on `em_lsim`'s own default**, so this field is provably right rather than an assumption about `em_lsim.m`'s internals |
| `state_numbers_index`, `dimension_numbers_index` | `int64` | `(C+1,)` | as above |
| `channel_num_states`, `dim_observation`, `num_gmm_component` | `int64` | `(C,)` | as above |
| `Y` | `float64` | `(D, T_all)`, `T_all = sum(length_observation)` | the observations the trace was run on, **concatenated across trials** in `em_lsim.m`'s own internal `all_observation` layout (single-trial fixtures: `T_all == T`, unchanged from `1.0.0`) — **without this, no E-step or M-step is replayable at all**, which was the whole point of an EM-trace fixture and was missing from the first version. Split back into per-trial `(D, T_r)` arrays via `length_observation`'s cumulative boundaries — the loader's `EmTraceFixture.Y_trials` does this |
| `init_pi_0`, `init_transitions_matrices`, `init_coupling_tetha_IM`, `init_P_all`, `init_mu_all`, `init_sigma_all` | `float64` | shapes as in "packed model parameters" above | the **raw** `init_model`, exactly as a caller constructs the cell-API input — this is *not* what iteration 1's E-step runs on, see the next row |
| `effective_pi_0_iter1`, `effective_transitions_matrices_iter1`, `effective_coupling_tetha_IM_iter1`, `effective_P_all_iter1`, `effective_mu_all_iter1`, `effective_sigma_all_iter1` | `float64` | same shapes as the `init_*` fields | what iteration 1's E-step **actually** runs on. `em_lsim.m:159–164` applies `pi_0 ← (1e-4+pi_0)/Σ(...)` per channel block, once, before the first E-step, whenever `extra.left2right==0` — every other parameter passes through unchanged. `init_pi_0` and `effective_pi_0_iter1` therefore differ (by ~1e-4 relative, confirmed by a real measurement: reproducing `em_loglik_params_in_iter1` from `init_pi_0` gives `-352.02734153308245`; from `effective_pi_0_iter1` gives `-352.02714309862256`, matching `em_lsim`'s own recorded value exactly). **A fixture that only stored `init_pi_0` could not be replayed correctly even by a perfectly correct Python port** — this is the reason both are stored, not just the raw one |

Then, for `k` in `1 .. num_iters`:

| key pattern | dtype | shape | meaning |
| --- | --- | --- | --- |
| `em_params_out_pi0_iter{k}` | `float64` | `(S,)` | this iteration's M-step output |
| `em_params_out_transition_iter{k}` | `float64` | `(S,S)` | |
| `em_params_out_coupling_iter{k}` | `float64` | `(C,C)` | |
| `em_params_out_Pall_iter{k}` | `float64` | `(C, max(M), max(K))` | |
| `em_params_out_muall_iter{k}` | `float64` | `(D, max(M), max(K))` | |
| `em_params_out_sigmaall_iter{k}` | `float64` | shape depends on `sigma_diag`, as above | |
| `em_loglik_params_in_iter{k}` | `float64` | scalar | **not** the likelihood of `params_out_iter{k}` — see below |
| `em_AIC_params_in_iter{k}` | `float64` | scalar | ditto |
| `em_BIC_params_in_iter{k}` | `float64` | scalar | ditto |

**Semantics, restated because getting this backwards silently breaks P4's gate:** for a given `k`,
`em_loglik_params_in_iter{k}` is the likelihood of the parameters that were *fed into* iteration `k`'s
E-step — i.e. `em_params_out_iter{k-1}`, or the fixture's **effective** (not raw `init_*`) parameters
when `k==1` — **not** of `em_params_out_iter{k}`. Reconstructing `params_in[k]` therefore never
requires a new array beyond what's already listed above: `em_params_out_*_iter{k-1}` for `k>1`, or
`effective_*_iter1` for `k==1`.

The exporter verifies this itself at generation time, not just by construction: it recomputes the
E-step likelihood from `effective_pi_0_iter1` and asserts it matches `em_lsim`'s own recorded
`em_loglik_params_in_iter1` to `1e-8` relative tolerance, so a future change to either side (the
regularization replica here, or `em_lsim.m` itself) fails the export loudly instead of silently
shipping a fixture nothing can actually replay.

## Init-reference fixture (`init_reference.npz`)

Exported by `tools/matlab/export_init_reference.m` (a standalone script, not part of
`export_fixtures.m`'s pipeline -- see that script's own header comment for why it does not reuse
`build_scenario`/`mk_model`/`mk_obs`) and converted by its own dedicated script,
`tools/convert_init_reference.py` -- same always-on round-trip verification as
`tools/convert_fixtures.py`, just not manifest-driven (this is a single file, not a `24×2`/`7`-member
family). Loaded through `tests/fixtures/loader.py`'s `load_init_reference_fixture` /
`validate_init_reference_fixture`, the same shape/NaN/Inf contract as the two families above (planted-
corruption tests in `test_fixture_validation.py`) -- not a raw `np.load`. Loaded directly by name in the
one test that needs it (`tests/test_init.py`'s statistical-gate test), not via `iter_posterior_fixture_ids`/
`iter_em_trace_scenario_ids`, since it isn't part of either manifest-driven family.

**Purpose, stated once because it is easy to reach for the wrong existing fixture instead**: P5's
statistical gate needs a genuine MATLAB run of `em_lsim.m`'s *no*-`init_model` branch (`fine_gmm_fitting`,
`sklearn.mixture.GaussianMixture`'s actual analogue) to compare `initialize_lsim_params` against. Every
`em_trace__*.npz` fixture's `init_*`/`effective_*_iter1` fields come from `build_scenario`'s `mk_model`
ground-truth parameters, passed to `em_lsim` as an explicit `init_model` (`em_lsim.m`'s `nargin==6`
branch) -- a *different* code path that never calls `fine_gmm_fitting` at all. A first version of the P5
gate compared against that by mistake; this fixture exists so the correct comparison has a real MATLAB
number to check against instead of silently reusing the wrong one again.

| key | dtype | shape | meaning |
| --- | --- | --- | --- |
| `schema_version`, `matlab_release` | unicode `<U*` | scalar | as in the posterior/EM-trace metadata blocks above -- `1.2.0` as of this fixture's introduction |
| `C` | `int64` | scalar | number of channels (3) |
| `channel_num_states`, `dim_observation`, `num_gmm_component` | `int64` | `(C,)` | `M`, `d`, `K` |
| `T` | `int64` | scalar | timesteps (single trial) |
| `sigma_diag` | `bool` | scalar | always `True` here |
| `data_seed` | `int64` | scalar | MATLAB `rng` seed for generating `Y` (plain iid Gaussian noise per channel -- not a simulated LSIM sequence, see the exporter's header comment for why that's fine for this purpose) |
| `init_seed` | `int64` | scalar | MATLAB `rng` seed set immediately before the `em_lsim` call, seeding `fine_gmm_fitting`'s own `fitgmdist` randomness deterministically |
| `num_iters` | `int64` | scalar | number of EM iterations run (10) |
| `Y` | `float64` | `(D, T)` | the exact observations `fine_gmm_fitting` and every EM iteration ran on -- feed this into `initialize_lsim_params` on the Python side, not a regenerated array, since MATLAB's and NumPy's RNGs don't agree |
| `loglik_params_in` | `float64` | `(num_iters,)` | `em_lsim.m`'s own `extra.iter_hook`-recorded per-iteration `loglik_params_in` (the E-step likelihood of the parameters iteration `k` was *fed*, same semantics as `em_loglik_params_in_iter{k}` above) -- entry `num_iters-1` (0-indexed) is what the statistical gate compares against |

## P6 fixture (`p6_fixture.npz`)

Exported by `tools/matlab/export_p6_fixtures.m` (standalone, not part of `export_fixtures.m`'s
pipeline) and converted by `tools/convert_p6_fixtures.py` (same always-on round-trip verification as
`convert_init_reference.py`). One hand-specified (not random) heterogeneous model: `C=2`,
`channel_num_states=[2,3]`, `dim_observation=[1,2]`, `num_gmm_component=[2,1]`, diagonal covariance --
every number in the fixture traces back to a literal value in the exporter script. Only the
observations `Y` come from a seeded MATLAB `rng`; per `docs/PYTHON_PORT_PLAN.md`'s P6 plan, random
*samples* don't need cross-language identity, so `Y` is exported as data and read back byte-for-byte,
never regenerated from the seed on the Python side.

`compute_joint_state_observation_prob.m` (unlike every other file called here) uses a different,
capitalized field-name convention (`.GMM_Param`, `.Sigma`) and wants the *transpose* of the per-channel
conditional transition (`(M_z, J)`, not `(J, M_z)`) -- both respected exactly in the exporter, not
"corrected", since the point is to call the function as it actually exists.

| key | dtype | shape | meaning |
| --- | --- | --- | --- |
| `schema_version`, `matlab_release` | unicode `<U*` | scalar | as above -- `1.3.0` as of this fixture's introduction |
| `C` | `int64` | scalar | number of channels (2) |
| `channel_num_states`, `dim_observation`, `num_gmm_component` | `int64` | `(C,)` | `M`, `d`, `K` -- heterogeneous by construction |
| `T` | `int64` | scalar | timesteps (8) |
| `sigma_diag` | `bool` | scalar | always `True` here |
| `Y` | `float64` | `(D, T)` | the exact observations every quantity below was computed on |
| `joint_pi0` | `float64` | `(J,)`, `J=Πchannel_num_states` | `im_para_eqhmm.m`'s equivalent-HMM initial distribution (`pi_0_ehmm{1,1}`) |
| `joint_transition` | `float64` | `(J, J)` | the equivalent-HMM joint transition (`transition_ehmm{1,1}`, via `chmm_cartesian_product.m`) |
| `joint_P_all`, `joint_mu_all`, `joint_sigma_all` | `float64` | `(J,K_joint)`, `(D,J,K_joint)`, `(D,J,K_joint)`, `K_joint=Πnum_gmm_component` | the equivalent-HMM's Cartesian-product GMM emissions (`ehmm_gmm_para{1,1}`), packed |
| `cond_trans_1`, `cond_trans_2` | `float64` | `(J, M_1)`, `(J, M_2)` | per-target conditional transitions (`transition_chmm{z,1}`), one array per channel (fixed count, `C=2` for this model -- not a variable-length list, since this fixture's `C` is fixed at export time) |
| `P_star_model` | `float64` | scalar | `viterbi_chmm.m`'s own output -- a **sum of per-channel log-domain maxima, not a log-probability** (§ `viterbi.py`'s own docstring) |
| `X_star` | `int64` | `(C, T)` | `viterbi_chmm.m`'s best per-channel path, **zero-based** (exporter subtracts MATLAB's 1-based indexing before saving) |
| `log_joint_prob_viterbi_path` | `float64` | scalar | `compute_joint_state_observation_prob.m`'s output on `X_star` |
| `X_alt` | `int64` | `(C, T)` | a second, deterministic, non-Viterbi path (a fixed modular pattern, not random), **zero-based** |
| `log_joint_prob_alt_path` | `float64` | scalar | `compute_joint_state_observation_prob.m`'s output on `X_alt` |
| `pi_0_packed`, `coupling_packed`, `transition_packed`, `P_all_packed`, `mu_all_packed`, `sigma_all_packed` | `float64` | as in the "packed model parameters" section above | the same hand-specified model, hand-packed into `LSIMParams.from_packed`'s own input shapes, so the Python side can build one `LSIMStructure`/`LSIMParams` directly instead of re-deriving them from the cell-level pieces |

## P10a scheduling fixtures (`p10a_scheduling__<scenario_id>.npz`)

**Corrected in a fixture/instrumentation correction round (2026-09-08, schema `1.5.0`)** -- the first
version of this fixture family (schema `1.4.0`) discarded the per-minibatch `params_in`/`params_out` the
`minibatch_hook` already received, conflated the caller-provided ("raw") initial model with the model
the first E-step actually used ("effective", after `em_lsim_par.m`'s own `pi_0` smoothing), lost the
exact `randperm` ORDER inside each minibatch (`batch_assignment` recorded membership only), had no
per-epoch parameter snapshot for `coherent_epoch_loglik` to be independently recomputed against, and
mislabeled the mixed-diagnostic `pi_steady` as if it were the only `pi_steady` that mattered. All fixed
below -- see `docs/PYTHON_PORT_PLAN.md`'s P10a section for the full before/after record.

Exported by `tools/matlab/export_p10a_scheduling_fixtures.m` (standalone, one file per `scenario_id` in
`diag`/`full`/`missing`/`unequal`/`uneven`/`fast`) and converted by `tools/
convert_p10a_scheduling_fixtures.py` (same always-on round-trip verification as `convert_p6_fixtures.py`,
plus an independent schedule sanity check -- see that script's own `_check_schedule`, which re-derives
the ordered schedule from `epoch_permutation`/`batch_offsets` and cross-checks it against
`updates_trial_indices`, not merely trusting the exporter's own assertions). Every scenario is a small,
heterogeneous `C=2` model (`channel_num_states=[2,3]`, `dim_observation=[1,2]`), seeded parameters/
observations (`data_seed`), run through the audited, drift-settled `em_lsim_par.m`
(`docs/PYTHON_PORT_PLAN.md`'s P10a section) for `max_itration=3` epochs at `mini_batch=3`, with a
separate, explicit `schedule_seed` re-seeded immediately before training so the schedule's own
reproducibility does not depend on how many random draws model/data construction happened to consume.
Six scenarios cover the §2.6 gate list's minimum coverage set plus both forward-backward variants:

| `scenario_id` | what it covers |
| --- | --- |
| `diag` | diagonal covariance, **heterogeneous `K`** (`num_gmm_component=[1,2]` -- channel 2 has a genuine 2-component mixture, exercising real mixture-weight updates), 6 trials, even 2/2/2 partition |
| `full` | full covariance (`sigma_diag=False`), `K=[1,1]`, 6 trials, even partition |
| `missing` | diagonal, `K=[1,1]`, scattered `NaN` in `Y`, 6 trials, even partition |
| `unequal` | diagonal, `K=[1,1]`, six different trial lengths, even partition |
| `uneven` | diagonal, `K=[1,1]`, equal lengths, but **7 trials into 3 minibatches** -- a genuinely uneven partition (sizes vary run to run per MATLAB's own `round(linspace(...))` edges, e.g. 2/3/2), specifically to exercise `epoch_permutation`/`batch_offsets`/`updates_trial_indices` on a non-uniform split |
| `fast` | diagonal, `K=[1,1]`, `fb_constants.variant='fast'` (SVI backward message) instead of the other five scenarios' `'sed'` -- `em_lsim_par.m` no longer rejects `'fast'` (see docs/PYTHON_PORT_PLAN.md; `em_lsim_fast.m` is itself only a thin `em_lsim.m` wrapper, proving the shared M-step is already variant-agnostic) |

**Five distinct parameter snapshots are exported, never to be confused with each other** -- see
`P10aSchedulingFixture`'s own docstring in `tests/fixtures/loader.py` for the full explanation:
`raw_init_*` (caller-provided, pre-smoothing), the first update's own `updates_*_in[0]` (effective,
post-smoothing -- confirmably different from `raw_init_*`, not merely in principle), every update's
`updates_*_in`/`updates_*_out`, each epoch's own final `epoch_*_final`, and `em_lsim_par.m`'s actual
returned `returned_best_*` (the best-by-legacy-mixed-metric snapshot).

| key | dtype | shape | meaning |
| --- | --- | --- | --- |
| `schema_version`, `matlab_release` | unicode `<U*` | scalar | as above -- `1.5.0` as of this correction |
| `scenario_id` | unicode `<U*` | scalar | one of `diag`/`full`/`missing`/`unequal`/`uneven`/`fast` -- must match the filename it was loaded from |
| `variant` | unicode `<U*` | scalar | `'sed'` (five scenarios) or `'fast'` (the `fast` scenario) |
| `C` | `int64` | scalar | number of channels (2) |
| `channel_num_states`, `dim_observation`, `num_gmm_component` | `int64` | `(C,)` | `M`, `d`, `K` |
| `sigma_diag` | `bool` | scalar | `False` only for `scenario_id=="full"` -- the loader/validator reject a non-bool value outright (no `bool(x)` coercion of e.g. `2`) |
| `num_trials` | `int64` | scalar | `6` for every scenario except `uneven` (`7`) |
| `length_observation` | `int64` | `(num_trials,)` | `T_r` per trial -- uniform (16) except for `unequal` |
| `Y` | `float64` | `(D, T_all)`, `T_all=sum(length_observation)` | observations, concatenated across trials in `em_lsim_par.m`'s own internal `all_observation` layout -- split back via `length_observation`'s cumulative boundaries (`P10aSchedulingFixture.Y_trials()`). May contain `NaN` (missing, `missing` scenario only); **never** `Inf`/`-Inf` |
| `data_seed`, `schedule_seed` | `int64` | scalar | the two explicit, separately-set `rng(...,'twister')` seeds (model/observations, and the schedule's own `randperm` calls, respectively) |
| `max_itration`, `mini_batch`, `max_batch_size`, `n_updates` | `int64` | scalar | `n_updates = max_itration*mini_batch` exactly (checked); `max_batch_size` upper-bounds every minibatch's own trial count (`<= num_trials`) and sizes the padded `updates_*` arrays below |
| `raw_init_pi_0`, `raw_init_transition`, `raw_init_coupling`, `raw_init_P_all`, `raw_init_mu_all`, `raw_init_sigma_all` | `float64` | as in the "packed model parameters" section above | the caller-provided `init_model`, packed via `lsim_pack_para`, **before any smoothing** -- differs measurably from `updates_pi_0_in[0]` (typically `~1e-5`-`~1e-4` per-entry pi_0 difference, scenario-dependent), confirmed by an exporter-side assertion that independently recomputes `em_lsim_par.m`'s own pi_0-smoothing formula and matches it against the first stored update's `params_in` exactly, not merely asserted to be close |
| `updates_epoch_idx`, `updates_minibatch_idx` | `int64` | `(n_updates,)` | 1-based (epoch, minibatch) each flattened update row belongs to; `update_idx = (epoch-1)*mini_batch + (minibatch-1)` |
| `updates_n_trials`, `updates_T_all`, `updates_n_transitions` | `int64` | `(n_updates,)` | this update's own trial count, total timesteps, and transition count (`n_transitions == T_all - n_trials` exactly, MATLAB's own `L_T_1_batch` identity -- checked) |
| `updates_trial_indices` | `int64` | `(n_updates, max_batch_size)` | the exact, **ORDERED** (not sorted) 1-based trial indices this update processed, exactly as `em_lsim_par.m`'s own `randperm`-derived `batches{mb}` produced them -- `[:updates_n_trials[u]]` is the active region; padding past that is `0` (never a valid 1-based index) |
| `updates_pretrial_loglik` | `float64` | `(n_updates, max_batch_size)` | each active trial's own pre-update log-evidence, in the same order as `updates_trial_indices`; padding is `NaN` |
| `updates_pi_0_in`/`_out`, `updates_transition_in`/`_out`, `updates_coupling_in`/`_out`, `updates_P_all_in`/`_out`, `updates_mu_all_in`/`_out`, `updates_sigma_all_in`/`_out` | `float64` | `(n_updates, ...)`, packed-parameter shape as above | every single update's own pre-/post-update packed model -- the complete incremental path, not a schedule summary |
| `epoch_permutation` | `int64` | `(max_itration, num_trials)` | the full per-epoch `randperm` output (1-based) -- reconstructed by concatenating that epoch's `updates_trial_indices` in minibatch order, which recovers it exactly since `batch_offsets` slices it into contiguous, order-preserving chunks |
| `batch_offsets` | `int64` | `(max_itration, mini_batch+1)` | boundaries into `epoch_permutation[itr]`: minibatch `mb`'s trials are `epoch_permutation[itr, batch_offsets[itr,mb]+1 : batch_offsets[itr,mb+1]]` (1-based); starts at `0`, ends at `num_trials`, strictly increasing (no empty minibatch) |
| `epoch_pi_0_final`, `epoch_transition_final`, `epoch_coupling_final`, `epoch_P_all_final`, `epoch_mu_all_final`, `epoch_sigma_all_final` | `float64` | `(max_itration, ...)` | the single, coherent packed-model snapshot at the END of each epoch -- equals `updates_*_out` at that epoch's last update exactly (checked); this is what `epoch_coherent_loglik`/`epoch_coherent_pi_steady` are evaluated under, and what a consumer needs to independently recompute them |
| `epoch_mixed_loglik` | `float64` | `(max_itration, num_trials)` | the §2.6 **legacy batch-training diagnostic**: trial `tr`'s own log-evidence, evaluated under the PRE-update parameter snapshot of the minibatch it was assigned to that epoch. Summing a row equals `log_likelyhood_trace[itr]` exactly (checked both as a per-epoch identity and, independently, as a sum over the active region of `updates_pretrial_loglik`) |
| `epoch_coherent_loglik` | `float64` | `(max_itration, num_trials)` | every trial's log-evidence evaluated under that epoch's own `epoch_*_final` snapshot instead (a genuine, independently-recomputable coherent re-score) -- differs meaningfully from `epoch_mixed_loglik` early in training, converges as parameters stabilize (confirmed directly during the P10a audit) |
| `epoch_legacy_mixed_pi_steady` | `float64` | `(max_itration, S)` | **legacy, mixed-parameter** `pi_steady` (`sum(alpha_T_epoch,2)/T_all`, mixing whichever minibatch snapshot was current when each trial was processed) -- never pair this with a coherent parameter snapshot |
| `epoch_coherent_pi_steady` | `float64` | `(max_itration, S)` | genuinely coherent, **gamma-based** `pi_steady`: one full forward-BACKWARD pass (not forward-only) over every trial under `epoch_*_final`, accumulating `alpha_T` -- the pairing `epoch_coherent_pi_steady[itr]` with `epoch_*_final[itr]` is internally consistent by construction |
| `returned_best_epoch_idx` | `int64` | scalar | which epoch (1-based) `em_lsim_par.m`'s own best-iterate bookkeeping selected -- independently re-derived by the exporter (replicating the exact `itr<=2: always update; itr>2: update only if strictly greater` rule) and cross-checked against the returned output, not merely trusted |
| `returned_best_pi_0`, `returned_best_transition`, `returned_best_coupling`, `returned_best_P_all`, `returned_best_mu_all`, `returned_best_sigma_all`, `returned_best_pi_steady` | `float64` | packed-parameter shape / `(S,)` | `em_lsim_par.m`'s own actual return value -- equals `epoch_*_final[returned_best_epoch_idx-1]` / `epoch_legacy_mixed_pi_steady[returned_best_epoch_idx-1]` exactly (checked); **not** automatically "the final epoch" -- only if `returned_best_epoch_idx == max_itration` |
| `log_likelyhood_trace`, `AIC_trace`, `BIC_trace` | `float64` | `(max_itration,)` | `em_lsim_par.m`'s own per-epoch mixed-parameter diagnostics, unchanged in meaning from `em_lsim.m`'s own `log_likelyhood`/`AIC`/`BIC` -- must be finite (never `NaN`/`Inf`, unlike `Y`) |

**Still no raw (unfinalized) `ExpectedStats`-equivalent sufficient statistics are exported** -- the
`extra.minibatch_hook` mechanism in `em_lsim_par.m` passes a `batch_diagnostics` struct (`sum3_zeta`,
`gamma_P0`, `n_trials`, `T_all`, `n_transitions`) that the exporter uses internally but does not export
verbatim; it has no emission/mixture-weight sufficient statistics at all (no `gamma_sum`, no per-channel
weighted-observation moments), so it is explicitly NOT a complete raw-stats payload. Defining the
complete MATLAB-to-Python raw-statistics mapping remains P10b's job. What IS exported and complete for
P10a's own purpose is the per-update `params_in`/`params_out` pair at every single update -- sufficient
to replay the whole incremental parameter path exactly, which was this correction round's actual point.

## Validation the loader performs on every fixture, before returning it

**Corrected from the first version of this document: this used to say "must perform" and describe
checks that in fact lived only in `test_fixture_schema.py`, not in the loader itself — a fixture loaded
directly (as `tests/fixtures/loader.py` is explicitly designed to be, with no pytest dependency, so P2+
can import it without pulling in the test runner) got zero validation. `load_posterior_fixture` and
`load_em_trace_fixture` now both take a `validate: bool = True` parameter and run this by default,
raising `FixtureValidationError`:**

- `schema_version` major component matches this document's — raises `SchemaVersionError` before any of
  the checks below run.
- Every array's shape matches the metadata scalars (`S`, `D`, `C`, `T_i`, `max(M)`, `max(K)`) exactly —
  a shape mismatch is a hard error, not a reshape-and-hope.
- No unexpected `NaN`/`Inf`: `Y_trial{i}` (or the EM-trace fixture's `Y`) may contain `NaN` (missing
  data, by design); no other array in a posterior fixture may, except where a scenario's `notes` field
  says so explicitly (there are none in the current 24 — `c3_sparse_l2r`'s whole point is that it must
  **not** produce `NaN` after the P0b fix).
- `has_sed_diagnostics` and the presence/absence of `diag_*` keys agree, and never both hold on a
  `fast`-variant fixture.
- For the EM trace: every `params_out_pi0_iter{k}` is `NaN`-free and every `loglik_params_in_iter{k}` is
  finite.
- `fbc_pi0_reg` etc. are read from the file and used as-is — asserting the *value* differs from the
  canonical default is exactly how the test for `c3_sparse_l2r` knows its override actually took effect,
  rather than silently testing the canonical case twice under two names.

`validate=False` is available for a caller who has deliberately mutated a loaded fixture (e.g. a
negative test constructing a broken one on purpose) and wants to skip re-validation; production code
has no reason to pass it.
