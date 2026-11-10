"""chmm_lsim.discriminative: the MMI trainer (published recipe), checked against the MATLAB trainer.

MATLAB reference fixtures: tests/fixtures/discriminative/{disc_units,disc_runs}.mat, exported from
the MATLAB discriminative trainer. Only their MMI entries are used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

scipy_io = pytest.importorskip("scipy.io")

from chmm_lsim.core.constants import FbConstants  # noqa: E402
from chmm_lsim.discriminative import (  # noqa: E402
    NumpyEngine, ObjectiveConfig, OptionError, at_bounds, build_domain, ebw_proposal,
    make_options, objective, packed_to_params, predict, project, published_update, train_discriminative, weak_sense,
    em_lsim_discriminative, lsim_discriminative_predict,
)
from chmm_lsim.learning.discriminative.counts import counts_from_packed_struct  # noqa: E402
from chmm_lsim.learning.discriminative.domain import Domain  # noqa: E402
from chmm_lsim.learning.discriminative.options import default_max_iteration  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "discriminative"
FBC = FbConstants(variant="sed")

try:
    import torch

    HAS_TORCH = True
    N_CUDA = torch.cuda.device_count() if torch.cuda.is_available() else 0
except ImportError:  # pragma: no cover
    HAS_TORCH = False
    N_CUDA = 0


def _load(name):
    path = FIX / name
    if not path.exists():
        pytest.fail(f"MATLAB reference {path} is missing; it must be committed "
                    "(regenerate with tools/matlab/export_disc_fixtures.m in the development repository)")
    return scipy_io.loadmat(str(path), simplify_cells=True)


@pytest.fixture(scope="module")
def units():
    return _load("disc_units.mat")


@pytest.fixture(scope="module")
def runs():
    return _load("disc_runs.mat")


def _pdiff(a, b):
    out = [np.max(np.abs(a.pi_0 - b.pi_0)), np.max(np.abs(a.transition - b.transition)),
           np.max(np.abs(a.coupling - b.coupling))]
    for x, y in zip(a.emissions, b.emissions):
        out += [np.max(np.abs(x.weights - y.weights)), np.max(np.abs(x.mu - y.mu)), np.max(np.abs(x.sigma - y.sigma))]
    return max(out)


def _counts_problem(units):
    c = units["counts"]
    p = packed_to_params(c["model"])
    Y = [np.asarray(y, float).reshape(p.structure.D, -1) for y in c["Y"]]
    d = c["dom"]
    dom = Domain(mu_lo=np.atleast_1d(d["mu_lo"]), mu_hi=np.atleast_1d(d["mu_hi"]), var_lo=np.atleast_1d(d["var_lo"]),
                 var_hi=np.atleast_1d(d["var_hi"]), prob_floor=float(d["prob_floor"]),
                 support_pi=np.asarray(d["support_pi"], bool), support_A=np.asarray(d["support_A"], bool),
                 support_theta=np.asarray(d["support_theta"], bool),
                 support_w=[g.weights > 0 for g in p.emissions])
    num = counts_from_packed_struct(c["num"], p.structure)
    den = counts_from_packed_struct(c["den"], p.structure)
    return c, p, Y, dom, num, den


def _toy(runs, name):
    t = runs["toys"][name]
    D = int(np.sum(t["dim"]))
    Ytr = [np.asarray(y, float).reshape(D, -1) for y in t["Ytr"]]
    Yva = [np.asarray(y, float).reshape(D, -1) for y in t["Yva"]]
    ub = t["update_blocks"]
    ub = [ub] if isinstance(ub, str) else list(ub)
    return (Ytr, np.asarray(t["label_tr"]).ravel() - 1), (Yva, np.asarray(t["label_va"]).ravel() - 1), \
        [packed_to_params(g) for g in t["gen"]], ub


# ------------------------------------------------------------------------------ options
def test_options_defaults_and_errors():
    o = make_options()
    assert o.objective == "mmi" and o.mmi_recipe == "published" and default_max_iteration(o) == 8
    assert o.score_scale == 1 and o.ebw_E == 2 and o.ismooth_tau == 100 and o.ismooth_backoff == "ml"
    assert o.fb_constants.variant == "poise" and o.fb_constants.poise_refinement is False   # raw POISE default
    assert make_options(objective="mmi") == o
    for v in ("sed", "fast"):
        make_options(fb_constants=FbConstants(variant=v))
    make_options(score_scale=0.1, ismooth_backoff="previous", ismooth_tau_discrete=1.0)
    with pytest.raises(OptionError, match="objective must be 'mmi'"):
        make_options(objective="ml")
    for bad in [dict(foo=1), dict(damping_E=2), dict(max_backtracks=3), dict(tol_rel=0), dict(kappa=2),
                dict(objective="MMI"), dict(mmi_recipe="legacy"), dict(score_scale=0),
                dict(update_blocks=["bogus"]), dict(ismooth_backoff="x"), dict(ebw_E=0)]:
        with pytest.raises(OptionError):
            make_options(**bad)


# ------------------------------------------------------------------------------ single steps vs MATLAB
@pytest.mark.parametrize("backend,n_jobs", [("numpy", 1), ("numba", 1), ("numba", 3)])
def test_counts_and_scores_match_matlab(units, backend, n_jobs):
    pytest.importorskip("numba") if backend == "numba" else None
    c, p, Y, _dom, num_m, den_m = _counts_problem(units)
    eng = NumpyEngine(p.structure, FBC, backend=backend, n_jobs=n_jobs, chunk_size=5)
    try:
        eng.register("t", Y)
        num, den = eng.counts(p, "t", np.arange(len(Y)), np.column_stack([c["w_num"], c["w_den"]]))
        L = eng.scores([p], "t")[:, 0]
    finally:
        eng.close()
    for mine, ref in ((num, num_m), (den, den_m)):
        for a, b in zip(mine.arrays(), ref.arrays()):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(L, c["scores"], rtol=1e-12)


@pytest.mark.parametrize("engine", ["numpy", "torch"])
def test_poise_counts_match_matlab(units, engine):
    if "num_poise" not in units["counts"]:
        pytest.skip("fixture without POISE counts")
    if engine == "torch" and not HAS_TORCH:
        pytest.skip("torch not installed")
    c, p, Y, _dom, *_ = _counts_problem(units)
    fbc = FbConstants(variant="poise", poise_refinement=False)
    if engine == "numpy":
        eng = NumpyEngine(p.structure, fbc, backend="numpy")
    else:
        from chmm_lsim.discriminative import TorchEngine
        eng = TorchEngine(p.structure, fbc, devices=("cpu",))
    eng.register("t", Y)
    num, den = eng.counts(p, "t", np.arange(len(Y)), np.column_stack([c["w_num"], c["w_den"]]))
    eng.close()
    for mine, key in ((num, "num_poise"), (den, "den_poise")):
        ref = counts_from_packed_struct(c[key], p.structure)
        for a, b in zip(mine.arrays(), ref.arrays()):
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-11)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_gemm_poise_backward_matches_backend():
    import torch
    from chmm_lsim.backends import torch_batched as tb
    from chmm_lsim.learning.discriminative.kernels import backward_poise_gemm
    rng = np.random.default_rng(1)
    for C, M in ((3, 4), (2, 7)):
        from chmm_lsim.emissions.gmm import ChannelGMM
        from chmm_lsim.models.lsim.params import LSIMParams
        from chmm_lsim.core.structure import LSIMStructure
        st = LSIMStructure((M,) * C, (1,) * C, (1,) * C)
        A = np.zeros((st.S, st.S))
        for c in range(C):
            for z in range(C):
                A[st.state_slice(c), st.state_slice(z)] = rng.dirichlet(np.ones(M), size=M)
        th = rng.dirichlet(np.ones(C), size=C).T
        em = [ChannelGMM(np.ones((M, 1)), rng.normal(size=(1, M, 1)), np.ones((1, M, 1)), True) for _ in range(C)]
        p = LSIMParams(st, np.ones(st.S) / M, A, th, em)
        ker = tb.TorchKernel.build(st, p, device=torch.device("cpu"), dtype=torch.float64, pi0_reg=1e-10)
        Y = torch.as_tensor(rng.normal(size=(6, C, 40)))
        lt = torch.full((6,), 40, dtype=torch.int64)
        B, _bc, _ls = tb.compute_emissions_torch(ker, Y)
        alpha, scale, _ll = tb._forward_pass_batch(ker, B, list(range(6)), lt, False)
        b_ref, g_ref, _ = tb._backward_poise_batch(ker, alpha, scale, FbConstants(variant="poise"), list(range(6)), lt, False)
        b, g = backward_poise_gemm(torch, tb, ker, alpha, scale, lt)
        np.testing.assert_allclose(g.numpy(), g_ref.numpy(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(b.numpy(), b_ref.numpy(), rtol=1e-12, atol=1e-14)


def test_fused_counts_equal_separate(units):
    from chmm_lsim.discriminative import ObjectiveConfig as OC
    from chmm_lsim.learning.discriminative.objective import WeightSpec
    c, p, Y, *_ = _counts_problem(units)
    p2 = p.copy()
    for g in p2.emissions:
        g.mu = g.mu + 0.3
    label = (np.asarray(c["label"]) - 1).astype(np.int64)
    T = np.array([y.shape[1] for y in Y], float)
    for cfg in (OC("mmi", np.log([0.5, 0.5]), score_scale=0.1, N_stat=len(Y)),
                OC("mmi", np.log([0.3, 0.7]), score_scale=1.0, N_stat=len(Y))):
        engs = [NumpyEngine(p.structure, FBC, backend="numpy", chunk_size=7)]
        if HAS_TORCH:
            from chmm_lsim.discriminative import TorchEngine
            engs.append(TorchEngine(p.structure, FBC, devices=("cpu",), chunk_size=7))
        for eng in engs:
            eng.register("t", Y)
            idx = np.arange(len(Y))
            L, cnts = eng.fused_counts([p, p2], "t", idx, WeightSpec(cfg, np.bincount(label).astype(float)), label, T)
            L2 = eng.scores([p, p2], "t")
            assert np.array_equal(L, L2)
            _J, parts = objective(L2, label, T, cfg)
            for k, q in enumerate((p, p2)):
                ref = eng.counts(q, "t", idx, np.column_stack([parts.W_num[:, k], parts.W_den[:, k]]))
                for r in range(2):
                    for a, b in zip(cnts[k][r].arrays(), ref[r].arrays()):
                        if isinstance(eng, NumpyEngine):
                            assert np.array_equal(a, b)            # same sequences, same order
                        else:                                       # counts() drops zero-weight rows
                            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)   # from the batch
            eng.close()


def test_engine_results_do_not_depend_on_n_jobs(units):
    pytest.importorskip("numba")
    c, p, Y, *_ = _counts_problem(units)
    W = np.column_stack([c["w_num"], c["w_den"]])
    out = []
    for n_jobs in (1, 2, 4):
        eng = NumpyEngine(p.structure, FBC, n_jobs=n_jobs, chunk_size=3)
        eng.register("t", Y)
        out.append((eng.counts(p, "t", np.arange(len(Y)), W), eng.scores([p, p], "t")))
        eng.close()
    for cs, L in out[1:]:
        assert all(np.array_equal(a, b) for x, y in zip(cs, out[0][0]) for a, b in zip(x.arrays(), y.arrays()))
        assert np.array_equal(L, out[0][1])


@pytest.mark.parametrize("key,E,ub", [("E2", 2, ()), ("E005", 0.05, ()), ("E2_mu_A", 2, ("mu", "A"))])
def test_ebw_proposal_bit_identical(units, key, E, ub):
    _c, p, _Y, dom, num, den = _counts_problem(units)
    q, info = ebw_proposal(p, num, den, dom, E, ub)
    assert _pdiff(q, packed_to_params(units["ebw"][key])) == 0
    assert info["max_D"] == units["ebw"][key + "_info"]["max_D"]
    assert info["median_D"] == units["ebw"][key + "_info"]["median_D"]


def test_published_update_bit_identical(units):
    _c, p, _Y, dom, num, den = _counts_problem(units)
    P = units["published"]
    st = p.structure
    cases = [("tau100", {}, num, den),
             ("prev_tau3", dict(ismooth_backoff="previous", ismooth_tau_discrete=0.5, ismooth_tau=3), num, den),
             ("heavyden", {}, counts_from_packed_struct(P["num_heavy"], st), counts_from_packed_struct(P["den_heavy"], st))]
    for key, kw, nn, dd in cases:
        q, info = published_update(p, nn, dd, dom, **kw)
        ref = P[key + "_info"]
        assert _pdiff(q, packed_to_params(P[key])) == 0, key
        assert info["projection_changed_total"] == ref["projection_changed_total"]
        assert info["weak_sense_max_iterations"] == ref["weak_sense_max_iterations"]
        assert info["max_D"] == ref["max_D"]


def test_weak_sense_and_domain(units):
    for w in units["weak_sense"]:
        x, inf = weak_sense(w["gnum"], w["gden"], w["p0"], 100, 1e-10)
        assert np.array_equal(x, np.asarray(w["p"], float)) and inf["iterations"] == w["iterations"]
        assert inf["monotone"]
    _c, p, Y, dom, *_ = _counts_problem(units)
    Dm = units["domain"]
    q = project(packed_to_params(Dm["outside"]), dom)
    assert _pdiff(q, packed_to_params(Dm["projected"])) == 0
    b = at_bounds(q, dom)
    for k, v in Dm["at_bounds"].items():
        assert b[k] == pytest.approx(float(v), abs=1e-15)
    dom2 = build_domain(Y, p, prob_floor=1e-4, mean_margin=0.5, var_lo_frac=1e-3, var_hi_frac=10)
    np.testing.assert_allclose(dom2.var_lo, dom.var_lo, rtol=1e-12)
    np.testing.assert_allclose(dom2.mu_hi, dom.mu_hi, rtol=1e-12)


def test_objectives_match_matlab(units):
    cases = {name: o for name, o in units["objective"].items() if o["cfg"]["objective"] == "mmi"}
    assert len(cases) >= 2
    for name, o in cases.items():
        cf = o["cfg"]
        assert cf.get("mmi_recipe", "published") == "published"
        cfg = ObjectiveConfig("mmi", np.asarray(cf["log_prior"]), score_scale=cf.get("score_scale", 1.0),
                              N_stat=cf.get("N_stat", 0))
        J, parts = objective(o["L"], np.asarray(o["label"]) - 1, o["T"], cfg)
        assert J == pytest.approx(o["J"], rel=1e-13), name
        np.testing.assert_allclose(parts.W_num, o["W_num"], rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(parts.W_den, o["W_den"], rtol=1e-13, atol=1e-15)


def test_objective_weights_are_gradients():
    rng = np.random.default_rng(3)
    N, M = 30, 3
    label = np.r_[np.arange(M), rng.integers(0, M, N - M)]
    T = rng.integers(5, 40, N).astype(float)
    L = -rng.random((N, M)) * T[:, None] - 5
    lp = np.log(np.array([0.2, 0.3, 0.5]))
    for cfg, scale in [(ObjectiveConfig("mmi", lp, score_scale=1.0, N_stat=N), 1.0 / N),
                       (ObjectiveConfig("mmi", lp, score_scale=0.05, N_stat=N), 0.05 / N)]:
        J, parts = objective(L, label, T, cfg)
        G = (parts.W_num - parts.W_den) * scale
        h = 1e-6
        for n, k in [(0, 0), (3, 1), (7, 2), (12, 0), (20, 1)]:
            Lp = L.copy(); Lp[n, k] += h
            Lm = L.copy(); Lm[n, k] -= h
            fd = (objective(Lp, label, T, cfg)[0] - objective(Lm, label, T, cfg)[0]) / (2 * h)
            assert fd == pytest.approx(G[n, k], rel=1e-5, abs=1e-9)


# ------------------------------------------------------------------------------ full runs vs MATLAB
RUN_NAMES = [f"{m}_{t}" for t in ("T2", "T3ii", "T4ii", "H2") for m in ("mmi_k1", "mmi_k1e_3")] \
    + ["mmi_k1e_2_prev_T4ii", "mmi_auprc_k1e_3_T4ii", "mmi_k1e_3_sed_T4ii"]


def _run_like_matlab(runs, name, **extra):
    r = runs["R"][name]
    tr, va, init, ub = _toy(runs, r["toy"])
    o = dict(r["options"])
    o.pop("verbose", None)
    assert o.get("objective") == "mmi"
    if ub:
        o["update_blocks"] = ub
    if r.get("fb_variant", "sed") != "poise":
        o["fb_constants"] = FbConstants(variant=r.get("fb_variant", "sed"))
    return r, em_lsim_discriminative(tr, init, max_iteration=int(r["max_iteration"]), validation=va, verbose=0,
                                   **o, **extra)


@pytest.mark.parametrize("name", RUN_NAMES)
def test_full_run_matches_matlab(runs, name):
    r, res = _run_like_matlab(runs, name)
    h, mh = res.history, r["history"]
    Jm = np.atleast_1d(mh["objective"])
    assert len(h["objective"]) == len(Jm)
    np.testing.assert_allclose(h["objective"], Jm, rtol=1e-11, atol=0)
    np.testing.assert_array_equal(np.atleast_2d(mh["backtracks"])[:, 0], h["backtracks"][:, 0])
    np.testing.assert_array_equal(np.atleast_1d(mh["val_balanced_accuracy"]), h["val_balanced_accuracy"])
    assert res.diagnostics["checkpoint_iteration"] == r["checkpoint"]
    assert res.diagnostics["stop_reason"] == r["stop_reason"]
    for a, b in zip(res.final_models, r["final"]):
        assert _pdiff(a, packed_to_params(b)) < 1e-9
    # validation / training metrics recorded by both implementations
    for key in ("val_auroc", "val_auprc", "val_max_f1", "val_ba_best_threshold", "val_selection_value",
                "train_auroc", "train_auprc", "train_max_f1"):
        if key in mh:
            np.testing.assert_allclose(h[key], np.atleast_1d(mh[key]), rtol=1e-9, atol=1e-12, err_msg=key)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
@pytest.mark.parametrize("name", ["mmi_k1_T4ii", "mmi_k1e_3_H2", "mmi_k1e_2_prev_T4ii"])
def test_torch_cpu_matches_matlab(runs, name):
    r, res = _run_like_matlab(runs, name, engine="torch", devices=("cpu",))
    np.testing.assert_allclose(res.history["objective"], np.atleast_1d(r["history"]["objective"]), rtol=1e-11)
    assert res.diagnostics["checkpoint_iteration"] == r["checkpoint"]


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_torch_units_do_not_depend_on_device_count(runs):
    tr, va, init, _ = _toy(runs, "T4ii")
    a = em_lsim_discriminative(tr, init, max_iteration=4, validation=va, verbose=0, engine="torch", devices=("cpu",),
                             chunk_size=50)
    b = em_lsim_discriminative(tr, init, max_iteration=4, validation=va, verbose=0, engine="torch",
                             devices=("cpu", "cpu", "cpu"), chunk_size=50)
    assert np.array_equal(a.history["objective"], b.history["objective"])


@pytest.mark.skipif(N_CUDA < 1, reason="needs CUDA")
@pytest.mark.parametrize("name", ["mmi_k1e_3_T4ii", "mmi_k1_H2", "mmi_k1_T3ii", "mmi_k1e_3_sed_T4ii"])
def test_cuda_matches_matlab(runs, name):
    r, res = _run_like_matlab(runs, name, engine="torch", devices=("cuda:0",))
    np.testing.assert_allclose(res.history["objective"], np.atleast_1d(r["history"]["objective"]), rtol=1e-10)
    np.testing.assert_array_equal(np.atleast_2d(r["history"]["backtracks"])[:, 0], res.history["backtracks"][:, 0])
    assert res.diagnostics["checkpoint_iteration"] == r["checkpoint"]


@pytest.mark.skipif(N_CUDA < 2, reason="needs >= 2 CUDA devices")
def test_multi_gpu_equals_single_gpu(runs):
    tr, va, init, _ = _toy(runs, "T4ii")
    one = em_lsim_discriminative(tr, init, max_iteration=5, validation=va, verbose=0, engine="torch",
                               devices=("cuda:0",), chunk_size=40)
    many = em_lsim_discriminative(tr, init, max_iteration=5, validation=va, verbose=0, engine="torch",
                                devices=tuple(f"cuda:{i}" for i in range(N_CUDA)), chunk_size=40)
    assert np.array_equal(one.history["objective"], many.history["objective"])


# ------------------------------------------------------------------------------ mini-batches, resume
def test_minibatch_one_batch_equals_full_batch(runs):
    tr, va, init, _ = _toy(runs, "T4ii")
    a = em_lsim_discriminative(tr, init, max_iteration=4, validation=va, verbose=0)
    b = em_lsim_discriminative(tr, init, max_iteration=4, validation=va, verbose=0, minibatches=1, minibatch_seed=9)
    assert np.array_equal(a.history["objective"], b.history["objective"])


@pytest.mark.parametrize("score_scale", [1.0, 1e-3])
def test_minibatch_runs_and_is_reproducible(runs, score_scale):
    tr, va, init, _ = _toy(runs, "T3ii")
    kw = dict(score_scale=score_scale, minibatches=4, minibatch_seed=3, verbose=0)
    a = em_lsim_discriminative(tr, init, max_iteration=3, validation=va, **kw)
    b = em_lsim_discriminative(tr, init, max_iteration=3, validation=va, **kw)
    assert np.all(np.isfinite(a.history["objective"]))
    assert np.array_equal(a.history["objective"], b.history["objective"])
    c = em_lsim_discriminative(tr, init, max_iteration=3, validation=va, **{**kw, "minibatch_seed": 4})
    assert not np.array_equal(a.history["objective"][1:], c.history["objective"][1:])


def test_resume_reproduces_uninterrupted_run(runs, tmp_path):
    tr, va, init, _ = _toy(runs, "T4ii")
    full = em_lsim_discriminative(tr, init, max_iteration=6, validation=va, verbose=0, score_scale=0.01)
    state = str(tmp_path / "state.pkl")
    log = str(tmp_path / "log.jsonl")
    em_lsim_discriminative(tr, init, max_iteration=3, validation=va, verbose=0, score_scale=0.01,
                         state_path=state, log_path=log)
    res = em_lsim_discriminative(tr, init, max_iteration=6, validation=va, verbose=0, score_scale=0.01,
                               state_path=state, log_path=log, resume=True)
    assert np.array_equal(full.history["objective"], res.history["objective"])
    assert _pdiff(full.final_models[0], res.final_models[0]) == 0
    with open(log) as fh:
        assert len(fh.readlines()) == 7
    with pytest.raises(Exception):
        em_lsim_discriminative(tr, init, max_iteration=6, validation=va, verbose=0, score_scale=0.5,
                             state_path=state, resume=True)


def test_train_and_predict(runs):
    tr, va, init, _ = _toy(runs, "T3ii")
    res = em_lsim_discriminative(tr, init, max_iteration=5, validation=va, verbose=0)
    assert np.all(np.isfinite(res.history["objective"]))
    assert res.diagnostics["objective"] == "mmi" and res.diagnostics["iterations_run"] == 5
    L, pred, prob = lsim_discriminative_predict(res.models, va[0])
    assert L.shape == (len(va[0]), 2) and pred.shape == (len(va[0]),)
    np.testing.assert_allclose(prob.sum(axis=1), 1)
    assert set(np.unique(pred)) <= {0, 1}


# ------------------------------------------------------------------------------ metrics, selection, early stopping
def test_binary_metrics_match_matlab(units):
    from chmm_lsim.discriminative import binary_metrics

    if "metrics" not in units:
        pytest.skip("fixture without metrics (re-run export_disc_fixtures.m)")
    for case in units["metrics"]:
        m = binary_metrics(np.asarray(case["score"], float), np.asarray(case["label"]) > 0)
        for k in ("auroc", "auprc", "max_f1", "max_f1_threshold", "ba_best_threshold", "best_ba_threshold"):
            assert m[k] == pytest.approx(float(case["m"][k]), rel=1e-12, abs=1e-12), k


def test_binary_metrics_match_sklearn_and_brute_force():
    skm = pytest.importorskip("sklearn.metrics")
    from chmm_lsim.discriminative import binary_metrics, threshold_metrics, tune_threshold

    rng = np.random.default_rng(5)
    for trial in range(6):
        n = 300
        y = rng.random(n) < (0.2 + 0.1 * trial)
        s = np.round(rng.normal(size=n) + 1.2 * y, 1 if trial % 2 else 3)   # ties in odd trials
        m = binary_metrics(s, y)
        assert m["auroc"] == pytest.approx(skm.roc_auc_score(y, s), abs=1e-12)
        assert m["auprc"] == pytest.approx(skm.average_precision_score(y, s), abs=1e-12)
        thr = np.unique(s)
        f1 = [skm.f1_score(y, s >= t) for t in thr]
        ba = [skm.balanced_accuracy_score(y, s >= t) for t in thr] + [0.5]
        assert m["max_f1"] == pytest.approx(max(f1), abs=1e-12)
        assert m["ba_best_threshold"] == pytest.approx(max(ba), abs=1e-12)
        t = tune_threshold(s, y, "f1")
        assert threshold_metrics(s, y, t)["f1"] == pytest.approx(m["max_f1"], abs=1e-12)
        t = tune_threshold(s, y, "ba")
        assert threshold_metrics(s, y, t)["balanced_accuracy"] == pytest.approx(m["ba_best_threshold"], abs=1e-12)


@pytest.mark.parametrize("metric", ["auroc", "auprc", "max_f1", "ba_best_threshold", "objective"])
def test_selection_metric_picks_best_validation_iteration(runs, metric):
    tr, va, init, _ = _toy(runs, "T4ii")
    res = em_lsim_discriminative(tr, init, max_iteration=6, validation=va, verbose=0, score_scale=1e-3,
                               selection_metric=metric)
    h = res.history
    key = "val_objective" if metric == "objective" else f"val_{metric}"
    vals = h[key]
    best = np.flatnonzero(vals == np.nanmax(vals))
    ck = res.diagnostics["checkpoint_iteration"]
    assert ck in best
    assert ck == best[np.argmax(h["val_objective"][best])] or len(best) == 1
    for k in ("val_auroc", "val_auprc", "val_max_f1", "val_ba_best_threshold", "train_auroc"):
        assert np.all(np.isfinite(h[k]))


def test_early_stopping_and_resume(runs, tmp_path):
    tr, va, init, _ = _toy(runs, "T3ii")
    kw = dict(validation=va, verbose=0, score_scale=1e-3, selection_metric="auroc", early_stopping_patience=2)
    res = em_lsim_discriminative(tr, init, max_iteration=40, **kw)
    h = res.history
    n = int(h["iteration"][-1])
    if res.diagnostics["stop_reason"].startswith("early stopping"):
        v = h["val_auroc"]
        best_before = np.max(v[: n - 1])
        assert np.all(v[n - 1:] <= best_before)            # no strict improvement in the last 2 iterations
        assert v[res.diagnostics["checkpoint_iteration"]] == np.max(v)   # checkpoint = best (ties allowed)
    state = str(tmp_path / "s.pkl")
    a = em_lsim_discriminative(tr, init, max_iteration=2, state_path=state, **kw)
    b = em_lsim_discriminative(tr, init, max_iteration=40, state_path=state, resume=True, **kw)
    assert np.array_equal(b.history["objective"], h["objective"])
    assert b.diagnostics["stop_reason"] == res.diagnostics["stop_reason"]
    assert a.diagnostics["iterations_run"] == 2
    with pytest.raises(Exception):
        em_lsim_discriminative(tr, init, max_iteration=3, verbose=0, selection_metric="auroc")  # needs validation


def test_evaluate_models(runs):
    from chmm_lsim.discriminative import evaluate_models

    tr, va, init, _ = _toy(runs, "T3ii")
    ev = evaluate_models(init, {"train": tr, "val": va}, tune_on="val")
    assert ev["threshold"] is not None
    for name in ("train", "val"):
        m = ev[name]
        for k in ("auroc", "auprc", "max_f1", "balanced_accuracy", "balanced_accuracy_at_tuned", "f1_at_tuned"):
            assert np.isfinite(m[k])
    assert ev["val"]["balanced_accuracy_at_tuned"] == pytest.approx(ev["val"]["ba_best_threshold"])


def test_matlab_named_entry_points_and_aliases():
    """The entry points carry the MATLAB names; the former names stay as aliases."""
    import chmm_lsim as P
    assert P.em_lsim_discriminative is em_lsim_discriminative is train_discriminative
    assert P.lsim_discriminative_predict is lsim_discriminative_predict is predict
    assert {"em_lsim_discriminative", "lsim_discriminative_predict"} <= set(P.__all__) & set(dir(P))
