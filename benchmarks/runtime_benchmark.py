"""Runtime and memory of the LSIM backward recursions (POISE, SED, fast/SVI).

Two modes, both using chmm_lsim's batched PyTorch backend unchanged:

  backward  Time ONLY the backward recursion. One forward pass per configuration is
            shared by all methods and excluded from timing and from the memory
            baseline. Reports the median and IQR over `--repeats` runs after
            `--warmup` runs, and the peak memory the backward pass adds (GPU:
            torch.cuda.max_memory_allocated; CPU: the torch profiler's memory
            timeline). The peak includes the returned beta/gamma arrays, which are
            the same size for every method; `work_mem_bytes` excludes them.

  estep     Wall time of one full E-step (emissions, forward and backward passes and
            sufficient statistics: `forward_backward_batch(mode="stats")`) over
            `--n-trials` sequences, chunked automatically. On CPU, large models are
            timed on a subset of sequences (about `--cpu-seconds` per method) and
            scaled linearly to `--n-trials`; `measured_trials` says how many ran.

Models follow the random-LSIM generator of the paper: N states in every channel,
observation dimensions 1-5, Gaussian means N(state, 1), variances U(1, 5), transition
rows U(0,1) normalized, and one target channel with outgoing strength U[0, C].

Examples
  python runtime_benchmark.py backward --device cpu --dtype float64 --C 4 8 --N 4 8 --out cpu.csv
  python runtime_benchmark.py backward --device cuda --dtype float32 --B 1 1024 --out gpu.csv
  python runtime_benchmark.py estep --device cuda --n-trials 100000 --T 100 --out estep_gpu.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import statistics
import time

import numpy as np
import torch

import chmm_lsim.backends.torch_batched as tb
from chmm_lsim import ChannelGMM, FbConstants, LSIMParams, LSIMStructure

VARIANTS = {"POISE": FbConstants(variant="poise"), "SED": FbConstants(variant="sed"),
            "SVI": FbConstants(variant="fast")}


# ---------------------------------------------------------------- models and data

def random_lsim(C: int, N: int, seed: int) -> tuple[LSIMStructure, LSIMParams]:
    rng = np.random.default_rng(seed)
    dims = rng.integers(1, 6, size=C)
    st = LSIMStructure((N,) * C, tuple(int(d) for d in dims), (1,) * C)
    pi_0, transition, emissions = np.zeros(st.S), np.zeros((st.S, st.S)), []
    for xi in range(C):
        p = rng.random(N)
        pi_0[st.state_slice(xi)] = p / p.sum()
        for c in range(C):
            a = rng.random((N, N))
            transition[st.state_slice(c), st.state_slice(xi)] = a / a.sum(axis=1, keepdims=True)
        d = int(dims[xi])
        mu = np.arange(1, N + 1)[None, :] + rng.standard_normal((d, N))
        var = 1.0 + 4.0 * rng.random((d, N))
        emissions.append(ChannelGMM(np.ones((N, 1)), mu[:, :, None], var[:, :, None], True))
    target, per_column = int(rng.integers(C)), rng.random()      # theta_out of target ~ U[0, C]
    coupling = np.zeros((C, C))
    coupling[target, :] = per_column
    others = [c for c in range(C) if c != target]
    for col in range(C):
        shares = rng.random(C - 1)
        coupling[others, col] = (1 - per_column) * shares / shares.sum()
    params = LSIMParams(st, pi_0, transition, coupling, emissions)
    params.validate()
    return st, params


def sample(st: LSIMStructure, params: LSIMParams, B: int, T: int, seed: int) -> np.ndarray:
    """B sequences (B, D, T) from the LSIM (simulate_lsim, vectorized over the batch)."""
    rng = np.random.default_rng(seed)
    C, N = st.C, st.channel_state_num[0]
    blocks = [[params.transition[st.state_slice(cp), st.state_slice(c)] for c in range(C)] for cp in range(C)]
    blocks = np.array(blocks)                                  # (C', C, N, N)
    X = np.zeros((B, C, T), dtype=np.int64)

    def draw(p):
        u = rng.random(p.shape[:2] + (1,))
        return np.minimum((np.cumsum(p, axis=-1) < u).sum(-1), N - 1)

    pi = np.stack([params.pi_0[st.state_slice(c)] for c in range(C)])
    X[:, :, 0] = draw(np.broadcast_to(pi, (B, C, N)))
    for t in range(1, T):
        p = np.zeros((B, C, N))
        for cp in range(C):
            p += params.coupling[cp][None, :, None] * blocks[cp][:, X[:, cp, t - 1], :].transpose(1, 0, 2)
        X[:, :, t] = draw(p)
    Y = np.empty((B, st.D, T))
    for c, g in enumerate(params.emissions):
        mu, sd = g.mu[:, :, 0][:, X[:, c, :]], np.sqrt(g.sigma[:, :, 0])[:, X[:, c, :]]
        Y[:, st.dim_slice(c), :] = (mu + sd * rng.standard_normal(mu.shape)).transpose(1, 0, 2)
    return Y


# ---------------------------------------------------------------- measurement

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, device, warmup, repeats):
    for _ in range(warmup):
        out = fn()
        del out
    ts = []
    for _ in range(repeats):
        gc.collect()
        sync(device)
        t0 = time.perf_counter()
        out = fn()
        sync(device)
        ts.append(time.perf_counter() - t0)
        del out
    q25, q75 = np.percentile(ts, [25, 75])
    return statistics.median(ts), float(q75 - q25)


def peak_memory(fn, device):
    """Peak memory added by fn() above what is allocated before it."""
    if device.type == "cuda":
        gc.collect()
        sync(device)
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = fn()
        sync(device)
        peak = torch.cuda.max_memory_allocated() - before
        del out
        return int(peak)
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,
                 with_stack=True) as prof:
        out = fn()
    del out
    cur = peak = 0
    created = set()
    for _t, action, (key, _v), size in prof._memory_profile().timeline:
        if key.device.type != "cpu":
            continue
        if str(action).endswith("CREATE"):
            created.add(key)
            cur += size
        elif str(action).endswith("DESTROY") and key in created:
            cur -= size
        peak = max(peak, cur)
    return int(peak)


# ---------------------------------------------------------------- modes

def run_backward(a, device, dtype, writer):
    backward = {"POISE": tb._backward_poise_batch, "SVI": tb._backward_fast_batch}
    for B in a.B:
        for C in a.C:
            for N in a.N:
                base = dict(mode="backward", device=device.type, dtype=a.dtype, B=B, C=C, N=N, T=a.T)
                try:
                    st, params = random_lsim(C, N, a.seed + 1000 * C + N)
                    Y = sample(st, params, B, a.T, a.seed + 1000 * C + N + 1)
                    kernel = tb.TorchKernel.build(st, params, device=device, dtype=dtype,
                                                  pi0_reg=FbConstants().pi0_reg)
                    idx, lengths = list(range(B)), torch.full((B,), a.T, dtype=torch.int64, device=device)
                    with torch.inference_mode():
                        emis, _, _ = tb.compute_emissions_torch(kernel, torch.as_tensor(Y, dtype=dtype, device=device))
                        alpha, scale, _ = tb._forward_pass_batch(kernel, emis, idx, lengths, False)
                        del emis
                except torch.OutOfMemoryError:
                    writer.writerow({**base, "status": "skipped: out of memory in the forward pass"})
                    torch.cuda.empty_cache()
                    continue
                out_bytes = 2 * alpha.numel() * alpha.element_size()
                for name, fbc in VARIANTS.items():
                    if name == "SED":
                        fn = lambda fbc=fbc: tb._backward_sed_batch(kernel, alpha, scale, fbc, idx, lengths, False, "block")  # noqa: E731
                    else:
                        fn = lambda f=backward[name], fbc=fbc: f(kernel, alpha, scale, fbc, idx, lengths, False)  # noqa: E731
                    try:
                        with torch.inference_mode():
                            mem = peak_memory(fn, device)
                            med, iqr = timed(fn, device, a.warmup, a.repeats)
                        row = {**base, "method": name, "median_s": med, "iqr_s": iqr, "peak_mem_bytes": mem,
                               "work_mem_bytes": mem - out_bytes, "status": "ok"}
                    except torch.OutOfMemoryError:
                        row = {**base, "method": name, "status": "skipped: out of memory in the backward pass"}
                    writer.writerow(row)
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    print({k: row[k] for k in ("device", "B", "C", "N", "method", "status")},
                          f"{row.get('median_s', float('nan')) * 1e3:.1f} ms", flush=True)
                del kernel, alpha, scale


def run_estep(a, device, writer):
    for C in a.C:
        for N in a.N:
            st, params = random_lsim(C, N, a.seed + 1000 * C + N)
            distinct = [y for k in range(0, min(a.distinct, a.n_trials), 1000)
                        for y in sample(st, params, min(1000, a.distinct - k), a.T, a.seed + k)]
            data = [distinct[i % len(distinct)] for i in range(a.n_trials)]   # values do not affect runtime
            base = dict(mode="estep", device=device.type, dtype=a.dtype, B=a.n_trials, C=C, N=N, T=a.T)

            def run(trials, fbc):
                tb.forward_backward_batch(st, params, trials, fbc, mode="stats", device=device.type, dtype=a.dtype)
                sync(device)

            for name, fbc in VARIANTS.items():
                if name not in a.methods:
                    continue
                if device.type == "cuda":
                    run(data[:2048], fbc)                                         # warm-up
                    med, iqr = timed(lambda: run(data, fbc), device, 0, a.repeats)
                    measured = a.n_trials
                else:
                    probe = data[:min(20, a.n_trials)]
                    run(probe, fbc)
                    t0 = time.perf_counter()
                    run(probe, fbc)
                    per_trial = (time.perf_counter() - t0) / len(probe)
                    measured = int(min(a.n_trials, max(len(probe), a.cpu_seconds / per_trial)))
                    t0 = time.perf_counter()
                    run(data[:measured], fbc)
                    med, iqr = (time.perf_counter() - t0) * a.n_trials / measured, float("nan")
                row = {**base, "method": name, "median_s": med, "iqr_s": iqr, "measured_trials": measured, "status": "ok"}
                writer.writerow(row)
                print({k: row[k] for k in ("device", "C", "N", "method")}, f"{med:.2f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["backward", "estep"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default=None, choices=["float64", "float32"],
                    help="default: float64 on CPU, float32 on CUDA")
    ap.add_argument("--C", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--N", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--B", type=int, nargs="+", default=[1], help="backward mode: batch sizes")
    ap.add_argument("--T", type=int, default=None, help="default: 1000 (backward), 100 (estep)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=None, help="default: 20 (backward), 3 (estep, GPU)")
    ap.add_argument("--n-trials", type=int, default=100_000)
    ap.add_argument("--distinct", type=int, default=10_000, help="estep: distinct simulated sequences")
    ap.add_argument("--cpu-seconds", type=float, default=60.0)
    ap.add_argument("--methods", nargs="+", default=["POISE", "SED"], help="estep: methods to time")
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.dtype = a.dtype or ("float32" if a.device == "cuda" else "float64")
    a.T = a.T or (1000 if a.mode == "backward" else 100)
    a.repeats = a.repeats or (20 if a.mode == "backward" else 3)

    torch.set_num_threads(1)                  # single-threaded CPU timings
    device = torch.device(a.device)
    fields = ["mode", "device", "dtype", "B", "C", "N", "T", "method", "median_s", "iqr_s",
              "peak_mem_bytes", "work_mem_bytes", "measured_trials", "status"]
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if new:
            writer.writeheader()
        if a.mode == "backward":
            run_backward(a, device, getattr(torch, a.dtype), writer)
        else:
            run_estep(a, device, writer)


if __name__ == "__main__":
    main()
