# Runtime benchmark: POISE vs SED vs fast/SVI

`runtime_benchmark.py` measures the cost of the LSIM backward recursions with the package's batched PyTorch backend.

| mode | what is timed | default setting |
|---|---|---|
| `backward` | the backward recursion only; one shared forward pass per model is excluded. Median and IQR of 20 runs after 3 warm-ups, plus the peak memory the backward pass adds | T = 1000, C, N ∈ {2, 4, 8, 16, 32} |
| `estep` | one full E-step (emissions, forward and backward passes, sufficient statistics) over a large dataset | 100,000 sequences, T = 100 |

Models are random LSIMs with N states per channel (observation dimensions 1–5, one Gaussian per state, one channel with outgoing strength drawn uniformly on [0, C]).

## Run

```bash
pip install "chmm-lsim[torch]"        # a CUDA build of PyTorch for GPU runs

# CPU, float64, one thread
python runtime_benchmark.py backward --device cpu --out backward_cpu.csv
python runtime_benchmark.py estep    --device cpu --out estep_cpu.csv

# GPU, float32, single sequences and batches of 1024
python runtime_benchmark.py backward --device cuda --B 1 1024 --out backward_gpu.csv
python runtime_benchmark.py estep    --device cuda --out estep_gpu.csv
```

Use `--C` and `--N` to choose a smaller grid. On CPU, the E-step of large models is timed on a subset of sequences (about `--cpu-seconds` per method) and scaled linearly to `--n-trials`; the `measured_trials` column says how many were run. Configurations that do not fit in memory are written with a `skipped` status.

Output columns: `mode, device, dtype, B, C, N, T, method, median_s, iqr_s, peak_mem_bytes, work_mem_bytes, measured_trials, status`. `peak_mem_bytes` includes the returned backward and smoothed arrays, which are the same size for every method; `work_mem_bytes` excludes them.

## Reference results

AMD EPYC 7402P (one thread, float64) and NVIDIA A30 24 GB (float32); PyTorch 2.14, Python 3.12.

| | SED time / POISE time |
|---|---|
| backward pass, CPU | 2.4–4.3× |
| backward pass, GPU, one sequence | 2.3–5.5× |
| backward pass, GPU, 1,024 sequences | 2.3–4.0× |
| full E-step, 100,000 sequences, GPU | 1.1–2.9× |
| full E-step, 100,000 sequences, CPU | 1.5–3.8× |

The advantage grows mainly with the number of channels C (about 2.4× at C = 2 and 3.5–5.5× for the backward pass at C = 32). It is smaller for a full E-step because emissions, the forward pass and the statistics cost the same for both methods. SED needs 1.2–2.9× POISE's working memory. fast/SVI runs at about the speed of POISE. Batched GPU execution was 18–106× faster than one CPU thread for the full E-step.
