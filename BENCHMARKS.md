# CPU baseline v1

These are **measured native wall-clock results**, not fabricated Google Benchmark
output. Google Benchmark 1.9.4 registrations are implemented in
`bench/cpu_batch.cpp`, but this sandbox cannot download it (shell DNS fails and
CMake/uv are absent). The dependency-free `bench/cpu_baseline.cpp` uses the same
`bench/cpu_workload.hpp`, `cpu::fit_batch`, initial guesses, options and persistent
thread pool. Google Benchmark execution remains **unverified** until local
dependency setup is available.

Every subsequent GPU performance step must compare against this workload and
these CPU baselines, reporting accuracy/status changes as well as throughput.

## Machine and build

- Mac model identifier: **Mac16,10**, as reported by `sysctl hw.model`.
- Chip: **Apple M4**, 10 physical/logical CPU cores, no affinity/QoS changes.
- RAM: **16 GiB**; macOS **27.2**, arm64.
- Compiler: Apple Clang **21.0.0 (clang-2100.4.1.1)**; installed SDK
  `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk` used read-only.
- Flags: `-std=c++20 -O3 -DNDEBUG -pthread`; **no OpenMP, fast-math or LTO**.
- Scalar: **float32**, matching Metal's supported precision. Double remains
  covered by reference/batch correctness tests, not this timing matrix.

## Method

N is **100,000** in every case. M is 32, 128, 512 or 2048. The two modes use one
worker or `std::thread::hardware_concurrency()` = **10 workers**. These are actual
`std::jthread` workers, not Google Benchmark's own Threads(n) facility. The calling
thread waits. Ordinary checks use at most two compute workers; this full matrix
uses the explicitly requested all-core benchmark exception.

Report the median of **three timed API calls** after a 256-fit warm-up per mode.
One reusable pool persists through all three calls. Time includes four result
vector allocations/initialization, job dispatch, all single-fit validation/LM
work, and synchronization until completion. It excludes dataset generation,
automatic initial-guess preparation, pool construction/destruction, output
validation/hash computation and result destruction. Each call fits all N rows;
no failed rows are dropped. N/median_seconds is the fits/s value below.

One full dataset is resident at a time, reused by single/all-core modes. The
largest observation array is 819,200,000 bytes; generation uses 1,024-fit chunks.
Measured process peak RSS across the complete matrix was **1,045,463,040 bytes
(997.03 MiB)**, including allocator retention from earlier cases. Datasets are
in-memory only and were released; no generated dataset or scratch result file
is retained.

### Reproducible inputs

- Fit-major `y[N*M]`, `init[N*P]`, shared linear `x[M]` on [-5,5].
- Gaussian noise, linear per-fit RMS SNR **20**.
- Seed `20261107 + 137*synthetic_model_id + M` (model ids 0,1,2,3 in model order).
- Heights 1–3. Single-peak centers -1–1. Gaussian sigma and Lorentzian HWHM
  0.6–1.0; pseudo-Voigt FWHM 1.4–2.0 and mixing 0–1. These widths keep M=32
  meaningfully sampled and remain identical distributions at all M values.
- Paired centers -2–-1 and 1–2, each HWHM 0.5–0.8, intercept 0.3–0.5 and slope
  -0.02–0.02; both heights 1–3.
- Initial parameters use the data-only automatic guesses, **not true parameters**.
  Failed guesses use fixed non-oracle seeds: single peaks [2,0,0.8], pseudo-Voigt
  [2,0,1.7,0.5], paired [2,-1.5,0.65,2,1.5,0.65,0.4,0]. Fallbacks are counted.
- Unchanged LM defaults: 100 attempts, lambda 1e-3, float gtol/xtol/ftol
  1e-5/1e-6/1e-6. Nonnegative heights, centers inside x, widths in
  [grid_spacing/8,x_span], mixing in [0,1]; baseline coordinates unbounded.

## Results

All rates are **attempted fits/s**, including numerical failures. Counts and mean
iterations are identical in both worker modes and across all three repetitions.
All full-result hashes (parameters/status/iterations/chi-squared) match across
worker modes and repetitions; smaller unit suites compare every field bitwise
against individual `fit_single` calls for float and double.

| Model | M | 1 worker fits/s | 10 workers fits/s | Speedup | Converged / 100,000 | Seed fallbacks | Mean attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian | 32 | 1,387,352 | 6,104,587 | 4.40x | 100,000 | 0 | 3.367 |
| Lorentzian | 32 | 2,176,731 | 11,529,481 | 5.30x | 100,000 | 0 | 3.324 |
| PseudoVoigt | 32 | 852,343 | 4,394,928 | 5.16x | 100,000 | 0 | 3.614 |
| TwoLorentzian | 32 | 477,593 | 2,426,905 | 5.08x | 100,000 | 135 | 4.206 |
| Gaussian | 128 | 490,293 | 2,255,177 | 4.60x | 100,000 | 0 | 3.294 |
| Lorentzian | 128 | 781,280 | 3,942,376 | 5.05x | 100,000 | 0 | 3.247 |
| PseudoVoigt | 128 | 303,533 | 1,516,474 | 5.00x | 100,000 | 0 | 3.523 |
| TwoLorentzian | 128 | 203,469 | 1,060,210 | 5.21x | 100,000 | 4 | 3.777 |
| Gaussian | 512 | 136,374 | 649,667 | 4.76x | 100,000 | 0 | 3.199 |
| Lorentzian | 512 | 217,461 | 1,142,094 | 5.25x | 100,000 | 0 | 3.201 |
| PseudoVoigt | 512 | 84,899 | 404,679 | 4.77x | 100,000 | 0 | 3.493 |
| TwoLorentzian | 512 | 59,500 | 270,610 | 4.55x | 100,000 | 0 | 3.678 |
| Gaussian | 2048 | 31,348 | 126,915 | 4.05x | 100,000 | 0 | 3.317 |
| Lorentzian | 2048 | 50,975 | 199,180 | 3.91x | 100,000 | 0 | 3.457 |
| PseudoVoigt | 2048 | 19,430 | 80,618 | 4.15x | 100,000 | 0 | 3.841 |
| TwoLorentzian | 2048 | 14,672 | 68,121 | 4.64x | 99,967 | 0 | 3.986 |

The largest paired case has **33 `singular_matrix`** results, zero `max_iterations`
and zero `invalid_input`, confirmed by a separate status inspection. These are
unchanged float reference outcomes, not pool errors; keep them in GPU comparisons
instead of silently discarding them or timing only successful fits.

These are three observations per mode, not confidence intervals or guaranteed
hardware ceilings. OS scheduling, cache/thermal state and background load are not
controlled. For example, Gaussian M=2048 serial times ranged 3.057831–3.702066 s
(median 3.189960 s); Lorentzian M=32 all-core times ranged 0.008629–0.010090 s
(median 0.008673 s). Re-measure CPU alongside GPU on the target machine. GPU
kernel-only times must be labeled separately from equivalent end-to-end API times.

## Reproduce

With the pinned project-local CMake/Google Benchmark environment available:

```sh
export PATH="$PWD/.venv/bin:$PATH"
cmake -S . -B build/cpu-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -DBATCHFIT_METAL=OFF -DBATCHFIT_BUILD_TESTS=OFF
cmake --build build/cpu-release --parallel 2
build/cpu-release/batchfit_bench --benchmark_filter='^cpu_batch_(gaussian|lorentzian|pseudo_voigt|two_lorentzian)/' --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
```

Registrations use N/M/worker-flag arguments; worker flag 0 means all detected
cores, and 1 means one worker. `UseRealTime()` is required for the pool, and the
explicit `fits_per_second` counter is a wall-clock rate. Each registration uses
one iteration (N full fits); repetitions provide the three samples. The framework
suite also reports convergence, fallback and iteration counters. **This framework
command was not run in the current environment.**

The following native command **was run** and produced the recorded table:

```sh
mkdir -p .agent-local/check
clang++ -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++20 -O3 -DNDEBUG -pthread -Wall -Wextra -Wpedantic -Werror -Iinclude src/cpu/thread_pool.cpp src/cpu/npy.cpp src/cpu/synthetic.cpp bench/cpu_baseline.cpp -o .agent-local/check/batchfit_cpu_baseline
.agent-local/check/batchfit_cpu_baseline
rm .agent-local/check/batchfit_cpu_baseline
rmdir .agent-local/check
```

Its stdout is CSV with min/median/max seconds, median fits/s, status/seed/iteration
counts and result hash. `--smoke` instead uses N=128, M=32, one repetition and
1/2 workers for a cheap normal-budget check. `scripts/check.sh` similarly filters
Google Benchmark to residual and small CPU-batch smoke runs; the full all-core
matrix is an explicit performance step, never an ordinary Debug test workload.
