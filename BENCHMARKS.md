# CPU baseline v1

**Turn-11 numerical update:** NUMERICS.md and `profiling/numerics_results.csv`
record the float64-reference remedy study. Function-constant-specialized default
kernels retain the measured performance path; Gaussian M=512 now uses v1. The
new full 128-row dispatch matrix in `profiling/numerics_dispatch_results.csv`
passes unchanged numerical and 5% performance gates. Earlier baseline/turn-9
tables below remain their original measurements, not relabeled current tuning.

**Turn-9 verification update:** project-local prerequisites are now installed and
the production metal-cpp, CMake, GoogleTest and Google Benchmark smoke targets
compile/run. Earlier numbers below remain the originally recorded native-runner
measurements, not retroactively relabeled framework results. The new production
cooperative comparison, input-separation fix and dispatch policy are recorded in
DESIGN.md and `profiling/cooperative_results.csv`.

These are **measured native wall-clock results**, not fabricated Google Benchmark
output. Google Benchmark 1.9.4 registrations are implemented in
`bench/cpu_batch.cpp`; at the time of this recording shell DNS failed and local
CMake/uv were absent. The dependency-free `bench/cpu_baseline.cpp` uses the same
`bench/cpu_workload.hpp`, `cpu::fit_batch`, initial guesses, options and persistent
thread pool. The framework smoke was subsequently verified in turn 9; this
historical full-matrix table was not regenerated with the framework.

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

## Metal v1: one thread per fit

**Historical v1 recording, with its original parameter-parity failures.** The table below is
real GPU execution of the production embedded `src/metal/lm.metal`, driven through
the public shared-batch `Device::Metal` API by the test-only native SDK adapter.
The metal-cpp production host and Google Benchmark wrappers could not be compiled
then because downloads were unavailable. Do not label these figures as metal-cpp or
Google Benchmark execution results. No CPU fitting fallback/correction is used.

Machine/compiler match the CPU baseline above. Context creation embeds/compiles
MSL at runtime, with safe math, precise FP32 functions and explicit per-statement
contraction. All pipelines reported SIMD width 32 and used 256-thread groups,
derived from execution width/max-thread limits. N=100,000; inputs, seeds, noise,
initial parameters, fallbacks, constraints and LM defaults are identical to CPU v1.
Only y is transposed once to point-major shared storage outside the timed call.

Three GPU calls follow a 256-fit warm-up. GPU seconds are command-buffer
GPUEndTime-GPUStartTime after completion; wall seconds are an external steady-clock
measurement around the full shared-buffer API call, including config, encoding,
commit, wait and returned views. Both exclude context/pipeline creation and input
generation/transpose. Dispatch performs no array copies or blits. These are
ready-shared-buffer API times, not raw-image preparation/transfer-inclusive times.
Native and Google Benchmark paths share `bench/metal_workload.hpp`.

Each result is compared per fit against the float CPU reference (two CPU workers,
outside GPU timing). GPU/status/cost failures are never dropped from N. One CPU
fit-major and one GPU point-major dataset are resident during preparation; the
CPU dataset is released before timing. Largest observation payloads total
1,638,400,000 bytes during preparation. An earlier full diagnostic run measured
1,664,745,472 bytes peak RSS (1,587.63 MiB); datasets are memory-only and released.

| Model | M | GPU-time fits/s | Wall-time fits/s | Wall vs 10-core CPU | Strict parameter outlier rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian | 32 | 25,891,643 | 23,076,701 | 3.78x | 0 |
| Lorentzian | 32 | 26,506,743 | 23,380,191 | 2.03x | 0 |
| PseudoVoigt | 32 | 17,898,160 | 17,177,458 | 3.91x | 7 |
| TwoLorentzian | 32 | 4,542,125 | 4,485,898 | 1.85x | 0 |
| Gaussian | 128 | 10,432,787 | 9,969,054 | 4.42x | 0 |
| Lorentzian | 128 | 9,681,050 | 9,265,694 | 2.35x | 0 |
| PseudoVoigt | 128 | 8,717,446 | 8,364,614 | 5.52x | 0 |
| TwoLorentzian | 128 | 2,819,102 | 2,798,364 | 2.64x | 0 |
| Gaussian | 512 | 4,152,845 | 4,108,639 | 6.32x | 0 |
| Lorentzian | 512 | 3,643,651 | 3,606,566 | 3.16x | 0 |
| PseudoVoigt | 512 | 2,835,261 | 2,796,443 | 6.91x | 2 |
| TwoLorentzian | 512 | 786,327 | 784,105 | 2.90x | 0 |
| Gaussian | 2048 | 808,527 | 806,544 | 6.35x | 0 |
| Lorentzian | 2048 | 605,114 | 603,909 | 3.03x | 0 |
| PseudoVoigt | 2048 | 503,862 | 503,081 | 6.24x | 0 |
| TwoLorentzian | 2048 | 138,392 | 137,616 | 2.02x* | 3* |

### Accuracy qualification — not a fully green matrix

- All **1.6 million status and chi-squared comparisons match**. Cost bound is
  `1e-6 + 2e-5*abs(CPU)`. GPU has the same 33 singular paired rows as CPU.
- The strict 2,832-fit unit suite passes every parameter at
  `2e-4 + 2e-4*abs(CPU)`; that tolerance was not relaxed.
- The full matrix exposes **12** rows beyond that stricter parameter diagnostic:
  seven pseudo-Voigt M=32, two M=512, three paired M=2048. The pseudo-Voigt maxima
  are about 0.000324/0.000290, within the CPU reference's already documented float
  floor `5e-4*(1+abs(parameter))`, but still reported as strict outliers.
- **Two paired rows also exceed that reference floor**, with worst absolute
  disagreement about 0.00516. Both CPU and GPU report convergence to poor local
  minima placing both components on one physical peak. At row 18,304, chi-squared
  is approximately 187.99 and both centers are negative; at row 41,636 it is
  approximately 172.708 and both centers are positive, despite the input's
  separated left/right peaks. Matching objective/status is not unique-parameter
  equivalence. These remain genuine v1 limitations, not certified matches.
- `*` The largest paired performance row is therefore **timing only, not
  accuracy-qualified**. The native benchmark emits all rows/mismatch diagnostics
  and deliberately returns exit 1 for the two reference-floor violations. The
  Google Benchmark wrapper marks the corresponding case with SkipWithError.
  Neither CPU reference, initializer, options nor strict tests were changed to
  manufacture a pass. That original-input benchmark validator was red.

Observed timing is not a promise of speedup on every device. For the largest
paired case GPU/wall medians were 0.722585/0.726661 s. Source/host compilation,
zero-copy lifetime, hardware validation and numerical parity are separate claims:
the native driver verifies the latter shared-kernel/API behavior, not executed
NS::SharedPtr production ownership. Production compilation/execution was subsequently
verified in turn 9; the historical timing row still comes from the native driver.

**Current regression update:** after the data-only shoulder-separation fix, the
production v1 N=100,000 matrix was rerun. All 16 cases pass CPU status, cost and
reference-floor checks, with zero paired parameter outliers at M=2048. Nine
pseudo-Voigt rows still exceed the stricter diagnostic but remain within the
unchanged float reference floor. The current full benchmark exits zero. Earlier
failure counts/figures above are preserved rather than silently overwritten;
new guesses mean a new comparison, not retroactive qualification of old rows.

### Reproduce Metal measurements

With normal Metal dependencies available, build Release with BATCHFIT_METAL=ON
and run `build/metal-release/batchfit_metal_baseline`. Google Benchmark cases
`metal_lm_*` use manual GPU time and expose both `gpu_fits_per_second` and
`wall_fits_per_second` counters; use `--benchmark_repetitions=3`. `metal_lm_smoke`
uses only 257 fits and is part of the ordinary Metal check gate.

The native SDK fallback used here is explicit and test-only:

```sh
mkdir -p .agent-local/check/generated
python3 tools/embed_metal.py .agent-local/check/generated/lm_source.hpp
clang++ -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++20 -O3 -DNDEBUG -pthread -fobjc-arc -Wall -Wextra -Wpedantic -Werror -Iinclude -Isrc/metal -I.agent-local/check/generated -x objective-c++ tests/metal_runtime_native.mm src/cpu/thread_pool.cpp src/cpu/npy.cpp src/cpu/synthetic.cpp bench/metal_baseline.cpp -framework Foundation -framework Metal -o .agent-local/check/batchfit_metal_baseline
.agent-local/check/batchfit_metal_baseline  # original turn-7 guesses exited 1; corrected turn-9 seeds pass
rm .agent-local/check/batchfit_metal_baseline .agent-local/check/generated/lm_source.hpp
rmdir .agent-local/check/generated .agent-local/check
```

`--smoke` instead runs 257 fits at M=32 for each model and passes parity. Keep the
historical full-matrix failure visible. The turn-9 production full-matrix rerun,
not smoke success alone, establishes the corrected-seed regression result.

## Out-of-core pipeline (turn 10)

These production Release measurements are **input-file to CPU-verified streamed
results**, including NPY validation, window mapping/page faults, conversion,
initial-file reads, point-major transpose, shared allocation, async submission,
completion/readback, every-output checking and event-stream flush. Shader/library
compilation and dataset generation are outside that interval. The benchmark
consumer verifies rather than persists full result arrays; the CLI separately
tests streamed NPY output/flush. Only two CPU workers are used. No concurrent
batchfit build/test ran during a measurement; OS disk/cache load is not controlled,
and no global cache flush or OS settings were used.

Two independently fully written files have **10,240,000,128 bytes** each (10.24 GB,
9.54 GiB), not sparse extents or a header pretending to be a large dataset:

- float64: N=312,500, M=4,096, automatically selected threadgroup-per-fit.
- native float32: N=5,000,000, M=512, automatically selected SIMD-per-fit.

Both use TwoLorentzian with baseline, 64 deterministic noiseless spectral
patterns repeated across all rows, and initial guesses derived from each pattern's
stored float32 observations, then streamed from `initial.npy`. Patterns are
evaluated in double before explicit float32 conversion for CPU/GPU fitting.
Cached `fit_single<float>` results for those identical inputs allow checking
**every fit**, not a sample, with no full-N reference/output allocation. Parameter
tolerance stays `2e-4 + 2e-4*abs(CPU)` and chi-squared `1e-6 + 2e-5*abs(CPU)`;
status must match. No rejected/failed row is dropped. Success-path checking avoids
formatting millions of diagnostic strings, but failures retain detailed messages.

The automatic three-slot chunk is also used unchanged for the one-slot comparison.
Both depths use the same read-ahead policy, so depth 1 can already overlap a
next-chunk read-ahead request with current GPU work; it is **not** a fully
serialized I/O baseline. Depth 3 permits preparation/readback overlap and at most
three committed buffers, not a promise that the producer fills them on every
workload. Small smoke tests actually reach three pending buffers. Disk-bound
full-file runs and timestamp overlap are qualified in PROFILING.md.

This is not the original noisy CPU-baseline workload (different N/M, data and
on-disk dtype); no unsupported GPU-versus-CPU speedup is inferred from comparing
these numbers to earlier tables. The file is larger than the configured working
budget, but still below this Mac's **16 GiB physical RAM**. Bounded process RSS
does not imply bounded global/reclaimable filesystem cache, a cold-cache result,
or a measurement on a machine with less than 10 GB total RAM.

### Recorded results — Apple M4, final production callback bridge

One final run per cell, same file/chunk/initials within each comparison; these are
not medians or a cold-cache benchmark. MaxBufferLength was **9,534,832,640 bytes**;
recommended working set **12,713,115,648 bytes**. Automatic chunks were 2,560 fits
(float64, 123 chunks) and 25,344 fits (float32, 198 chunks), including short tails.

| Input / M / fits | Slots | Wall s | Fits/s | Input GB/s | Observed peak pending |
|---|---:|---:|---:|---:|---:|
| float64 / 4096 / 312,500 | 1 | 7.494164 | 41,699 | 1.3664 | 1 |
| float64 / 4096 / 312,500 | 3 | 8.165082 | 38,273 | 1.2541 | 1 |
| float32 / 512 / 5,000,000 | 1 | 11.580432 | 431,763 | 0.8843 | 1 |
| float32 / 512 / 5,000,000 | 3 | 10.469354 | 477,584 | 0.9781 | 1 |

All **10,625,000** per-fit comparisons across those four rows passed status,
parameter and cost checks; all fits converged. No CPU fallback. Program peak RSS,
including generation and both depths, was **258,195,456 bytes / 246.23 MiB** for
float64 and **274,153,472 bytes / 261.45 MiB** for float32. Three-slot shared payloads
were 126,462,228 and 161,498,388 bytes; largest single mappings 83,886,208 and
51,904,640 bytes, respectively. These remain bounded, not proportional to N.
Allocated on-disk bytes were 10,242,494,464 / 10,250,883,072, confirming dense files.
Generation/write/close plus small CPU reference preparation took 6.1332 / 7.4724 s,
outside the fitting interval. Both large datasets were removed after the runs.

The native-float32 three-slot run achieved **1.106×** the one-slot throughput;
the float64 three-slot run was **8.95% slower** in wall time. Do not claim a
universal pipelining speedup or sustained three-buffer occupancy from these data.
The producer remained page-in/I/O limited: in the float64 three-slot run,
map/convert took 5.320 s of 6.034 s preparation; transpose took 0.680 s. Float32
map/convert took 7.720 s of 8.966 s preparation; transpose took 1.037 s. Native
per-program counters recorded roughly 625k/635k major faults per run. CPU time
and wall time distinguish waiting/kernel overhead from arithmetic conversion.
Large-file observed peak pending was one; the unchanged small-memory smoke
actually reaches three. Three slots provide capacity, not faster disk bandwidth.
Earlier diagnostic runs varied substantially with mapping advice/cache/load;
only the final-source measurements above are used as the reproducible baseline.

### Reproduction commands

```sh
export PATH="$PWD/.venv/bin:$PATH"
cmake -S . -B build/metal-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBATCHFIT_METAL=ON -DBATCHFIT_BUILD_TESTS=OFF -DBATCHFIT_BUILD_BENCHMARKS=ON
cmake --build build/metal-release --target batchfit_pipeline_baseline --parallel 2
build/metal-release/batchfit_pipeline_baseline
build/metal-release/batchfit_pipeline_baseline --float32
```

Each runner checks free space, creates only its uniquely owned
`.agent-local/pipeline-benchmark-*` directory, writes/closes the dense files,
executes depths 1 and 3, prints actual byte allocation/RSS/counters and removes
the dataset on completion/failure. Existing `profiling/pipeline_events.json` and
`profiling/pipeline_float32_events.json` are refused: archive them deliberately
before rerunning. Retained event JSON is application/Mach instrumentation, **not
Metal System Trace**. Add `--smoke` for the bounded ordinary gate; it never creates
a multi-GB dataset. A forced process kill may require cleanup of that exact owned
directory after verifying the process has ended.
