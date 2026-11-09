# batchfit

C++20 infrastructure for fitting batches of small independent nonlinear models
on CPU and Apple Metal. See [DESIGN.md](DESIGN.md) for the fitting API and solver
contract. Shared Gaussian, Lorentzian, pseudo-Voigt and paired-Lorentzian models
compile as both C++ and MSL. Single-fit/batched CPU LM solvers support float and
double. Float32 Metal provides thread-, SIMD-, and threadgroup-per-fit paths with
measured automatic dispatch. See the Metal sections for validation and limits.
Python bindings expose `batchfit.fit` through pybind11/scikit-build-core; see
[the Python API](python/README.md) for precision, copy and ownership rules.
The [unified benchmark](bench/README.md) compares SciPy loops, CPU and forced/
automatic Metal paths from 1,000 to 10 million fits with bounded memory. Measured
GPU/wall times, small-batch crossovers and saved log-log figures are in
[BENCHMARKS.md](BENCHMARKS.md#turn-14-unified-python-loop-cpu-and-metal-sweep).

```cpp
#include <batchfit/models.hpp>
double params[3] = {4.0, 1.0, 2.0}; // height, center, sigma
double jacobian[3];
double value = batchfit::models::Gaussian<double>::eval(3.0, params);
batchfit::models::Gaussian<double>::jacobian(3.0, params, jacobian);
```

Parameter ordering, width conventions and domains are specified in
[the model contract](DESIGN.md#shared-model-definitions).

## Python quick start

```sh
export PATH="$PWD/.venv/bin:$PATH"
uv pip install -e .
```

```python
import batchfit
result = batchfit.fit(x, y, model="lorentzian", p0=None, bounds=None, device="auto")
params, status, iterations, chi2 = result  # also result.params, result.status, ...
```

Request `uncertainties=True` for additional `result.covariance`, `result.stderr`
and `result.uncertainty_status` arrays. The default four-array result allocates
no uncertainty buffers; requested covariance is recomputed at the accepted
solution and scaled by reduced chi-squared. See NUMERICS.md for assumptions,
failure diagnostics and measured 10,000-realization coverage per model/device.

Use shared `x[M]`, fit-major `y[N,M]` (or a single `y[M]`). Outputs are NumPy
arrays `[N,P]`, `[N]`, `[N]`, `[N]`. CPU borrows suitable contiguous arrays and
preserves float64; auto uses Metal for float32 when available. Explicit Metal
uses float32 with the necessary point-major layout preparation, never silent
CPU fallback. Fitting releases the GIL; type stubs are shipped. See
[python/README.md](python/README.md) for full defaults, bounds, statuses and the
1,000-fits-per-model SciPy checks. `sh scripts/check_python.sh` verifies the
editable package and native binding diagnostics.

## Single-fit CPU reference

```cpp
#include <batchfit/batchfit.hpp>
std::array<double, 5> x{-2, -1, 0, 1, 2};
std::array<double, 5> y{};
std::array<double, 3> truth{2, 0.2, 0.7}, initial{1.5, 0, 1};
for (unsigned j = 0; j < x.size(); ++j)
    y[j] = batchfit::models::Gaussian<double>::eval(x[j], truth.data());
batchfit::cpu::SolverOptions<double, 3> options;
options.lower[0] = 0;
options.lower[2] = 0.01; // explicit positive-width constraint
auto result = batchfit::cpu::fit_single<double, batchfit::models::Gaussian>(x, y, initial, options);
// Inspect result.status before using result.parameters, iterations, chi_squared.
```

The model is a class template and the scalar type is explicit. Input x/y spans
must have equal positive length; initial parameters must match the model's fixed
count. Everything inside the call uses stack arrays (P <= 8), including J^T J and
Cholesky. Input buffers are borrowed read-only and returned parameters are owned
by a fixed-size array. No allocation, weights, batch scheduling or GPU fit dispatch
is hidden in this API.

Bounds default to unbounded. Initial and trial parameters are clamped; equal
finite bounds fix a coordinate. Set positive lower bounds for widths and [0,1]
for pseudo-Voigt mixing. The solver reports `FitStatus::converged`,
`max_iterations`, `singular_matrix` or `invalid_input`, plus spectral diagnostics
`degenerate_data` and `unresolved_peak`. A rejected/non-finite trial
never overwrites accepted parameters or chi-squared. The iteration count includes
attempted systems, even rejected ones; zero-budget calls return initial cost.
For exact damping, stopping/default-tolerance and failure semantics see
[the reference algorithm](DESIGN.md#levenbergmarquardt-reference-algorithm).

The CPU reference is the source of truth for later parallel/GPU solvers. Its
suite recovers 768 synthetic fits across every model/scalar/noise combination,
with noise-scaled parameter tolerances and deliberate status failures. A separate
global-new counter verifies zero C++ allocations across all eight model/scalar
calls, not only the inner iteration loop.

## Float32 numerical robustness

[NUMERICS.md](NUMERICS.md) records the float64-reference study across models,
signal-to-noise levels, coordinate/parameter ranges, kernels and safe/fast math.
Normalize **original double inputs before converting to float32** with
`batchfit::Coordinates::from(x_double,y_double)`. Use its `x()`, `y()` and
`normalize<Model>()` transforms to prepare the existing CPU/shared-batch APIs;
`physical<Model>()` reconstructs physical parameters in host double afterward.
It cannot restore coordinates already rounded to float32. Constraints/tolerances
must be specified in the fitted units; centered paired-baseline bounds can be
coupled, so they are not silently converted as independent boxes.

```cpp
batchfit::cpu::SolverOptions<float, 3> accurate;
accurate.compensated_sum = true;   // objective, J^T r and J^T J
accurate.scaled_cholesky = true;   // equivalent diagonally equilibrated solve
batchfit::metal::Context context;  // safe/precise math, as before
// Prepare normalized x/y/initial values, then pass accurate to fit_single/fit_batch.
```

The retained safe normalized recipe had maximum unit-parameter error **1.51e-4**
against float64 CPU across 122,880 GPU fit executions; raw poorly scaled cases
reached **0.291** despite converged status. Defaults stay reproducible; both
numerical flags are opt-in. `metal::Context(metal::MathMode::Fast)` is available
for explicit experiments, not the strict-accuracy recommendation. Nonfinite
shader guards use exponent bits and are tested in both compilation modes.
Flat/zero spectra with free shape parameters now return `degenerate_data` (4),
and converged sub-spacing FWHM fits return `unresolved_peak` (5). Edge peaks are
not automatically rejected. See NUMERICS.md for exact diagnostic conditions.

## Batched CPU fitting

```cpp
#include <batchfit/batchfit.hpp>
batchfit::cpu::ThreadPool pool(2); // reusable; no OpenMP or external pool dependency
// shared x[M], flat fit-major observations[N*M], initial[N*P]
auto results = batchfit::cpu::fit_batch<double, batchfit::models::Gaussian>(
    x, observations, initial_parameters, pool, options);
// results.parameters[N*P], status[N], iterations[N], chi_squared[N]
```

Pool workers are persistent `std::jthread`s, claiming 64-fit ranges. Each row calls
the unchanged allocation-free `fit_single` reference. Output ordering and per-fit
failure statuses are preserved regardless of scheduling; no invalid rows are
dropped. The call waits for every worker. Reuse the pool across batches to avoid
thread startup overhead. The normal default is two workers; zero is invalid.

N is inferred from `observations.size()/x.size()`. Wrong batch shapes throw
`std::invalid_argument`; bad data/options inside a well-shaped row return that
row's reference status. Zero fits return empty arrays with nonempty x. Input spans
are borrowed read-only and output vectors own their data. Automatic initial
guesses remain an explicit preparation step, not hidden inside the timed API.
Worker/model exceptions propagate to the caller after all workers drain, and the
pool remains reusable. Concurrent external submissions serialize; direct
same-pool recursion is rejected. Do not destroy a pool during a call or create
cyclic dependencies between pools/callbacks.

[BENCHMARKS.md](BENCHMARKS.md) records the required N=100,000 matrix for all four
models and M=32/128/512/2048 on one and all 10 Mac M4 cores. Three-run median
speedups are 3.91–5.30x. Peak process RSS was 997.03 MiB with one dataset resident.
Google Benchmark registrations and a shared-workload native runner are provided;
the recorded measurements are from the native runner because dependency downloads
remain blocked. The full matrix is opt-in; normal checks use a small two-worker
smoke case.

## Metal v1: shared point-major batches

Enable `BATCHFIT_METAL=ON` (the macOS default). The metal-cpp production host uses
`NS::SharedPtr`/autorelease pools and runtime source compilation; no offline Metal
compiler is needed. This path requires macOS 15+ for explicit safe/precise math.

```cpp
#include <batchfit/batchfit.hpp>
batchfit::metal::Context context; // compiles embedded MSL and caches all variant pipelines
auto batch = context.make_batch<batchfit::models::Gaussian>(N, M);
// Fill these CPU spans directly; their storage is MTLResourceStorageModeShared:
// batch.x()[m], batch.y()[m*N+i], batch.initial()[i*3+k].
batchfit::cpu::SolverOptions<float, 3> options;
options.lower[2] = 0.01f;
auto result = batchfit::fit_batch<batchfit::models::Gaussian>(
    batch, batchfit::Device::Metal, options);
// result.parameters[i*3+k], status[i], iterations[i], chi_squared[i]
// result.timings.gpu_seconds / wall_seconds; SIMD/group sizes are also reported.
```

Inputs/outputs live in shared Metal buffers; dispatch does not upload, transpose
or copy array data. **y is point-major**, unlike the fit-major CPU API. Prepare or
transpose it once before timing. Initial/output parameters remain fit-major.
`Device::Cpu` on this same shared-batch overload is an explicit single-thread
reference path that gathers rows; it is never a fallback for `Device::Metal`.
The existing pooled `cpu::fit_batch` is unchanged.

The call is synchronous and checks compilation, functions, pipelines, allocations,
encoder/command creation, command status/error and GPU timestamps. Embedded shared
headers retain file/line diagnostics through `#line`. Threadgroups are SIMD-width
multiples derived from the pipeline limits. A context serializes submissions;
callers must not mutate buffers or reuse one batch concurrently with a call.
Result spans retain their buffers/runtime after context/batch destruction, but
are **views, not snapshots**: reusing the batch overwrites earlier result views.
Empty batches are supported; dimensions must fit uint32 shader indexing and the
device's buffer length limits. CPU-only builds report unavailable Metal explicitly.

**Verification:** 2,832 per-fit CPU/GPU comparisons pass at parameter tolerance
`2e-4 + 2e-4*abs(CPU)` and cost tolerance `1e-6 + 2e-5*abs(CPU)`, including bounds,
failure statuses, SIMD tails, point-major indexing, lifetime and diagnostic tests.
Metal API/shader validation was enabled for native SDK checks. The original
download limitation was repaired during turn 9: the actual metal-cpp host now
compiles and passes the production CMake/GoogleTest suite. The test-only ARC
driver `tests/metal_runtime_native.mm` remains an independent SDK route, never a
production fallback. Pinned tools, both build presets, clang-tidy and benchmark
smokes now pass `scripts/check.sh`.

**Historical v1 numerical limit (original guesses):** the 1.6-million-fit benchmark matched every CPU
status/cost, but 12 rows exceed the stricter parameter diagnostic, and two large
paired rows also exceed the CPU reference's existing float numerical floor.
The latter converge to poor local minima in both backends. The full benchmark
validator deliberately exited nonzero; its tests were **not** relaxed and no CPU
fallback was added. This was not universal parameter equivalence.
[BENCHMARKS.md](BENCHMARKS.md) records GPU timestamp and wall-time fits/s, CPU
comparisons, mismatch counts and reproduction commands.

Turn 9 identified dense-spectrum shoulder noise selecting two initial components
on the same peak. The data-only separation rule is now a full measured width
rather than a half-width, with a regression test that fails under the old rule.
The new cooperative matrix keeps all fits and passes unchanged numerical budgets
on corrected seeds. A production v1 full-matrix rerun also now exits zero with
all CPU status/cost/reference-floor comparisons passing; nine stricter pseudo-Voigt
diagnostic outliers remain. Historical v1 measurements above retain their original inputs
and failure counts; this is not a claim of universal parameter identifiability.

## Cooperative Metal variants and automatic dispatch

`metal::KernelVariant` provides `ThreadPerFit` (v1), `SimdPerFit` (32 lanes per fit),
`ThreadgroupPerFit` (256 threads per fit on M4), and `Automatic` (the default).
Pass a forced variant after solver options to reproduce a comparison:

```cpp
auto result = batchfit::fit_batch<batchfit::models::Gaussian>(
    batch, batchfit::Device::Metal, options, batchfit::metal::KernelVariant::SimdPerFit);
// result.timings.variant reports the kernel actually used.
```

Lanes accumulate packed J^T J/J^T r terms and use `simd_sum`. One leader performs
Cholesky/LM decisions in threadgroup scratch; `simd_broadcast` distributes the
step. Whole-threadgroup mode combines per-SIMD partials through threadgroup
memory/barriers. v1 names and buffers remain available, and historical v1
benchmark helpers force v1 instead of silently switching algorithms.

The empirical M4 dispatcher is model/M-specific, including a pseudo-Voigt
SIMD-to-v1 reversal at M=2048. Its measured policy, all forced/automatic timings,
crossover brackets and 5% performance-noise rule are in
[DESIGN.md](DESIGN.md#cooperative-metal-v2-and-dispatch).
`profiling/cooperative_results.csv` records the production Release matrix at
N=16,384 and M=32,64,...,4096. Every variant has zero CPU status/parameter/cost
errors. The final automatic policy selected the exact fastest forced median in
29/32 cells and was within 4.01% in the three near-ties. This is a calibrated
large-batch, point-major M4 policy, not universal tuning for small N or other GPUs.

Turn 11's numerical guards shift Gaussian M=512 to v1 (SIMD starts at 1024).
All other buckets remain unchanged. The new 128-row matrix passes the same 5%
performance gate; the turn-9 table above remains historical. See NUMERICS.md.

## Out-of-core NPY fitting and three-slot pipelining

`metal::fit_npy<Model>` maps **windows**, never the complete observation file.
Inputs are immutable NPY v1.0, little-endian C-order `x[M]`, `y[N,M]`, and optional
`initial[N,P]`; float32 and float64 are accepted and explicitly converted to
float32 before fitting. No shader/double-precision solver is implied. Other
dtypes, versions, orders and ranks are rejected. Omitting `initial` uses the
existing data-only automatic guesses; failed guesses stay in the output as
`invalid_input`, and are counted rather than dropped.

```cpp
#include <batchfit/batchfit.hpp>
#include <iostream>

batchfit::metal::Context context;
auto stats = batchfit::metal::fit_npy<batchfit::models::Gaussian>(
    context, {"x.npy", "y.npy", {}},
    [](std::size_t first, const batchfit::metal::Result& chunk) {
        std::cout << first << ": " << chunk.status.size() << " results\n";
        // Write or copy the values here; these views are recycled after return.
    });
```

The producer maps/converts one fit-major chunk, prepares initials, and transposes
to point-major shared buffers while a second CPU worker consumes completed
results. Three reusable slots submit asynchronous command buffers; completion
handlers check errors/timestamps and signal CPU waiters. Every slot is drained
before reuse or failure propagation. This uses two CPU workers, not a full-core
pool. Results are delivered in input order, including a short final chunk.
**Do not retain callback views or mutate/truncate input files while fitting.**

`PipelineOptions{chunk_fits, in_flight, variant}` defaults to `{0,3,Automatic}`.
Zero selects a device-based chunk size: budget `min(recommendedWorkingSet/8,
256 MiB)` for the shared slots, conversion staging and mapped input (64 MiB when
no recommendation exists), also respecting `maxBufferLength` and shader uint32
index limits. Depth 1–3 is supported for controlled comparisons. An explicit
chunk overrides the automatic memory budget, but not device/index limits.
Planning budgets are not hard RSS limits; the Metal runtime and OS page cache
have their own overhead. `Context::limits()` exposes the queried limits.

```sh
build/metal/batchfit_fit_npy --model gaussian --x data/x.npy --y data/y.npy \
  --output .agent-local/fitted --chunk-fits 0 --in-flight 3 \
  --timeline profiling/my_pipeline.json
# Add --initial data/initial.npy for caller-provided initials.
```

The new output directory contains float32 `parameters.npy` / `chi_squared.npy`
and uint32 `status.npy` / `iterations.npy`. Status codes are 0 converged, 1 max
iterations, 2 singular matrix, 3 invalid input, 4 degenerate data, 5 unresolved
peak. Existing output/timeline paths
are refused; owned partial output is rolled back on failure. API consumers can
stream results elsewhere without allocating full-N output arrays. `submit_batch`
also exposes the asynchronous primitive directly: retain its `PendingBatch`,
call `get()` before reading/reusing buffers, and consume its result views before
resubmitting the same batch. Dropped handles synchronously drain; unobserved GPU
errors are reported, not silently discarded.

See [BENCHMARKS.md](BENCHMARKS.md#out-of-core-pipeline-turn-10) for the actual dense
10.24 GB run and memory/throughput, and [PROFILING.md](PROFILING.md#out-of-core-pipeline-turn-10)
for timestamp overlap and the unavailable Metal System Trace qualification.

## Metal profiling

For measured Metal group/layout/half-input experiments, capture files and exact
rerun commands, see [PROFILING.md](PROFILING.md). `BATCHFIT_GPU_CAPTURE=1` opts into
a checked first-dispatch `.gputrace` under `profiling/`; capture is off by default
and refuses existing files. Xcode-only occupancy/per-line metrics remain
unverified here; the report labels hypotheses separately from observations.

## Automatic initial guesses

```cpp
// With the x/y arrays above; the explicit-initial-parameter solver is unchanged.
auto guess = batchfit::cpu::initial_guess<double, batchfit::models::Gaussian>(x, y);
if (guess.status == batchfit::cpu::GuessStatus::success) {
    auto fit = batchfit::cpu::fit_single<double, batchfit::models::Gaussian>(
        x, y, guess.parameters, options);
    // Inspect fit.status as usual. guess.baseline is a diagnostic estimate.
}
```

`cpu::initial_guess<T, Model>` supports all four models in float/double on finite,
strictly increasing x grids with at least five points. It finds **positive emission
peaks**, estimates the baseline from medians of the outer 10% at each edge, derives
height above baseline, refines the peak position with a local centroid, and
linearly interpolates half-height crossings. FWHM is converted to Gaussian sigma,
Lorentzian HWHM, or retained as pseudo-Voigt FWHM (mixing starts at 0.5).

For paired peaks it seeds a linear baseline from the separate edge medians and
chooses the two highest separated/prominent candidates, ordered by center.
An overlapped inner half-width may be mirrored from the measured outer half;
outer half-height crossings must exist. Weak but resolved secondary peaks are
allowed; there is no fixed amplitude-ratio cutoff. See
[the heuristic contract](DESIGN.md#automatic-initial-guesses) for exact rules.

Check `GuessStatus`: `success`, `invalid_input`, `no_peak`, or `insufficient_peaks`.
No input is mutated. Guess preparation uses O(M) temporary scratch storage; the
LM call still uses only stack workspace. The baseline is used **only for feature
extraction**: the single-peak models still have no baseline parameter, and this API
does not silently subtract anything from y. Absorption dips, truncated outer
half-widths, unresolved blends and signals below the prominence threshold may
need caller-supplied initial parameters.

### Measured guess-to-fit convergence

The fixed-seed suite tests 500 fits per model/scalar/noise/SNR group: Gaussian
and Poisson noise, SNR 10/20/40, 129 points on [-5,5]. All **18,000 single-peak
cases passed** (100% for every model/type, and >99% in each individual group).
Initial guesses and fit bounds use no true parameters. Counts include failed
guesses; success additionally requires fitted signal RMS error <= twice the
expected noise RMS, not just a `converged` status.

Paired spectra use positive tilted baselines (intercept 0.3–0.5, slope -0.04–0.04)
and the generator's default peak ranges, including partially overlapping peaks.
Optimized Apple Clang 21 / Apple M4 measurements, **per scalar type**:

| Model | Trials | Valid guesses | LM converged | Signal recovered |
| --- | ---: | ---: | ---: | ---: |
| Gaussian (float or double) | 3,000 | 3,000 | 3,000 | 3,000 (100%) |
| Lorentzian (float or double) | 3,000 | 3,000 | 3,000 | 3,000 (100%) |
| Pseudo-Voigt (float or double) | 3,000 | 3,000 | 3,000 | 3,000 (100%) |
| Two Lorentzians (float or double) | 3,000 | 2,801 | 2,798 (93.267%) | 2,794 (93.133%) |

With the more conservative full-width separation rule, the paired end-to-end LM
convergence rate is **93.267%**; the stricter signal recovery rate is **93.133%**.
There are 199 failed paired guesses, three LM failures, and four converged fits
outside the signal-error threshold per type. Original turn-5 half-width-rule
measurements were 96.733%/96.667%; they accepted more ambiguous features. These are
measured test-distribution rates, not guarantees for arbitrary spectra or proof
of unique paired-parameter recovery. The same suite runs in CTest/GoogleTest and
as the standalone command below.

## Synthetic datasets

`batchfit_generate` streams datasets to a **new** output directory, refusing to
overwrite existing data. Build it with CMake (including a minimal CPU build), then:

```sh
build/cpu/batchfit_generate --output .agent-local/example-data --model pseudo_voigt --fits 10000 --points 64 --noise gaussian --snr 20 --seed 42 --chunk-fits 4096 --range amplitude=1:3 --range mixing=0.2:0.8
```

Models are `gaussian`, `lorentzian`, `pseudo_voigt`, and `two_lorentzian`. Noise
is `none`, `gaussian`, or `poisson`. `--range parameter=lower:upper` is repeatable;
equal bounds fix a parameter. `--x-min`/`--x-max` set a shared inclusive linear
grid (default -5 to 5). Run `--help` for all flags. SNR is a **linear per-fit RMS
ratio**, not decibels or peak height. Poisson output is scaled integer counts,
in signal units; it requires nonnegative noiseless samples. See the precise
[noise definitions](DESIGN.md#synthetic-data-and-npy).

Each dataset contains C-order little-endian float32 NPY v1.0 arrays:

- `x.npy`: `(M,)`
- `y.npy`: `(N, M)`, fit-major noisy observations
- `true_params.npy`: `(N, P)`, parameters in the documented model order

Model/seed/options are not inferred or serialized in these arrays; retain the
generation command/options with experiments. The reader deliberately supports
only this simple NPY subset (including ordinary NumPy-generated headers), not
object arrays, float64, Fortran order, or NPY v2/v3.

The same functionality is available without the CLI:

```cpp
#include <batchfit/synthetic.hpp>
batchfit::synthetic::Options options;
options.model = batchfit::synthetic::Model::two_lorentzian;
options.parameters = batchfit::synthetic::default_ranges(options.model);
options.fits = 10000;
batchfit::synthetic::save_dataset(".agent-local/paired-data", options);
batchfit::synthetic::load_dataset(".agent-local/paired-data", 1024,
    [](std::span<const float> x, const batchfit::synthetic::Chunk& chunk) {
        // Consume borrowed x, chunk.y and chunk.true_parameters here.
    });
// generate(options, consumer) streams directly to the same callback without disk.
```

Both generation and loading retain only one chunk plus x: O(chunk_fits*(M+P)+M)
memory, independent of N. A seed is repeatable with the same toolchain/build and
C++ standard-library implementation/version, byte-identical across chunk sizes.
Cross-toolchain bitwise identity of random distributions is not promised. Separate parameter/noise RNGs
keep ground truth unchanged when changing noise type or SNR.

### Million-fit measurement

Measured on Apple M4 / macOS 27.2 arm64 / Apple Clang 21, `-O3 -DNDEBUG`, one
worker, Gaussian model, N=1,000,000, M=64, seed 42, SNR 20, chunk size 4,096:

| Noise | Generation + write/close time | Fits/s | Maximum RSS (`time -l`) |
| --- | ---: | ---: | ---: |
| Gaussian | 0.459 s | 2,176,400 | 2.83 MiB |
| Poisson | 1.456 s | 686,874 | 2.81 MiB |

Each run wrote 268,000,640 bytes across the three NPY files. These are single-run
measurements including file close, **not fsync/durable-storage throughput**, and
not model-fitting performance claims. The million-fit Gaussian dataset was also
loaded through the library in 1,024-fit chunks (2.22 MiB maximum RSS). Generated
benchmark datasets were removed afterward. Reproduce with a release build:

```sh
cmake -S . -B build/generator-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DBATCHFIT_METAL=OFF -DBATCHFIT_BUILD_TESTS=OFF -DBATCHFIT_BUILD_BENCHMARKS=OFF
cmake --build build/generator-release --parallel 2
/usr/bin/time -l build/generator-release/batchfit_generate --output .agent-local/million-fits --model gaussian --fits 1000000 --points 64 --noise gaussian --snr 20 --seed 42 --chunk-fits 4096
# Inspect or consume the files, then delete these generated benchmark files/directory.
```

## Local setup

Requires a C++20 compiler and Python 3.9+ for native checks; the pinned Python
package/test environment supports Python 3.9–3.12. On macOS, Command Line Tools provide
Clang and framework headers; Metal executes on the default GPU. No offline shader
compiler is needed. Do not install or alter system tools from this repository.

Install uv **0.9.8** locally if it is not already supplied. For Apple Silicon,
with shell network access:

```sh
mkdir -p .tools .agent-local
curl -fL https://github.com/astral-sh/uv/releases/download/0.9.8/uv-aarch64-apple-darwin.tar.gz -o .agent-local/uv.tar.gz
tar -xzf .agent-local/uv.tar.gz -C .tools
export PATH="$PWD/.tools/uv-aarch64-apple-darwin:$PATH"
rm .agent-local/uv.tar.gz
sh scripts/bootstrap.sh
export PATH="$PWD/.venv/bin:$PATH"
```

Use the corresponding pinned uv archive on other platforms. Bootstrap creates
`.venv` when absent, preserves an existing one, and installs the editable package
plus exact development versions from `pyproject.toml`;
it does not download a Python interpreter. Preserve supplied project-local
HOME/TMPDIR/cache settings. Outside orchestration, set them under `.agent-local/`
before installing anything. CMake downloads GoogleTest/Benchmark in `build/` and
Apple's metal-cpp in `third_party/`; the CPU backend does not depend on metal-cpp.

```sh
cmake --preset cpu
cmake --build --preset cpu
ctest --preset cpu
# macOS: equivalently configure with -DBATCHFIT_METAL=ON (the macOS default)
cmake --preset metal
cmake --build --preset metal
ctest --preset metal
sh scripts/check.sh
```

Both presets cap build and test concurrency at two. For a minimal CPU-only
library build without downloading test/benchmark dependencies:

```sh
cmake -S . -B build/minimal -G Ninja -DBATCHFIT_METAL=OFF -DBATCHFIT_BUILD_TESTS=OFF -DBATCHFIT_BUILD_BENCHMARKS=OFF
cmake --build build/minimal --parallel 2
```

Formatting: `python3 tools/check_format.py`; apply fixes with
`clang-format -i <changed-source-files>`. Static analysis:
`python3 tools/run_tidy.py build/cpu` (also `build/metal` on macOS). Never format
or analyze downloaded third-party code as project source.

## Offline diagnostic check

Native installed-SDK verification when metal-cpp cannot be downloaded:

```sh
mkdir -p .agent-local/check/generated
python3 tools/embed_metal.py .agent-local/check/generated/lm_source.hpp
clang++ -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++20 -O2 -fobjc-arc -DBATCHFIT_NATIVE_METAL_TESTS=1 -Wall -Wextra -Wpedantic -Werror -Iinclude -Isrc/metal -I.agent-local/check/generated -x objective-c++ tests/metal_runtime_native.mm src/cpu/npy.cpp src/cpu/synthetic.cpp tests/metal_lm_check.cpp -framework Foundation -framework Metal -o .agent-local/check/metal_lm_check
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 .agent-local/check/metal_lm_check
rm .agent-local/check/metal_lm_check .agent-local/check/generated/lm_source.hpp
rmdir .agent-local/check/generated .agent-local/check
```

This verifies real MSL execution/public buffer semantics, not metal-cpp host
compilation or NS::SharedPtr lifetime instrumentation. See the full-gate limits below.

The batch/thread-pool suite needs no downloaded dependencies:

```sh
mkdir -p .agent-local/check
clang++ -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++20 -O2 -pthread -Wall -Wextra -Wpedantic -Werror -Iinclude src/cpu/thread_pool.cpp src/cpu/npy.cpp src/cpu/synthetic.cpp tests/batch_check.cpp -o .agent-local/check/batch_check
.agent-local/check/batch_check
clang++ -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++20 -O2 -pthread -Wall -Wextra -Wpedantic -Werror -Iinclude src/cpu/thread_pool.cpp tests/batch_failure_check.cpp -o .agent-local/check/batch_failure_check
.agent-local/check/batch_failure_check
rm .agent-local/check/batch_check .agent-local/check/batch_failure_check
rmdir .agent-local/check
```

Pool/batch checks pass ASan/UBSan and ThreadSanitizer, as well as normal native
compilation. The suite verifies actual parallelism, exactly-once tail coverage,
reuse, cancellation/exception propagation, concurrent submissions, zero/invalid
shapes, and bitwise single-fit equivalence across all models/scalars. The separate
failure executable covers mixed singular/converged/invalid rows and thrown model
exceptions. CTest reserves two job slots for pool tests to prevent oversubscription.

Run automatic-guess feature checks and the complete 24,000-case convergence study:

```sh
mkdir -p .agent-local/check
clang++ -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -Iinclude src/cpu/npy.cpp src/cpu/synthetic.cpp tests/initial_guess_check.cpp -o .agent-local/check/initial_guess_check
.agent-local/check/initial_guess_check
rm .agent-local/check/initial_guess_check
rmdir .agent-local/check
```

The complete study and feature checks also pass in Debug and ASan/UBSan builds;
all three builds produced the paired counts above on this toolchain. These native
checks do not substitute for the full CMake/GoogleTest/clang-tidy gate (verified
after the turn-9 dependency repair).

The full single-fit suite and allocation instrumentation need only the existing
C++ compiler/SDK, without CMake/GoogleTest or downloaded headers:

```sh
mkdir -p .agent-local/check
clang++ -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -Iinclude src/cpu/npy.cpp src/cpu/synthetic.cpp tests/solver_check.cpp -o .agent-local/check/solver_check
.agent-local/check/solver_check
clang++ -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/solver_allocation_check.cpp -o .agent-local/check/solver_allocation_check
.agent-local/check/solver_allocation_check
rm .agent-local/check/solver_check .agent-local/check/solver_allocation_check
rmdir .agent-local/check
```

Both are CTest targets; the recovery/status suite is also wrapped by GoogleTest.
The direct Clang suite passes with Debug and optimized builds and ASan/UBSan;
the allocation counter passes in both Debug and optimized builds. The public
API example above also compiles and recovers its noiseless parameters. These
checks do not imply a successful CMake/GoogleTest/clang-tidy gate when those
prerequisites are unavailable (see environment limitations below).

When installers cannot reach their hosts, the existing system Clang can still
exercise the CPU library and public header without CMake or external frameworks:

```sh
mkdir -p .agent-local/check
clang++ -std=c++20 -Wall -Wextra -Wpedantic -Werror -DBATCHFIT_HAS_METAL=0 -Iinclude -Isrc src/cpu/reference.cpp tests/cpu_smoke.cpp -o .agent-local/check/cpu_smoke
.agent-local/check/cpu_smoke
rm .agent-local/check/cpu_smoke
rmdir .agent-local/check
```

The complete float/double model checks can also run without external packages:

```sh
mkdir -p .agent-local/check
clang++ -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude tests/models_check.cpp -o .agent-local/check/models_check
.agent-local/check/models_check
# macOS: validate actual MSL compilation and compare GPU output to CPU references
clang++ -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++20 -fobjc-arc -Wall -Wextra -Wpedantic -Werror -Iinclude tests/models_metal.mm -framework Foundation -framework Metal -o .agent-local/check/models_metal
.agent-local/check/models_metal "$PWD"
rm .agent-local/check/models_check .agent-local/check/models_metal
rmdir .agent-local/check
```

Synthetic library/CLI checks also have a dependency-free path:

```sh
mkdir -p .agent-local/check
clang++ -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -Iinclude src/cpu/npy.cpp src/cpu/synthetic.cpp tests/synthetic_check.cpp -o .agent-local/check/synthetic_check
.agent-local/check/synthetic_check
clang++ -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -std=c++20 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Werror -Iinclude src/cpu/npy.cpp src/cpu/synthetic.cpp tools/generate.cpp -o .agent-local/check/batchfit_generate
python3 tools/check_generator.py .agent-local/check/batchfit_generate
rm .agent-local/check/synthetic_check .agent-local/check/batchfit_generate
rmdir .agent-local/check
```

The synthetic suite checks Gaussian/Poisson moments and RMS SNR over 660,000
observations each, seed/chunk invariance, all 12 model/noise roundtrips, exact
independent NPY header/payload fixtures, and invalid inputs/rollback. It also
passes ASan/UBSan. The Python CLI check uses only `ast`/`struct` and other stdlib
modules; no NumPy package was installed or used for these checks.

Omit `-isysroot ...` on non-macOS systems and do not run the Metal commands there.
Turn 2 verified all 72,000 float/double model derivative comparisons, hand values,
and 1,028 real GPU model/Jacobian cases. The CPU model suite also passed with
AddressSanitizer and UndefinedBehaviorSanitizer, and the earlier CPU smoke still
passes. These standalone checks are also CTest
targets; the float/double suites additionally run through GoogleTest. Their
success does not remove the full-gate limitations below.

Historical initial implementation environment: shell DNS
failed for GitHub, PyPI and developer.apple.com; uv/CMake/Ninja/clang-tidy were
absent. Consequently dependency installation, uv environment creation, CMake,
GoogleTest/Benchmark and the original metal-cpp backend could not be verified. Re-run the full
gate after connectivity and prerequisites are available; do not bypass isolation.
Turn 9 restored project-local dependency installation and verified the production
CPU/metal-cpp builds, GoogleTest/CTest, clang-tidy and bounded Google Benchmark
smokes through a successful `scripts/check.sh`. clang-tidy is pinned to 21.1.6
to parse the current Apple SDK; it uses the actual build compiler's resource
headers. ABI-width/style exceptions are documented in `.clang-tidy`; numerical
tests and solver tolerances were not relaxed.

The offline CPU smoke did pass with Clang 21 after explicitly supplying
`-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`; the bare compiler
did not discover its SDK headers in this environment. This flag reads the
existing system SDK and does not install or modify it. Formatting was checked
with the available Apple clang-format 21, not the pinned development version 18.
The CPU smoke also passed with AddressSanitizer and UndefinedBehaviorSanitizer.
Indexed diagnostics remained partial: no CMake compile database could be generated
and the Python diagnostic server was unavailable. Direct compiler and script
syntax checks, not that partial diagnostic snapshot, establish the subset results.
