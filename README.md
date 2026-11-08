# batchfit

C++20 infrastructure for fitting batches of small independent nonlinear models
on CPU and Apple Metal. See [DESIGN.md](DESIGN.md) for the fitting API and solver
contract. Shared Gaussian, Lorentzian, pseudo-Voigt and paired-Lorentzian models
compile as both C++ and MSL. A single-fit CPU LM solver is available in float and
double; batch/GPU fitting is not implemented yet.

```cpp
#include <batchfit/models.hpp>
double params[3] = {4.0, 1.0, 2.0}; // height, center, sigma
double jacobian[3];
double value = batchfit::models::Gaussian<double>::eval(3.0, params);
batchfit::models::Gaussian<double>::jacobian(3.0, params, jacobian);
```

Parameter ordering, width conventions and domains are specified in
[the model contract](DESIGN.md#shared-model-definitions).

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
`max_iterations`, `singular_matrix` or `invalid_input`. A rejected/non-finite trial
never overwrites accepted parameters or chi-squared. The iteration count includes
attempted systems, even rejected ones; zero-budget calls return initial cost.
For exact damping, stopping/default-tolerance and failure semantics see
[the reference algorithm](DESIGN.md#levenbergmarquardt-reference-algorithm).

The CPU reference is the source of truth for later parallel/GPU solvers. Its
suite recovers 768 synthetic fits across every model/scalar/noise combination,
with noise-scaled parameter tolerances and deliberate status failures. A separate
global-new counter verifies zero C++ allocations across all eight model/scalar
calls, not only the inner iteration loop.

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
| Two Lorentzians (float or double) | 3,000 | 2,903 | 2,902 (96.733%) | 2,900 (96.667%) |

Thus the paired end-to-end LM convergence rate is **96.733%**; the stricter signal
recovery rate is **96.667%**. There were 97 failed paired guesses, one LM failure,
and two converged fits outside the signal-error threshold per type. These are
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

Requires a C++20 compiler and Python 3.9+. On macOS, Command Line Tools provide
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
`.venv` using uv and installs exact development versions from `pyproject.toml`;
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
checks do not substitute for an unavailable full CMake/GoogleTest/clang-tidy gate.

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

This is **not** a full test pass. Initial implementation environment: shell DNS
failed for GitHub, PyPI and developer.apple.com; uv/CMake/Ninja/clang-tidy were
absent. Consequently dependency installation, uv environment creation, CMake,
GoogleTest/Benchmark and the original metal-cpp backend could not be verified. Re-run the full
gate after connectivity and prerequisites are available; do not bypass isolation.

The offline CPU smoke did pass with Clang 21 after explicitly supplying
`-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`; the bare compiler
did not discover its SDK headers in this environment. This flag reads the
existing system SDK and does not install or modify it. Formatting was checked
with the available Apple clang-format 21, not the pinned development version 18.
The CPU smoke also passed with AddressSanitizer and UndefinedBehaviorSanitizer.
Indexed diagnostics remained partial: no CMake compile database could be generated
and the Python diagnostic server was unavailable. Direct compiler and script
syntax checks, not that partial diagnostic snapshot, establish the subset results.
