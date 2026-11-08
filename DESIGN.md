# batchfit design

## Scope and workload

batchfit is a C++20 library for **N independent nonlinear least-squares fits**,
typically millions of spectra, each with M samples and P model parameters,
where 1 <= P <= 8. A hyperspectral image supplies one spectrum per pixel. Fits
share a model and usually an x grid, but have independent observations, initial
parameters, convergence decisions and outputs. There is no coupling between fits.

The implementation provides build/test infrastructure, an internal squared-residual
compute smoke primitive on CPU and Metal, shared CPU/MSL model definitions,
chunked synthetic generation/NPY I/O, a single-fit CPU LM reference, and data-only
automatic initial guesses and batched CPU fitting in float and double. It does
**not** implement GPU fitting, a runtime model registry or Python bindings yet.
The backend-neutral API below remains a later contract; the implemented typed
CPU APIs are described in the solver and batched-path sections.

## Public API contract

The intended synchronous entry point is:

```cpp
BatchResult fit_batch(Model model, span<const float> x,
                      span<const float> y, span<const float> init,
                      const Options& options);
```

`Model` identifies a built-in model with a fixed P and an analytic Jacobian;
arbitrary C++ callbacks cannot be passed to a shader. Future custom models need
both CPU evaluation and explicitly compiled Metal code. The first API uses a
shared `x[M]`, `y[N*M]` and `init[N*P]`, with N and M in options. Input spans are
borrowed only for the duration of the call. Results own their memory. The default
backend is CPU; explicitly selecting Metal must not silently fall back to CPU.

Options will specify backend, dimensions, maximum iterations, initial damping,
step/gradient/objective tolerances and a bounded CPU worker count (two by default
in this repository). The initial external layout is fit-major and contiguous.
Reject zero M, P outside 1..8, incompatible lengths, integer size overflow and
invalid solver options before allocating or dispatching. N=0 produces empty
results. Non-finite data or parameters are a per-fit invalid-input status, not a
batch-wide crash. No fit may corrupt or suppress the status of another fit.

`BatchResult` contains four separate arrays:

| Field | Shape | Meaning |
| --- | --- | --- |
| parameters | N * P | Final accepted parameters in fit-major order |
| status | N | converged, max_iterations, singular_matrix or invalid_input |
| iterations | N | Number of attempted LM updates (zero before any update) |
| chi_squared | N | Sum of squared residuals at the final accepted parameters |

The initial objective is unweighted `sum_j (y_j - f(x_j, p))^2`; it is not reduced
chi-squared and is not divided by degrees of freedom. Weighting is not implicit.
For invalid fits, chi-squared is NaN and parameters retain the initial values.
Batch-level errors (invalid dimensions, unavailable device, allocation or Metal
execution failures) throw exceptions with operation-specific context. Device
errors are not numerical fit statuses. Outputs are available only after GPU
completion and validation of command-buffer status.

## Shared model definitions

`include/batchfit/models.hpp` is the single source for CPU and MSL evaluation and
analytic Jacobians. The empty, standard-layout structs are templated on scalar
type: `models::Gaussian<T>`, `Lorentzian<T>`, `PseudoVoigt<T>` and
`TwoLorentzian<T>`. C++ supports float and double; shaders instantiate float only.
Each has an integral `parameter_count`, `parameter_name(index)`,
`eval(x, params)` and `jacobian(x, params, out)`. Jacobians are **df/dp**, not
residual derivatives, in parameter order. Caller-owned arrays have at least
`parameter_count` elements; the functions allocate nothing and do not mutate
parameters. Copy device/constant parameters into a thread-local array before
calling them in MSL, and supply a separate thread-local Jacobian output.

All amplitudes are signed **peak heights**, not areas. Let `d = x - center`:

| Model | Parameter order | Evaluation |
| --- | --- | --- |
| Gaussian | amplitude, center, sigma | `amplitude * exp(-d*d/(2*sigma*sigma))` |
| Lorentzian | amplitude, center, hwhm | `amplitude / (1 + d*d/(hwhm*hwhm))` |
| PseudoVoigt | amplitude, center, fwhm, mixing | `amplitude * ((1-mixing)*G + mixing*L)`, with `G = exp(-4*ln(2)*d*d/(fwhm*fwhm))`, `L = 1/(1+4*d*d/(fwhm*fwhm))` |
| TwoLorentzian | amplitude1, center1, hwhm1, amplitude2, center2, hwhm2, intercept, slope | Two independent Lorentzian peaks plus **one** shared `intercept + slope*x` |

Widths must be finite and strictly positive. `mixing` is the Lorentzian fraction,
normally in [0,1]; both pseudo-Voigt components have the same FWHM, so any mixture
has half height at `center +/- fwhm/2`. The other models have no baseline. Paired
peaks need not be sorted or separated, but coincident peaks can be unidentifiable
for a later solver. These arithmetic primitives do not clamp parameters, throw,
or impose bounds. The eventual fitting input/step validation owns those checks.

Metadata is a shared `ParameterName` enum, avoiding string literals and pointer
tables in shaders. On the host, `parameter_name_text(Model::parameter_name(k))`
returns the canonical string from the table above. Out-of-range indices return
`unknown`. There are no C/C++ standard library includes, containers or storage
objects in the shared header. Small macros select MSL `thread`, `metal::exp` and
float constants versus host Clang/GCC math intrinsics and double constants; all
macros are undefined after the header. Only the shader branch includes
`metal_math`. This intentionally targets the repository's Clang/GCC toolchains.

Validation uses the same checks through GoogleTest and a standalone executable:
2,000 fixed-seed parameter sets per model per scalar type, totaling 16,000 sets
and 72,000 central finite-difference derivative comparisons. It covers signed
and zero amplitudes, peak centers, mixing endpoints, nonzero baselines, output
canaries and parameter-name ordering. Steps scale with peak width for positions
and widths, and with parameter magnitude otherwise; the actual representable
perturbation is used in the denominator. Float comparisons use `3e-4 + 2e-3*|fd|`,
double comparisons `2e-8 + 2e-7*|fd|`, accounting for finite-difference roundoff
and truncation rather than solver accuracy. Hand-computed centers, half-widths,
off-center values and pure-mixture limits independently check model conventions.

`tests/models_check.metal` includes this header verbatim. The test-only native
Objective-C++ harness `tests/models_metal.mm` expands that include for runtime
compilation (Metal has no runtime filesystem include-path option), checks shader
compilation/pipeline/command errors, and compares every result for 257 parameter
sets per model against CPU double evaluation/Jacobians and CPU float Jacobians.
Tolerance is `2e-5 + 3e-5*|reference|`. It uses safe math and ARC, and does not
replace or change the metal-cpp production backend. Using installed SDK interfaces
here provides an independent, dependency-free MSL compiler oracle even when
metal-cpp/GoogleTest downloads are unavailable. The Metal preset runs it via CTest;
a missing GPU is an error, never a skip.

## Automatic initial guesses

`cpu::initial_guess<T, Model>(x,y)` in `include/batchfit/cpu/initial_guess.hpp`
returns a fixed-size parameter array, diagnostic `baseline`, and `GuessStatus`.
It is a separate, opt-in preparation step: existing explicit-init solver calls
and invalid-empty-init behavior are unchanged. It uses observations only, never
true parameters, model fitting, restarts or a known noise/SNR value. Inputs must
have matching lengths >= 5, finite values, and a strictly increasing finite-span
x grid; nonuniform grids are supported through physical-x interpolation.

Use `max(1,floor(M/10))` points at each edge. The reported baseline is the median
of the pooled edge samples (even medians average the central two values). For
single-peak models subtract that constant **only in a scratch feature profile**.
For paired peaks, connect the separate left/right median levels at the median x
of each edge group, seeding intercept and slope and detrending the feature profile
with that line. Long peak tails can bias these estimates; LM refines them where
the model has baseline parameters. The Gaussian, Lorentzian and pseudo-Voigt
parameterizations deliberately retain their original no-baseline definitions.

Candidate locations use a three-point [1/4,1/2,1/4] smoothed profile to reduce
noise maxima. The primary candidate must exceed three times an edge-noise scale:
1.4826 times the median absolute deviation after centering each edge group at
its own median. Refine its raw index to the highest raw sample within one bin,
take that baseline-corrected raw height as amplitude, and locate the nearest raw
half-height crossings by linear interpolation in x. The center is a positive
intensity-weighted local centroid over at most two bins on either side, clipped
to the above-half-height samples when crossings are available. A stable online
weighted mean avoids summing large absolute x coordinates. Convert measured
FWHM to sigma by dividing by 2.354820045..., to Lorentzian HWHM by dividing by 2,
or use the common FWHM directly for pseudo-Voigt with neutral mixing 0.5.

For two peaks choose the next-highest smoothed local maximum at least three raw
bins from the first and at physical separation >= max(3*minimum_grid_spacing,
smaller_primary_half_width). Its prominence must exceed three edge-noise scales;
prominence is height minus the higher of the two side minima, stopping a side
search at a higher candidate or the boundary. This rejects noise bumps on the
primary's wing without imposing an arbitrary relative-amplitude floor. Order the
two components by refined center. Split their width searches at the interpeak
valley. If overlap censors an inner half-height crossing, mirror the measured
outer half-width; never fabricate an outer crossing. This is a symmetric-peak
approximation for initialization, not deconvolution or a resolution guarantee.

Return `no_peak` if there is no strong bracketed positive primary,
`insufficient_peaks` if a primary exists but a second usable separated feature
cannot be formed, and `invalid_input` for invalid input or unrepresentable output.
Only `success` parameters may be passed to LM. Negative absorption features,
clipped outer half-widths, weak unresolved blends and severe background curvature
are outside this heuristic's guarantees. Guessing allocates two scratch vectors
of M and 2*edge_count scalars, independent of the number of fits; this does not
change the allocation-free LM iteration contract. Returned arrays own their data,
and x/y are never modified.

Verification combines independent median/centroid/interpolated-width checks,
nonuniform/translated axes, weak 100:1 resolved pairs, strongest-two-of-three
selection, tilted baselines, missing crossings and invalid/flat/absorption data.
The convergence study runs **24,000** noisy guess-to-fit cases (12,000 fixed
spectra exercised in both scalar types): 500 per model/scalar/noise/SNR
combination, four models, float/double, Gaussian/Poisson noise and SNR
10/20/40, with 129 points on [-5,5]. Seeds are
`20261107 + 137*model_id + 31*SNR + 11*noise_id` using the synthetic enum values.
Peak ranges are the generator defaults; paired intercepts are 0.3–0.5 and slopes
-0.04–0.04 to exercise linear backgrounds while keeping Poisson means positive.
LM uses its unchanged default 100-iteration budget, nonnegative amplitudes,
centers inside the observed x range, widths in [grid_spacing/8, x_span] and mixing
in [0,1]. Paired centers are **not** given truth-derived disjoint boxes.

Every generated fit remains in the denominator, including failed guesses. The
strict success metric requires both LM `converged` status and reconstructed
noiseless signal RMS error <= 2*signal_RMS/SNR. Truth is accessed only after
guessing/fitting to score that metric, not to seed or constrain individual fits.
Every single-peak model/type/noise/SNR group must exceed 99%, not merely a pooled
average; 18,000/18,000 passed. Paired rates are measured rather than forced to
that threshold: per scalar type, 2,903/3,000 guesses succeeded, 2,902 converged
(96.733%) and 2,900 met the signal test (96.667%) in the optimized run. Signal
reconstruction is not a claim of unique parameters in overlapping models. README
records the observed rates and reproducible commands; floating-point/toolchain
changes can alter individual borderline outcomes.

## Synthetic data and NPY

`synthetic::generate(Options, Consumer)` samples model parameters uniformly from
finite configured ranges, evaluates the shared float model, and delivers borrowed
fit-major chunks. Equal range endpoints fix parameters. Configure the shared x
grid, N, M, model, ranges, noise, SNR, seed and chunk size; N/chunk size must be
positive and M >= 2. Widths must stay positive, and pseudo-Voigt mixing must stay
in [0,1]. Defaults are supplied by `default_ranges(model)`; library callers must
set these when selecting a non-default model. No parallel scheduling is required
for generation. Independent parameter/noise RNG streams use the fixed seed and
persist across chunks: chunk size cannot change any output value. Distribution
algorithms are those of the C++ library, so reproducibility is guaranteed for the
same toolchain/build and implementation/version, not bitwise across different
standard libraries.

For noiseless samples `s[j]`, let `q = sum(s[j]^2)`, `u = sum(s[j])`, and S denote
the requested **linear RMS SNR**. Gaussian noise is independent with
`sigma = sqrt(q/M)/S`. For Poisson noise require every s[j] >= 0, set
`gain = S^2*u/q`, draw `k[j] ~ Poisson(gain*s[j])`, and return `k[j]/gain`.
Thus both noises have expected per-fit mean squared error `q/(M*S^2)`; Poisson
noise is heteroscedastic with variance `s[j]/gain`, not additive Gaussian noise
masquerading as shot noise. All-zero signals produce zeros. Zero-mean Poisson
samples are exactly zero without invoking a distribution with invalid mean.
Reject non-finite results, unrepresentable scaling or expected counts > 1e12
instead of clipping. Noise-free mode evaluates the signal directly.

`save_dataset(directory, options)` writes `x.npy` (M), `y.npy` (N,M), and
`true_params.npy` (N,P). It exclusively creates the output directory, refuses
existing directories and rolls back only its own output on failure. Its parent
must already exist. `load_dataset(directory, chunk_fits, consumer)` validates file
sizes/shapes before streaming chunks. Neither path retains all N fits in memory;
both use O(chunk_fits*(M+P)+M) floats. Metadata such as selected model, ranges, seed
and SNR is caller-owned experiment configuration, not guessed from array shapes.

The local `npy::Writer`/`Reader` implement the simple NPY v1.0 envelope: magic and
version bytes, uint16 little-endian header length, newline/space-padded dictionary,
and a contiguous little-endian IEEE float32 payload. The writer aligns to 64
bytes; the reader accepts the format's 16-byte minimum alignment. Support is
deliberately limited to rank 1/2, `<f4`, C order and NumPy's alphabetical dictionary
key order. Reject unsupported dtype/order/version/rank, malformed/oversized headers,
truncation, trailing payload bytes, integer overflow and shape mismatches. Never
evaluate Python expressions. Generic Writer callers must append the full declared
element count and call `finish()` to check flush/close; Reader accepts partial
reads without allocating the full array.

Tests use fixed-seed 660,000-observation samples per noise model to check bias,
standardized variance and expected RMS SNR, plus the Gaussian fourth moment and
Poisson count quantization. All model/noise combinations round-trip across
different generation/load chunks, preserving parameters and values exactly.
Independent byte fixtures and Python stdlib `ast`/`struct` inspection check NPY
interoperability separately from the C++ writer/reader. See README for actual
million-fit timing, memory, output size and reproducible commands; benchmark
datasets are never committed or retained after measurement.

## Memory layout and mapping

An array of structs (AoS) groups each fit's parameters/status/objective together.
It is convenient for CPU per-fit execution but causes strided GPU loads when
adjacent threads load the same field from different fits. A struct of arrays
(SoA), such as `parameters[k*N + fit]`, makes those accesses contiguous and
keeps scalar status and iteration buffers separate. It may require a transpose
at the API boundary; that cost must be included in end-to-end benchmarks.

For measurements, **fit-major** is `y[fit*M + point]`: all samples of one fit are
contiguous. CPU loops use this naturally. It also suits one SIMD group per fit,
where lanes read consecutive points and cooperate on Jacobian reductions.
**Point-major** is `y[point*N + fit]`: neighboring fits at a fixed point are
contiguous. It suits one GPU thread per fit with a serial point loop, because
neighboring lanes read neighboring addresses. Fit-major data with one thread per
fit instead produces M-strided loads. A shared x grid is small, read-only and
reused; per-fit x grids would need an explicit later contract, not broadcasting
guessed from input lengths.

Start with the simplest independently correct per-fit kernel, then measure both
mappings. Do not decide the winning layout from theory alone: M, P, divergence,
register pressure, small-matrix storage, transposition and transfers matter.
Chunk large N to bound shared buffers and scratch storage. Keep N/M/P and offset
arithmetic checked on the host; shader index width is explicit. No per-fit dynamic
allocations are allowed in a kernel.

## Levenberg–Marquardt reference algorithm

`cpu::fit_single<T, Model>(x, y, initial, options)` in
`include/batchfit/cpu/solver.hpp` is the implemented numerical reference. T is
float or double; Model is a class template such as `models::Gaussian` supplying
the shared eval/Jacobian interface and a compile-time P in 1..8. The header-only
implementation is necessary for model/scalar instantiation. All parameters,
Jacobians, gradients, matrices and factorization workspaces use fixed-size stack
arrays. No input copies proportional to M, heap storage, callbacks or dynamic
linear algebra dependencies are used in the fit call.

For each accepted point, compute residual `r = y - f(x,p)`, model Jacobian
`J[j,k] = df(x_j,p)/dp_k`, `A = J^T J`, `g = J^T r` and unweighted objective
`chi_squared = sum(r*r)`. Arithmetic and accumulations use T. Solve the free
parameter system by in-place, non-pivoted Cholesky with forward/back substitution:

```text
(A + lambda * diag(A)) delta = g
p_trial = clamp(p + delta, lower, upper)
```

There is deliberately **no diagonal floor**: a zero diagonal for an uninformed
free parameter cannot be rescued by this damping rule. Cholesky failure (a
nonpositive/non-finite pivot or non-finite solution) returns `singular_matrix`.
This is not an undamped Jacobian rank diagnostic; positive damping can make a
rank-deficient normal matrix solvable. A zero projected gradient can converge
before a solve, even if that problem has unidentifiable parameters.

Bounds default to +/- infinity. Finite initial parameters are projected before
any model evaluation; finite lower==upper fixes a parameter. Exclude fixed
coordinates and coordinates at a bound whose g points outward: zero their
cross-terms, use a unit diagonal and zero right-hand side. This lets the remaining
free parameters solve correctly, including when a fixed parameter has a zero
Jacobian column. Recompute the active set each iteration. Width positivity and
mixing bounds are caller-owned options, not hidden model-name-dependent behavior.

Use the **clamped** displacement d to predict reduction
`pred = 2*g^T*d - d^T*A*d`. Accept only finite trials with strictly positive actual
and predicted reductions. For `rho = actual/pred`, update
`lambda *= max(1/3, 1-(2*rho-1)^3)` (rho >= 1 uses 1/3 directly). Rejection,
including non-finite trial evaluation, leaves accepted parameters/cost unchanged
and multiplies lambda by 10. Clamp accepted damping to
`[epsilon(T), max(initial_lambda,1/epsilon(T))]`; cap rejection growth at that same
upper limit. No inverse or fallback solver is formed.

Stop on any one of these explicitly defined criteria:

- **Gradient:** infinity norm of g over feasible/free directions <= gtol.
- **Step:** infinity norm of the raw solved delta <=
  `xtol * (norm_inf(p) + xtol) / max(1,lambda)`. This check precedes trial
  evaluation and returns the last accepted point. Compensating large damping
  avoids manufacturing convergence just by shrinking the step artificially.
- **Relative cost:** both `abs(actual) <= ftol*cost` and
  `0 <= pred <= ftol*cost/max(1,lambda)`, with finite trial cost/prediction. This
  also recognizes objective-rounding plateaus in float: a trial may be rejected
  yet demonstrate negligible actual **and model-predicted** progress. Return the
  best accepted point; never accept a worse cost merely to stop.
- Otherwise return `max_iterations` when the attempted-system budget is exhausted.

Default maximum iterations is 100 and initial lambda is 1e-3. Float defaults are
gtol=1e-5, xtol=1e-6, ftol=1e-6; double defaults are 1e-10, 1e-10, 1e-12. Tolerances
must be finite/nonnegative and lambda finite/positive. These are numerical stop
criteria, not promises of a global minimum or statistical goodness of fit.

The fixed-size result contains parameters, `FitStatus`, attempted iteration count
and final chi-squared. Initial gradient convergence/all-fixed fits take zero
iterations; each attempted factorization, rejection or step stop counts once.
A zero iteration budget evaluates the projected initial objective and returns
`max_iterations`. Reject empty/mismatched data, wrong init length, non-finite
x/y/init, invalid bounds/options, and non-representable initial model/Jacobian/
normal equations as `invalid_input`. Invalid preflight returns NaN chi-squared;
if only linearization failed, the last finite objective remains available.
With a wrong init length, all output parameters are NaN; otherwise invalid
preflight retains initial values. Valid finite initial values are projected only
after input validation. Failures after starting retain the last accepted point.

Verification runs 32 fits per model/noise/scalar combination: **768 recovery
cases** across four models, noiseless/Gaussian/Poisson observations and float/
double. Noiseless observations use exact scalar evaluation; noisy float32 data
comes from the streaming generator at SNR 80 (129 points, seed 20261107). Initial
guesses are deterministically perturbed around truth; positive widths, mixing
bounds and disjoint paired-center bounds remove invalid domains/label symmetry.
This tests local recovery, not global search. Parameter tolerances use an
eight-standard-error local uncertainty allowance plus a small numerical floor.
The numerical floor is `(1+abs(truth)) * 5e-4` for float and `* 1e-8` for double.
Gaussian errors use `sigma^2*(J^T*J)^-1`; Poisson errors use the heteroscedastic
OLS sandwich covariance with per-point variance `signal/gain`. This is a test
accuracy scale, not a covariance feature exposed by the solver. Tests independently
recompute final cost and preserve inputs, isolate each stopping criterion, exercise
initial/trial clamping and mixed active/free/fixed coordinates, verify Cholesky
against a known SPD solution, and deliberately trigger every failure status.
An exponential fixture rejects an overshoot then recovers with adjusted damping;
a high-damping fixture ensures tiny artificial progress cannot masquerade as
convergence. A separate executable instruments C++ new/new[] across all eight
built-in model/scalar fit calls and verifies zero allocations for the whole call.

## Batched CPU path

`cpu::fit_batch<T, Model>(x, y, initial, pool, options)` borrows shared x[M],
fit-major y[N*M] and initial[N*P], returning owning vectors for parameters, status,
iterations and chi-squared in original fit order. Infer N from y/M and validate
shapes with division/remainders rather than overflowing products. Nonempty x is
required even for N=0. A well-shaped empty batch returns empty arrays. Shape
errors throw; numerical input/options failures retain each `fit_single` status,
including shared bad x affecting every row. Bounds/options are shared across fits.
No baseline preprocessing, auto-initialization, dtype/layout conversion or
reduction across fits is hidden in this operation.

`cpu::ThreadPool` defaults to two persistent standard C++20 `std::jthread`
workers. A synchronous `parallel_for` publishes one range job under a mutex and
condition variable, with an atomic 64-fit chunk cursor for load balancing. Its
stop-aware condition-variable waits allow idle/partially constructed pools to
join safely. Concurrent external submissions serialize under a submission mutex;
same-pool worker recursion throws instead of deadlocking. Lifetime remains
caller-owned: the pool must outlive all calls, and cross-pool dependency cycles
or destruction from a callback are not supported.

Workers write disjoint output slices and call the unchanged single-fit reference,
with no heap allocation inside LM. Output vectors and one job closure are allocated
outside that loop. A completion counter/mutex establishes the join before results
are returned. The first worker exception is saved, unscheduled chunks are cancelled
best-effort, all workers drain and the original exception is rethrown on the caller.
No partial batch is returned after an exception; subsequent submissions reset job
state and reuse the pool. Constructor/destructor thread ownership is RAII and uses
no OpenMP or third-party threading runtime.

Tests force two workers to meet at a barrier, verify exactly-once uneven-tail
coverage, stress job reuse/concurrent callers, propagate callback/model exceptions,
and exercise lifecycle and recursive rejection. All four models in float/double
compare every result bit against direct `fit_single` calls with one/two workers,
including NaNs, invalid rows and max-iteration behavior. Dedicated cases preserve
mixed singular/converged/invalid statuses and shared invalid-x behavior. Native
ASan/UBSan and ThreadSanitizer runs validate memory and synchronization. CTest
reserves two processor slots for pool tests; only the explicit required benchmark
uses all detected cores.

`BENCHMARKS.md` records the N=100,000, M=32/128/512/2048 float32 CPU baselines,
hardware, input distributions, median-of-three timing boundaries and failure
counts. Google Benchmark uses real time and fits/s counters; a dependency-free
runner shares the exact workload and API when framework downloads are unavailable.
Every GPU step must compare against these inputs/options and report equivalent
end-to-end times separately from kernel-only timing. Dataset/setup/auto-guess and
pool creation are excluded; result allocation, validation, fitting, dispatch and
completion are included. Largest arrays are bounded and datasets are released.

## Apple GPU constraints

- Apple Silicon uses unified physical memory. Shared Metal buffers avoid explicit
  discrete-GPU transfers, but CPU copies, ownership, synchronization and memory
  bandwidth still cost time. Do not access GPU-written results before completion.
- Threads execute in threadgroups subdivided into SIMD groups, typically 32 lanes
  on Apple GPUs. Query `threadExecutionWidth` and maximum threads per threadgroup;
  do not assume every Metal device is Apple Silicon or hard-code 32 for correctness.
  Threadgroup barriers must be reached uniformly. Mask inactive lanes for tails.
- Metal shaders have no native 64-bit floating point. GPU math is float32;
  CPU double precision is the numerical oracle, and CPU float32 helps isolate
  algorithm differences from precision differences. Double inputs must not be
  silently reinterpreted as float buffers.
- Fast math is enabled by default in Metal compilation and can change NaN/Inf,
  reassociation and convergence behavior. The scaffold explicitly disables it
  with `MTL::CompileOptions::setFastMathEnabled(false)`. Enabling it later needs
  a separately verified numerical/performance tradeoff, not an incidental flag.
- Check every shader compilation, function lookup, pipeline creation, allocation,
  encoder creation and command-buffer completion. Include shader/kernel name,
  operation and available `NS::Error` detail. Never return unchecked output.
- metal-cpp is a header-only host interface; implementation macros occur in one
  translation unit. RAII owns retained objects and an autorelease pool covers
  each call. No Metal headers or frameworks leak into the CPU-only build.

The scaffold compiles `.metal` source at runtime (embedded by CMake), so a full
offline Metal compiler installation is not required. Compilation errors surface
in tests, rather than disappearing at configure time. CPU-only builds do not
fetch metal-cpp. Shader and header versions are described in `third_party/README.md`.

## Verification and performance strategy

The **CPU reference is the source of truth**. First test it against analytic
solutions, finite-difference Jacobian checks, synthetic spectra with known
parameters, noisy data, and degenerate/non-finite inputs. Fix seeds. Validate
parameters, final objective, status and iteration behavior, not just successful
return. Test damping rejection, singular systems, zero iterations, empty batches,
overflow and dimensions. Batch/GPU fitting tests must extend these single-fit tests.

Every GPU result must be compared with the CPU reference. Use absolute plus
relative tolerances justified by float32 conditioning and model scale, and
document expected iteration/status differences near convergence boundaries.
Bitwise identity is not required. The current residual smoke comparison uses
`abs_error <= 1e-6 + 1e-5*abs(cpu)` across empty, SIMD-width and threadgroup tails;
these are not proposed universal solver tolerances. A missing GPU fails macOS
Metal validation instead of silently skipping it.

GoogleTest exercises both backends. A dependency-free CPU smoke executable also
allows a limited offline compiler check, but cannot replace the full suite.
Google Benchmark starts with a bounded residual allocation/compute smoke test;
it is not a solver performance claim. Future benchmarks report N/M/P, dtype,
hardware, throughput, accuracy, warm-up, peak memory, and both steady-state kernel
and complete API latency (including setup/transposes). Limit ordinary workers to
two; do not create million-fit datasets until a numbered task needs them.

`scripts/check.sh` is the acceptance gate. It checks formatting, CPU and (on
macOS) Metal configure/build/tests, clang-tidy, and benchmark smoke execution.
Dependencies and all generated state remain project-local. Missing prerequisites
or unavailable GPU execution are failures/limitations, never "all checks pass".

## References

- [Apple metal-cpp setup and ownership](https://developer.apple.com/metal/cpp/)
- [Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [CMake presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html)
