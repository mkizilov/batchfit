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
automatic initial guesses and batched CPU fitting in float and double. It also
includes selectable Metal v1/cooperative solvers and Python bindings via pybind11/
scikit-build-core. It does **not** implement an extensible runtime model registry;
Python selects the four fixed model templates. See python/README.md for its
implemented API, dtype/device policy, borrowed buffers, GIL and ownership rules.
The backend-neutral API below remains a later contract; the implemented typed
CPU/shared-Metal APIs are described in the implementation sections.

## Public API contract

Optional parameter covariance/standard errors are implemented in
`cpu/uncertainty.hpp` and a separate Metal post-fit kernel. They use undamped
information at the accepted solution and `chi2/(M-P_free)`, with zero conditional
variance for fixed coordinates and explicit unavailable statuses for failed,
singular, no-DOF or active-bound cases. No uncertainty arrays/passes are allocated
by default. CPU single-fit users call the estimator explicitly; batch/Metal/
pipeline and Python APIs request storage explicitly. NUMERICS.md specifies the
formulas, normalization transform and measured marginal coverage.

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
| status | N | converged, max_iterations, singular_matrix, invalid_input, degenerate_data or unresolved_peak |
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
2*smaller_primary_half_width). The full-width exclusion rejects dense-sampling
noise maxima on the same peak's shoulder; a half-width admitted false components
in the large-M cooperative comparison. Its prominence must exceed three edge-noise scales;
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
that threshold: with the full-width rule, per scalar type 2,801/3,000 guesses
succeeded, 2,798 converged (93.267%) and 2,794 met the signal test (93.133%). The
original half-width rule measured 2,903/2,902/2,900 and admitted more ambiguous
features. Signal
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
deliberately limited to rank 1/2, C order and NumPy's alphabetical dictionary
key order. Reject unsupported dtype/order/version/rank, malformed/oversized headers,
truncation, trailing payload bytes, integer overflow and shape mismatches. Never
evaluate Python expressions. Generic Writer callers must append the full declared
element count and call `finish()` to check flush/close; Reader accepts partial
reads without allocating the full array.

The original `Reader` and default `Writer` remain strictly `<f4`; synthetic
loading never implicitly converts. Turn 10 adds **explicitly typed** Writer
overloads for `<f8` benchmark/input files and `<u4` status/iteration output, plus
`MappedReader` float32/float64 conversion. `inspect()` validates the same narrow
header/shape/byte envelope. Mapped inputs reject uint32 and all other dtypes.

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

## Out-of-core pipeline (turn 10)

`metal::fit_npy<Model>` accepts immutable POSIX NPY files `x[M]`, fit-major
`y[N,M]`, optional `initial[N,P]`, and a synchronous per-chunk result consumer.
It is a float32 fitting API: float64 inputs are explicitly rounded to float32;
nonfinite/overflowed rows retain the reference's invalid-input behavior. No
parameter/noise conventions or LM tolerances change. An empty initial path uses
the existing automatic guesses; failures use their NaN parameters and remain
accounted-for rows, never truth seeds or CPU fallback.

Only one y window is mapped at a time. Page-aligned read-only `mmap` windows
include their header-offset delta, validate element ranges and current file size,
and unmap after conversion. Inputs must not be modified/truncated during a run:
the size check cannot make concurrent truncation safe. Sequential/WILLNEED
mapping advice and bounded next-chunk read-ahead (`F_RDADVISE` on macOS,
`posix_fadvise` on other POSIX platforms) avoid relying solely on demand faults.
Advisory failures are reported. OS file-cache pages are reclaimable and are not
claimed as a process-RSS bound. Ordinary streamed Reader/Writer remain usable
without mapping; unsupported mapping platforms fail explicitly.

One producer and one readback worker share a fixed-depth ring (default three,
configurable 1–3). Each slot owns a complete shared-buffer batch. The producer
converts into one reusable fit-major float staging buffer, copies/guesses initial
parameters, then transposes with source-contiguous 16x16 tiles. Active N and spans
shrink for the final chunk; its point-major stride is **active N**, not capacity.
The producer commits asynchronously; the reader waits for checked completion,
calls the consumer in input order, and only then releases that slot. Consumers
must copy/write values before returning: retained views are **not snapshots**.

`submit_batch<Model>` configures/leases a batch and installs a completion handler
before commit. `PendingBatch` retains storage, buffers, queue and runtime even if
the original Context/batch dies. Completion checks command status/error and GPU
timestamps, catches numerical/runtime error reporting into an exception pointer,
and signals a condition variable. `get()` propagates it on the caller thread.
Busy leases reject mutation/resubmission until get/drain, and release exactly once
so an old consumed handle cannot unlock a newer submission. Destruction drains;
unobserved errors are printed. The callback uses a weak completion reference to
avoid a command-buffer/handler ownership cycle. An enabled first-dispatch GPU
capture waits only that first submission to preserve capture semantics; capture
is disabled for throughput measurements. A small production Objective-C++ bridge
uses an immutable by-value block and a registration/invocation publication mutex.
This replaces metal-cpp's mutable `__block std::function` shim: TSan reported its
driver handoff, while the explicit C++ synchronization passes TSan without race
suppression. This bridge is not the test-only native SDK runtime adapter.

Producer, consumer and observer exceptions stop new work, drain every committed
slot, join the worker and rethrow the first failure. Input/output lifetimes stay
valid through draining. The CLI refuses an existing output directory/timeline,
streams typed NPY output, checks final flush/close, and removes only its own
partial output on error. No full-N results, event arrays or work queues grow with
dataset size; observation storage is O(chunk_fits*M), ring depth at most three.

Automatic sizing uses device `recommendedMaxWorkingSetSize()/8`, capped at
256 MiB (64 MiB when no recommendation is provided). The estimate includes all
ring payloads, float conversion staging and the active mapped input/initials;
small headers/grid/page rounding/runtime caches remain overhead, not a promised
RSS ceiling. It rounds down to a multiple of 256 fits when possible and caps
against every buffer's `maxBufferLength`, uint32 `N*M`/`N*P`, and total fits.
Oversized explicit chunks are rejected instead of silently exceeding the device
limit. Explicit sizing overrides only the heuristic budget. Three slots mean
three possible committed, not-yet-completed buffers—not a claim of three kernels
executing simultaneously or a guarantee that an I/O-bound producer fills them.

The small test matrix covers four models, every variant, depths 1 and 3, float64
conversion, automatic/caller initials, tail indexing, ordered CPU parity,
busy/empty/reused buffers and exception draining. The large-file runner checks
every output against a cached CPU reference for its 64 deterministic noiseless
patterns without retaining the full input/results. Profiling/benchmark documents
distinguish these data from earlier noisy CPU/Metal matrices.

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

Turn 11 adds exact-flat/zero spectral diagnostics (`degenerate_data`) when x
varies and shape coordinates are free, plus `unresolved_peak` when an otherwise
converged FWHM is smaller than positive monotone-grid spacing. Codes 0–3 and
constant-x/fixed-coordinate behavior are preserved; new codes are 4 and 5.
`NUMERICS.md` specifies the conditions, coordinate transforms, opt-in compensated
accumulation/scaled Cholesky and float64-reference error study. Solver tolerances,
box projection and exact diagonal damping do not change. Normalization is explicit
preparation, not an implicit alteration of data or physical bounds.

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

## Metal LM v1 and shared-buffer API

`src/metal/lm.metal` ports the float CPU reference directly: one thread owns one
fit's fixed P<=8 private arrays, serial point accumulation, Cholesky, active/fixed
bounds, gain ratio/damping, stopping rules and status/iteration/cost outputs.
The shared model header is included, not reimplemented. Its Metal math macro is
globally qualified (`::metal::exp`) to avoid shadowing by the host API namespace.
Safe math plus precise FP32 functions disable fast math; explicit per-statement
contraction matches the CPU compiler policy, not a claim of bitwise transcendental
or ill-conditioned nonlinear-trajectory identity.

`tools/embed_metal.py` expands project includes into a generated C++ string at
build time and inserts `#line` before/after each shared header. MSL compilation
uses `newLibrary(source,options)` at context creation, with source file/line in
reported NSError diagnostics. No offline compiler or runtime source-file access
is required. CPU-only compilation never includes or downloads metal-cpp.

`metal::Context` caches device/queue/library and four pipelines.
`make_batch<Model>(N,M)` allocates x[M], point-major y[M*N], fit-major init[N*P],
config and output buffers in `MTL::ResourceStorageModeShared`. Public spans map
their contents directly. `fit_batch<Model>(batch, Device::Metal, options)` updates
only the small shared config, binds the existing buffers, dispatches, waits and
returns owning views. No array staging, blit, upload, transpose or result copy is
performed during that call. The CPU reference overload on the same storage is
explicit `Device::Cpu`; ordinary pooled CPU users retain their fit-major API.

Each pipeline supplies `threadExecutionWidth` and `maxTotalThreadsPerThreadgroup`.
Use `width*min(8,maxThreads/width)` threads per group, rejecting invalid limits;
dispatch exactly N threads and keep a shader bounds guard. The measured M4 uses
32-wide SIMD and 256-thread groups for every model. uint32 index-product checks
and device maxBufferLength checks precede allocations; a zero-fit batch skips
dispatch and returns empty views. The shared scalar config has checked C++
size/status-code layout and no STL/pointers in MSL (108 bytes since turn 11,
including numerical flags/grid diagnostics; historical v1 traces used 92 bytes).

`src/metal/runtime.cpp` is the production metal-cpp driver: NS::SharedPtr owns
retained objects and autorelease pools bound temporary lifetimes. Storage owns a
resource bundle which owns its runtime; result views retain storage, with no
ownership cycle. Objects survive context destruction while results remain, and
are released when the last owner goes away. Queue submissions serialize under a
runtime mutex; users retain exclusive access to any batch during execution.
Views alias subsequent writes if that batch is reused. Every command is checked
for completed status and non-null error before reading timestamps/output. GPU
time is GPUEndTime-GPUStartTime after waitUntilCompleted; wall time includes host
config/encoding/submission/wait, excluding context creation and input preparation.
The safe/precise host path requires macOS 15+; unsupported/disabled Metal throws,
never silently selects CPU.

The test-only installed-SDK driver `tests/metal_runtime_native.mm` implements the
same narrow host entry points with ARC so the production embedded MSL, shared
storage mapping, public Device API, timestamps and numerical comparisons can run
without downloadable metal-cpp. It is not a production fallback. The actual
metal-cpp host was uncompiled during v1 recording. In turn 9, repaired DNS allowed
local dependencies and successful production builds/CTest/GoogleTest/static
analysis. Native tests remain an independent oracle and verify buffer ownership
after context/batch destruction, stable spans, model/dimension rejection, empty
batches, failure statuses, clamping, and compiler file/line diagnostics. Metal API
and shader validation were enabled for the 2,832-fit strict suite, which keeps
parameter atol/rtol=2e-4/2e-4 and cost atol/rtol=1e-6/2e-5.

The full CPU-v1 benchmark datasets were also evaluated per fit (N=100,000 at each
model/M). All 1.6 million statuses/costs match their float CPU reference, including
33 singular paired fits. Twelve parameter rows exceed the strict diagnostic:
7 pseudo-Voigt at M=32, 2 at M=512, and 3 paired at M=2048. Two paired rows also
exceed the pre-existing CPU float floor `5e-4*(1+abs(parameter))`. Both backends
have high-cost local minima with both components on the same peak for those rows;
largest parameter disagreement is about 0.00516. These were **historical parity
limitations on the original guesses**, not accepted by the benchmark gate. The gate emits
all timing rows, preserves mismatches and exits nonzero. No tolerance assertion
in the strict suite was relaxed, no CPU fallback or per-fit result correction was
introduced, and the CPU solver/initializer/baseline were left unchanged. See
BENCHMARKS.md for timings and exact observed cases. The turn-9 initializer repair
is documented below; a new production v1 full-matrix regression exits zero with
all statuses, costs and reference-floor checks passing (nine stricter pseudo-Voigt
diagnostic rows remain). This does not certify arbitrary ill-conditioned inputs.

## Cooperative Metal v2 and dispatch

`src/metal/cooperative.metal` adds two explicitly selectable float32 variants:

- **SimdPerFit:** one 32-lane SIMD group owns a fit. Lane k traverses points
  k,k+32,... and accumulates packed lower-triangle J^T J plus J^T r (9, 14 or 44
  floats for P=3,4,8). `simd_sum` reduces every term. Only lane zero performs
  checked Cholesky, projection, predicted reduction and LM acceptance/damping.
  `simd_broadcast` shares the raw step; every lane computes the same trial
  parameters before the cooperative trial-cost reduction.
- **ThreadgroupPerFit:** one whole group (256 threads on this M4) owns a fit.
  Threads split points by group width. SIMD reductions write one partial vector
  per SIMD group to threadgroup memory; a uniform group barrier precedes the
  leader's inter-SIMD fold. Initial/trial costs and validation flags use the same
  two-level reduction. This avoids a full P-by-P scratch copy for every lane.

The leader's two matrices/vectors and per-SIMD partials live in threadgroup
scratch. SIMD mode partitions scratch into at most eight independent fit states
per group. It uses **only SIMD-group barriers**, so one fit can converge/return
without stranding another fit. Whole-group mode has one shared state and reaches
all threadgroup barriers uniformly, including invalid/singular/zero-budget paths.
Padding SIMD fits beyond N return uniformly within their own SIMD group. Host
dispatch uses full threadgroups (not partial groups whose size would change fit
indexing); fit bounds guards preserve arbitrary N. Cooperative kernels require
pipeline execution width 32 and reject unsupported widths instead of misindexing.

All solver policies, bounds, defaults and status codes remain those of the CPU
reference. Only accumulation order changes: lane-local sums and reduction trees
are not the serial float sum. v1 remains `KernelVariant::ThreadPerFit`, including
its original entry-point names and profiling specializations. Historical v1
benchmark helpers now force v1 explicitly. The public shared-buffer call accepts
a final `KernelVariant` argument and defaults to `Automatic`; returned timings
report the resolved variant. Layout remains point-major, with no hidden transpose,
CPU fallback or per-fit result correction.

### Data-only initialization regression uncovered by the sweep

The first comparison retained the old automatic seeds and exposed collapsed
paired local minima at dense M. Some guesses placed both components on a primary
peak/shoulder; CPU and GPU then had high-cost, weakly determined local solutions.
The original data are retained in `profiling/cooperative_original_seeds.csv`,
including all error counts. These preliminary timings used three sequential
samples per variant and are **not** the final order-balanced crossover table.

This was fixed at its source, not by relaxing GPU tests: the initializer's
data-derived exclusion distance is now twice the smaller measured primary
half-width (a full-width separation) instead of one half-width. A new dense-grid
shoulder-noise regression failed with the old rule (selected center -2.246 instead
of -1.5) and passes in float/double with the fix, still finding the weaker peak at
+1.5. No truth parameters, disjoint truth-derived boxes, solver changes or output
filters were added. Rejected guesses still remain in denominators, and benchmarks
retain their documented fixed non-oracle fallback seeds.

The synthetic observations/seeds/distributions and fitting options are unchanged;
only data-derived initial arrays can change. Therefore the original CPU/GPU v1
tables remain historical observations, not relabeled measurements. The new fair
comparison times **forced v1 and both cooperative variants on the same corrected
initial arrays**, and evaluates the CPU reference on those exact arrays. All
128 final matrix rows have zero status/parameter/cost errors. Single-peak guess
recovery remains 18,000/18,000; the conservative paired guess study now reports
2,801 guesses, 2,798 converged and 2,794 signal recoveries per 3,000 trials/type
(93.267% convergence, 93.133% signal recovery), rather than hiding extra rejections.

### Production benchmark and measured crossover brackets

Machine: Mac16,10 / Apple M4, 10 CPU cores, 16 GiB, Apple Clang 21.0.0. These are
**production metal-cpp** Release runs, not the SDK adapter. Pinned local tools
and FetchContent dependencies became available after the sandbox DNS repair.
The production host, NS::SharedPtr path, GoogleTest and Google Benchmark smoke
now compile/run; historical unverified-host notes describe earlier turns only.

N=16,384 keeps the largest M=4,096 observation array at 268,435,456 bytes and the
comparison bounded in memory. Peak RSS in the final run was **635,699,200 bytes
(606.25 MiB)**. One CPU dataset and one shared GPU dataset are used during setup;
the fit-major CPU dataset is freed before GPU timing. No datasets remain on disk.
This is a large-batch, point-major float32 calibration; it is not a promise of
optimal scheduling for tiny N or other GPU architectures.

Every forced variant gets a full untimed batch warm-up. Five timed rounds rotate
the order of v1/SIMD/group/automatic to reduce clock/cache ordering bias. Each
timed output is compared against CPU outside the timing interval. GPU columns
are command-buffer timestamps; the CSV also contains external wall time/fits/s,
actual dispatch variant, status/cost/parameter errors and selection regret.
Generation, guessing, buffer creation and CPU validation are excluded; existing
buffer binding, encoding/submission and completion are included in wall time.
No capture or debug validation was enabled for performance measurements.

The following are **median GPU milliseconds**; T=v1 thread/fit, S=SIMD/fit,
G=whole group/fit, A=automatic. The complete 128 rows are preserved in
`profiling/cooperative_results.csv`.

| Model | M | T ms | S ms | G ms | A ms | Selected |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Gaussian | 32 | 0.394 | 1.657 | 11.418 | 0.397 | T |
| Gaussian | 64 | 0.664 | 1.851 | 10.834 | 0.660 | T |
| Gaussian | 128 | 1.130 | 2.262 | 10.713 | 1.157 | T |
| Gaussian | 256 | 2.183 | 3.019 | 11.859 | 2.189 | T |
| Gaussian | 512 | 4.573 | 4.756 | 12.943 | 4.756 | S* |
| Gaussian | 1024 | 10.452 | 10.164 | 15.471 | 10.619 | S |
| Gaussian | 2048 | 24.614 | 33.358 | 24.749 | 24.754 | G* |
| Gaussian | 4096 | 65.187 | 178.446 | 49.917 | 49.904 | G |
| Lorentzian | 32 | 0.393 | 1.665 | 11.810 | 0.395 | T |
| Lorentzian | 64 | 0.635 | 1.860 | 11.311 | 0.636 | T |
| Lorentzian | 128 | 1.249 | 2.339 | 11.611 | 1.246 | T |
| Lorentzian | 256 | 2.411 | 3.168 | 12.373 | 2.418 | T |
| Lorentzian | 512 | 5.167 | 5.102 | 13.472 | 5.104 | S |
| Lorentzian | 1024 | 13.389 | 10.435 | 16.386 | 10.641 | S |
| Lorentzian | 2048 | 35.135 | 29.680 | 24.883 | 24.864 | G |
| Lorentzian | 4096 | 79.371 | 182.590 | 50.039 | 50.085 | G |
| PseudoVoigt | 32 | 0.576 | 3.011 | 22.983 | 0.572 | T |
| PseudoVoigt | 64 | 0.871 | 3.308 | 23.011 | 0.892 | T |
| PseudoVoigt | 128 | 1.618 | 3.979 | 24.459 | 1.673 | T |
| PseudoVoigt | 256 | 3.089 | 5.492 | 28.308 | 3.145 | T |
| PseudoVoigt | 512 | 6.980 | 8.997 | 30.524 | 6.873 | T |
| PseudoVoigt | 1024 | 17.418 | 17.457 | 35.627 | 17.004 | S* |
| PseudoVoigt | 2048 | 41.875 | 47.865 | 47.709 | 41.366 | T |
| PseudoVoigt | 4096 | 96.431 | 215.353 | 78.271 | 78.392 | G |
| TwoLorentzian | 32 | 3.545 | 11.427 | 90.070 | 3.608 | T |
| TwoLorentzian | 64 | 4.687 | 11.483 | 83.222 | 4.645 | T |
| TwoLorentzian | 128 | 7.146 | 13.234 | 84.838 | 6.982 | T |
| TwoLorentzian | 256 | 12.958 | 16.798 | 90.225 | 13.083 | T |
| TwoLorentzian | 512 | 28.365 | 24.216 | 94.695 | 24.224 | S |
| TwoLorentzian | 1024 | 61.983 | 47.433 | 107.845 | 44.757 | S |
| TwoLorentzian | 2048 | 137.978 | 123.032 | 134.240 | 122.058 | S |
| TwoLorentzian | 4096 | 317.922 | 384.713 | 191.953 | 193.481 | G |

`*` Near-ties: Gaussian 512/2048 and pseudo-Voigt 1024 have selected forced
medians within 5% of the fastest. The benchmark treats differences below 5% as
performance-noise ties, not numerical tolerance. Final selection is the exact
fastest in 29/32 cells and within **4.01%** in the other three; all 32 pass the
hard 5% selection-regret check. Actual auto-call time can differ slightly from
the same forced pipeline's median because GPU clocks/load are uncontrolled.

The following is the **historical turn-9** policy. Turn 11's specialized numerical
pipelines/diagnostic guards shift Gaussian's T/S threshold to 1024: M=512 now
uses T. Other buckets are unchanged. Its separate full matrix has maximum auto
regret 2.61568%; see NUMERICS.md and `profiling/numerics_dispatch_results.csv`.

| Model | M-policy on measured domain | Observed crossover brackets |
| --- | --- | --- |
| Gaussian | T for M<512; S for 512<=M<2048; G for M>=2048 | T/S in (256,512], weak tie at 512; T/G near 2048, robust G by 4096 |
| Lorentzian | T for M<512; S for 512<=M<2048; G for M>=2048 | T/S in (256,512]; S/G in (1024,2048] |
| PseudoVoigt | T below 1024; S for 1024<=M<2048; T for 2048<=M<4096; G at 4096 | T/S near 1024 is weak; S loses again by 2048; G wins in (2048,4096] |
| TwoLorentzian | T for M<512; S for 512<=M<4096; G at 4096 | T/S in (256,512]; S/G in (2048,4096] |

Only the power-of-two points were measured; between them the policy is an
empirical bucket choice, not a discovered exact continuous threshold. Outside
M=32..4096, automatic selection conservatively uses v1. Forced variants remain
available. Automatic also stays on GPU v1 if a cooperative pipeline's SIMD width
is not 32; forcing a cooperative variant on that hardware reports unsupported
width. This capability branch is source-checked but not hardware-tested here.
SIMD mode slows sharply at M=4096 with point-major access, so a
monotonic "larger M always SIMD" rule is wrong on this hardware. Whole-group mode
has too much leader/barrier overhead at small M but wins at the largest sampled M.

At M=4096 automatic GPU time improves over forced v1 by 1.31x Gaussian, 1.58x
Lorentzian, 1.23x pseudo-Voigt and 1.64x paired. Small-M v1 is deliberately kept.
The CPU baseline tables in BENCHMARKS.md use N=100,000 and original guesses, so
these are new same-dataset **v1-v2** comparisons, not relabeled CPU-v1 speedups.

### Verification and reproduction

The strict unit matrix has 5,152 reference fits across all models, N=1/31/33/257
and M=32/128/512/4096, exercised through forced T/S/G and automatic (20,608 GPU
fit comparisons, plus failure/empty/bound cases). It keeps parameter
atol/rtol=2e-4/2e-4 and cost atol/rtol=1e-6/2e-5. Tail masking, zero/singular/
invalid states and fixed/active bounds all pass. Benchmark comparisons use the
pre-existing float reference floor `5e-4*(1+abs(parameter))` for converged outputs
and the same cost bound; no numerical threshold was weakened. Every one of five
timed repetitions is validated; CSV error counts are worst repetition counts.

The production CPU and Metal presets, GoogleTest/CTest and bounded Google
Benchmark smoke, generator/NPY checks and clang-tidy pass `scripts/check.sh`.
clang-tidy 21.1.6 is pinned to match the supplied compiler/SDK. It reads the actual
build compiler's resource headers on macOS; intentional enum ABI widths and
homogeneous shape-validated numeric parameters are documented style exceptions.
Metal bridge analysis models metal-cpp's intentional retain/release-to-nil calls
without disabling other memory checks. Small boundary/type/exception fixes exposed
by the formerly unavailable static gate do not change solver math.
Production ASan plus Metal API/shader-validation runs also pass the cooperative
suite. A separate production v1 N=100,000 regression reran all 16 historical
model/M cases on corrected seeds: no status, cost or reference-floor differences,
including the former large paired failures; nine stricter pseudo-Voigt diagnostic
rows remain. Historical v1 tables retain their original measurements.

```sh
export PATH="$PWD/.venv/bin:$PATH"
cmake -S . -B build/metal-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk -DBATCHFIT_METAL=ON -DBATCHFIT_BUILD_TESTS=OFF
cmake --build build/metal-release --parallel 2
build/metal-release/batchfit_cooperative_baseline > profiling/cooperative_results.csv
sh scripts/check.sh
```

`--smoke` uses N=257 at M=32/512, one timing sample, and checks numerical parity
without claiming tiny-N performance calibration. Full selection-regret failures
or any CPU parity mismatch cause a nonzero benchmark exit; no rows are omitted.
Raw timing data, original-seed failures and exact commands are retained; all
scratch datasets/probes are removed, and no server is required.

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
