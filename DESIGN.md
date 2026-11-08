# batchfit design

## Scope and workload

batchfit is a C++20 library for **N independent nonlinear least-squares fits**,
typically millions of spectra, each with M samples and P model parameters,
where 1 <= P <= 8. A hyperspectral image supplies one spectrum per pixel. Fits
share a model and usually an x grid, but have independent observations, initial
parameters, convergence decisions and outputs. There is no coupling between fits.

Turn 1 provides build/test infrastructure and an internal squared-residual compute
smoke primitive on CPU and Metal. It does **not** implement fitting, a model
registry, CPU fit scheduling or Python bindings yet. The following fitting API
and solver are the implementation contract for subsequent tasks, not a claim
that an unimplemented function works.

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
| status | N | converged, max-iterations, singular, invalid-input or numerical-failure |
| iterations | N | Number of attempted LM updates (zero before any update) |
| chi_squared | N | Sum of squared residuals at the final accepted parameters |

The initial objective is unweighted `sum_j (y_j - f(x_j, p))^2`; it is not reduced
chi-squared and is not divided by degrees of freedom. Weighting is not implicit.
For invalid fits, chi-squared is NaN and parameters retain the initial values.
Batch-level errors (invalid dimensions, unavailable device, allocation or Metal
execution failures) throw exceptions with operation-specific context. Device
errors are not numerical fit statuses. Outputs are available only after GPU
completion and validation of command-buffer status.

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

For each fit, evaluate residual `r = y - f(x,p)` and the model Jacobian
`J[j,k] = df(x_j,p)/dp_k`. Accumulate `A = J^T J`, `g = J^T r` and the objective.
Compute a candidate step by solving:

```text
(A + lambda * diag(max(A[k,k], scale_floor))) delta = g
p_trial = p + delta
```

Use a small dense solve with explicit singularity and non-finite checks; do not
form an inverse. P <= 8 bounds local matrix storage. Start with a straightforward
double-precision CPU reference, including a well-defined factorization and
damping policy, before parallelizing. The current float residual smoke primitive
is only a plumbing check, not that future double-precision solver.

Accept a trial only when its objective improves and the gain ratio is positive;
reduce damping on accepted steps and increase it on rejection. A rejected step
must not overwrite the accepted parameters or objective. Check gradient, scaled
step and relative objective tolerances, maximum iterations and numerical failure
explicitly. Document exact thresholds and lambda updates with solver code/tests
when implemented, so CPU and GPU use the same decisions. Do not promise a global
minimum; bad initial guesses, degenerate data and unidentifiable models must have
deterministic, tested outcomes.

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
overflow and dimensions. Later fit-level tests must supplement the scaffold tests.

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
