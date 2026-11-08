# Float32 numerical robustness

## Recommendation and scope

For difficult units/ranges, **normalize in double on the CPU before converting
to float32**, fit with **safe/precise Metal math**, enable compensated accumulation
and scaled Cholesky, then reconstruct physical parameters in double on the CPU.
This does not put double precision in a shader. Keep the normalized float32
parameters if a downstream consumer cannot represent the physical coordinates.

The measured recipe reduced pooled unit-parameter RMS error against float64 CPU
from **3.50811e-2 to 4.77326e-6**, with maximum error **1.51354e-4**. All 122,880
GPU fit executions in the study converged and matched reference status, but raw
float32 sometimes stopped with **0.290755** parameter error despite reporting
convergence. Status alone is not an accuracy guarantee when tolerances and
parameters have incompatible units.

The default remains the existing safe/precise, standard-accumulation path, with
unchanged solver tolerances and explicit opt-in numerical flags. This preserves
ordinary reference behavior and the ability to reproduce earlier measurements.
New diagnostics deliberately distinguish degenerate/unresolved spectral inputs.
The retained remedies are production code, not benchmark-only substitutes.

## What was measured

Production metal-cpp on **Mac16,10 / Apple M4, 16 GiB**, Apple Clang 21.0.0,
macOS 27.2, Release `-O3 -DNDEBUG`, without CPU fast-math. Shader libraries are
compiled separately with `MTLMathModeSafe` + precise FP functions and
`MTLMathModeFast` + fast FP functions. The preexisting project default already
overrides Metal's fast-math default with safe/precise; this turn adds an explicit
experimental fast context rather than pretending the old path used fast math.

- Four models, **M=65 and 2049**, 16 fits per cell, SNR **10, 30, 100, 1000 and
  noiseless** (`snr=0` in CSV). Noise is Gaussian; linear SNR is clean-signal RMS
  divided by noise standard deviation, including the paired baseline.
- Every one-thread/fit, SIMD/fit and threadgroup/fit kernel; every combination of
  raw/normalized coordinates, ordinary/Kahan accumulation, ordinary/scaled
  Cholesky, safe/fast math: **7,680 CSV rows / 122,880 GPU fits**.
- **2,560 input cases** and normalized float64 CPU references. Seeds are fixed:
  `20261108 + P*7919 + M`. Unit regimes and SNRs intentionally share underlying
  parameter/random draws to isolate changes, not independent statistical trials.
- Initial guesses use the noisy observed data, never truth. Identical conceptual
  initial parameters are transformed/rounded for each configuration. Truth is
  used only to generate data and measure statistical estimation error.
- The reference uses the existing `cpu::fit_single<double>` LM equations, tight
  double tolerances, normalized coordinates, Kahan and scaled Cholesky. No
  alternative optimizer or GPU-to-CPU fallback is used.

| Regime | x origin | x half-range | Signal/amplitude scale |
|---|---:|---:|---:|
| ordinary | 0 | 1 | 1 |
| offset | 1e6 | 5 | 1 |
| small_units | 1e3 | 0.005 | 1e-6 |
| large_units | -1e5 | 500 | 1e6 |

Single-peak normalized centers span about [-0.2,0.2], widths 0.12–0.27 (common
FWHM 0.15–0.35 for pseudo-Voigt), amplitudes 1.5–2.5, and mixing 0.2–0.8.
Paired centers are separated, widths 0.06–0.12, amplitudes 1.5–2.5, with baseline
0.15 and small signed slope. These ranges are controlled coverage, not a claim
about all blends, amplitude ratios, grids or noise models.

## Error definitions and complete configuration table

Errors are against the float64 fit to the **original double observations**, not
against the noisy generating truth. Physical parameters are mapped back to the
same fitted unit coordinates `q` for error calculation. Centers/widths therefore
have a range-relative error, amplitudes/baseline a signal-relative error, and
mixing remains dimensionless. This avoids hiding a bad center under a huge
physical offset. CSV also includes every physical parameter's maximum absolute
error; parameter order/names remain those in DESIGN.md.

Pooled RMS weights each fitted parameter equally across input cases and kernels.
Max is the largest absolute q-coordinate error. **No failed guesses, reference
fits, nonfinite outputs or status mismatches were discarded:** all four counts
were zero. The CSV nevertheless records those denominators explicitly; finite
best-state errors would remain included even for a non-converged GPU result.

| Coordinates | Accumulation | Cholesky | Math | Pooled RMS error | Max error |
|---|---|---|---|---:|---:|
| raw | ordinary | ordinary | safe | 3.50811e-2 | 2.90755e-1 |
| raw | Kahan | ordinary | safe | 3.50811e-2 | 2.90755e-1 |
| raw | ordinary | scaled | safe | 3.50811e-2 | 2.90755e-1 |
| raw | Kahan | scaled | safe | 3.50811e-2 | 2.90755e-1 |
| normalized | ordinary | ordinary | safe | 5.74532e-6 | 2.10083e-4 |
| normalized | Kahan | ordinary | safe | 4.86783e-6 | 1.51354e-4 |
| normalized | ordinary | scaled | safe | 5.70285e-6 | 2.10142e-4 |
| **normalized** | **Kahan** | **scaled** | **safe** | **4.77326e-6** | **1.51354e-4** |
| raw | ordinary | ordinary | fast | 3.50811e-2 | 2.90755e-1 |
| raw | Kahan | ordinary | fast | 3.50811e-2 | 2.90755e-1 |
| raw | ordinary | scaled | fast | 3.50811e-2 | 2.90755e-1 |
| raw | Kahan | scaled | fast | 3.50811e-2 | 2.90755e-1 |
| normalized | ordinary | ordinary | fast | 6.99304e-6 | 2.64472e-4 |
| normalized | Kahan | ordinary | fast | 6.41633e-6 | 2.16930e-4 |
| normalized | ordinary | scaled | fast | 7.18383e-6 | 2.64472e-4 |
| normalized | Kahan | scaled | fast | 7.20647e-6 | 2.64532e-4 |

Normalization dominates; compensation/scaling alone cannot fix already rounded
coordinates or a stopping tolerance expressed in unsuitable units. After
normalization, Kahan reduces pooled RMS about 15%; scaled Cholesky adds a modest
further improvement. Keep both for the accurate recipe. Fast math is not a
universal improvement and is not recommended for the strict accuracy budget.
Raw configurations intentionally remain in the table, not relabeled as passes.

### Recommended recipe by model and SNR

| Model | Pooled RMS | Max q error |
|---|---:|---:|
| Gaussian | 1.45672e-6 | 1.86601e-5 |
| Lorentzian | 1.33689e-6 | 1.71070e-5 |
| Pseudo-Voigt | 9.79822e-6 | 1.51354e-4 |
| Two Lorentzians + baseline | 1.33998e-6 | 2.60703e-5 |

| SNR | Pooled RMS vs CPU64 | Max vs CPU64 | CPU64 max error vs generating truth |
|---|---:|---:|---:|
| noiseless | 2.90955e-7 | 5.45621e-6 | 9.63567e-11 |
| 10 | 8.88697e-6 | 1.51354e-4 | 1.11063e-1 |
| 30 | 5.33961e-6 | 1.02481e-4 | 3.63349e-2 |
| 100 | 2.50210e-6 | 3.64918e-5 | 1.08347e-2 |
| 1000 | 2.92231e-7 | 3.35227e-6 | 1.08101e-3 |

At low SNR, parameter estimation uncertainty is much larger than the remaining
float32/float64 gap. Pseudo-Voigt mixing is the largest remaining parameter-error
contributor here. Neither fact guarantees identifiability or a global minimum for
other spectra. All fits in this controlled matrix converged; pathological inputs
below are separate tests, not omitted difficult accuracy-study rows.

## Quantization is a separate limitation

The diagnostic refits rounded raw x/y in float64 after normalization, using the
same data-derived conceptual seed. It measures the effect of data rounding on
this optimizer, **not a mathematical lower bound** or a guarantee of the same
basin for arbitrary data. A second diagnostic rounds normalized inputs first.

| Regime | Raw ordinary RMS | Recommended RMS | Max raw-input rounding diagnostic | Max normalized-input rounding diagnostic |
|---|---:|---:|---:|---:|
| ordinary | 7.47408e-6 | 4.77343e-6 | 6.15588e-8 | 4.18563e-8 |
| offset | 5.04207e-2 | 4.77283e-6 | 6.32215e-2 | 4.18566e-8 |
| small_units | 4.87903e-2 | 4.77335e-6 | 6.27616e-2 | 4.18567e-8 |
| large_units | 8.51131e-6 | 4.77343e-6 | 3.10638e-6 | 4.18563e-8 |

Float32 near x=1e6 has an ULP of 0.0625. Converting those x values to float32
first, then centering, cannot recover lost sample positions. Normalization must
see the original double values. `Coordinates::physical<Model>` reconstructs
physical values in host double; casting them back to physical-unit float32 can
lose the center precision again. The out-of-core API retains its existing
float32 conversion contract; normalization is an explicit preparation step, not
a hidden promise that already converted NPY data can be repaired.

## Retained algorithms and API

`Coordinates::from(x_double,y_double)` chooses origin at the x-range midpoint,
half-range `r`, and `a=max(abs(y))` (1 for zero data). It rejects nonfinite inputs,
empty/mismatched arrays or a collapsed x range with a preparation exception.
The fitting APIs themselves report invalid/degenerate status for pathological
data. Coordinates transform `u=(x-origin)/r`, `v=y/a` before conversion to float.
Amplitudes divide by a, centers translate/divide by r, widths divide by r, and
pseudo-Voigt mixing is unchanged. For the paired baseline:

```
B = (b + slope*origin)/a
S = slope*r/a
slope = S*a/r
b = B*a - slope*origin
```

Host `fma` reduces reconstruction cancellation. Parameter counts, physical model
definitions and analytic Jacobians are unchanged. **Bounds/tolerances supplied
to a normalized fit must be in normalized units.** In particular, independent
physical b/slope box bounds become coupled after centering; the helper does not
pretend they remain independent boxes. Keep the raw API or choose appropriate
normalized constraints for that case. Do not silently transform only one bound.

Enable `options.compensated_sum=true` and `options.scaled_cholesky=true` for CPU
or Metal fitting. Kahan accumulates objective, J^T r and J^T J. Cooperative kernels
compensate per-lane partials and use existing SIMD/group reductions; no public
promise about a particular hardware reduction order is made. Shared numerical
helpers remain C++/MSL compatible and heap-free in the single-fit reference.

Scaled Cholesky preserves **exact diagonal LM damping**. After constructing
`A=H+lambda*diag(H)` and excluding active coordinates, form `D_i=sqrt(A_ii)`, solve
`(D^-1 A D^-1) z = D^-1 g`, then `delta=D^-1 z`. Evaluate entries as
`(A_ij/D_i)/D_j` rather than overflowing `D_i*D_j`. Non-positive/nonfinite pivots
still fail; there is no arbitrary ridge or hidden regularization to disguise a
singular fit. Accepted-state, projection and stopping policies are unchanged.
The raw/normalized table therefore measures the combined effects of input
rounding, conditioning and expressing those same stopping tests in suitable
units; it is not mislabeled as an isolated hardware rounding-error experiment.

Numerical choices are specialized with Metal function constant 0 and cached
lazily under the runtime mutex (12 base pipelines, up to four option sets).
Standard kernels do not retain Kahan's private correction arrays or scaled-solve
workspace. Serialized config/pipeline flags are checked for agreement. Shader
compilation, specialization and pipeline failures retain contextual diagnostics.

## Pathological inputs and statuses

The original numeric codes 0–3 are unchanged; codes 4–5 are appended. All kernels,
both scalar CPU types, all numerical-flag combinations and safe/fast Metal are
covered by `tests/numerics_checks.hpp`.

| Input / condition | Meaningful result |
|---|---|
| NaN or positive/negative Inf observation; nonfinite shared x | `invalid_input` (3), no successful result |
| Finite inputs that overflow objective/J^T J | `invalid_input` (3), not fabricated convergence |
| Exactly flat nonzero or all-zero spectrum, varying x and free shape parameters | `degenerate_data` (4), finite accepted initial objective when evaluable |
| Resolved peak centered at the sampled range edge | Valid fit can be `converged` (0); not rejected merely for being at the edge |
| Converged fit with FWHM below minimum positive grid spacing | `unresolved_peak` (5), parameters/cost retained but not advertised as resolved convergence |
| Unidentifiable columns / failed checked factorization | `singular_matrix` (2), best accepted state retained |

The FWHM test uses 2.354820045*sigma, 2*HWHM, or the pseudo-Voigt FWHM. Each
paired width is checked. It applies to monotone grids (duplicates are allowed);
unordered grids do not get an invented spacing estimate. This is a conservative
sampling warning, not an identifiability proof. A singular narrow fit can instead
end as singular. Constant-x historical tests preserve their singular/constrained
semantics; fully fixed shape coordinates do not invent free shape degeneracy.
Automatic initialization can independently report no peak/unresolved structure;
failed guesses remain accounted-for invalid rows in the pipeline.

## Fast math and nonfinite behavior

The native compiler probe in `profiling/numerics_math_probe.csv` tests ordinary,
NaN, +/-Inf and maximum-finite inputs. **On this compiler/device**, builtin
`isfinite`/`isnan` returned the same answers in safe and fast libraries, NaN/Inf
arithmetic remained nonfinite, and maximum-finite times two overflowed to Inf.
Do not fabricate a fast-math NaN failure that was not observed. Nor does this
probe establish a portable fast-math guarantee: fast mode permits lossy/finite
assumptions and reassociation, including transformations that can harm Kahan.

Production shader guards inspect IEEE exponent bits for finiteness/NaN, including
intermediate matrices/trials/objectives, instead of relying on an optimizer's
floating classification assumption. All tested invalid rows were rejected in
both math modes. Fast normalized fitting still had a larger measured parameter
gap and is experimental. Safe/precise remains the supported default.

## Performance regression found and corrected

An initial dynamic-flags implementation reserved correction workspace even when
disabled. It slowed v1 substantially and failed the existing 5% dispatch gate.
The partial failed run is retained as `profiling/numerics_dispatch_dynamic.csv`,
not hidden. Function-constant specialization removes unused workspaces. Combined
integer diagnostic reduction avoids an extra collective/barrier for flat data.

The resulting numerical checks moved one measured crossover: Gaussian M=512 is
now v1, while M=1024 remains SIMD. Two order-balanced runs showed the old M=512
choice more than 5% slower; `profiling/numerics_dispatch_before_policy.csv`
retains that failure. The final **128-row, five-run-median** production matrix
(`profiling/numerics_dispatch_results.csv`) has zero CPU status/parameter/cost
errors, and automatic selection's maximum regret is **2.61568%**, within the
unchanged 5% budget. No numerical tolerance or benchmark failure exit was relaxed.
Earlier turn-9 results/policy remain historical measurements, not relabeled.
These performance rows use default numerical options; accuracy-study GPU times
are single small-batch observations, not a universal robust-mode speed claim.

## Reproduce and verify

```sh
export PATH="$PWD/.venv/bin:$PATH"
sh scripts/check.sh
cmake -S . -B build/metal-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBATCHFIT_METAL=ON -DBATCHFIT_BUILD_TESTS=OFF -DBATCHFIT_BUILD_BENCHMARKS=ON
cmake --build build/metal-release --target batchfit_numerics_study batchfit_cooperative_baseline --parallel 2
build/metal-release/batchfit_numerics_study \
  --output profiling/numerics_results.csv --probe profiling/numerics_math_probe.csv
python3 tools/check_numerics.py --csv profiling/numerics_results.csv
build/metal-release/batchfit_cooperative_baseline
```

Archive existing result/probe CSV deliberately before rerunning: the study refuses
overwrites. `--smoke` uses four fits, M=65, ordinary units, SNR=30 and every
configuration/kernel; the ordinary gate runs it with project-local temporary
output and validates the retained safe recipe's **2e-4 maximum q-error** without
asserting that intentionally raw/fast experiments meet that bound. Full study
validation checks coverage/denominators and the same retained-recipe gate.
Sanitizer commands follow PROFILING.md; its narrowly scoped upstream metal-cpp
nil-message UBSan exception is unchanged. Indexed diagnostics remain partial
without a root compile-database marker and an MSL producer; actual compilers,
clang-tidy, CTest and real GPU checks are authoritative.

The final numerical, single-fit and pipelined suites pass ASan/UBSan with Metal
API/shader validation; numerical lazy-cache and pipeline checks also pass TSan
with C++/Objective-C++ instrumentation and no race suppression. The independent
legacy SDK adapter still passes its **2,832 CPU/GPU comparisons** after explicitly
specializing default function constants. That test-only adapter remains safe-math,
standard-numerics verification; it explicitly rejects new option sets rather than
pretending to test production specialized pipelines. Legacy profiling kernels
also specialize default constants. No binary traces or sanitizer builds are
committed, and LeakSanitizer was not run.
The final `scripts/check.sh` exits zero: **23 CPU and 32 Metal CTests**, all
GoogleTest wrappers, formatting, selected clang-tidy checks, independent generator/
pipeline CLI checks, legacy profiling smokes and the numerical study smoke.
All **32 model/scalar/numerical-option** single fits pass zero-C++-heap allocation
instrumentation. The separate full CSV validation and full 128-row dispatch gate
also pass. No global tools/services, extra CPU workers or host settings were used.

### References

- [Metal compile options and math modes](https://developer.apple.com/documentation/metal/mtlcompileoptions)
- [Metal mathMode / fastMathEnabled mapping](https://developer.apple.com/documentation/metal/mtlcompileoptions/mathmode)
- [MSL function constants](https://github.com/dogukanveziroglu/metal-shading-language-specification/blob/main/ch05-function-variable-declarations.md)
