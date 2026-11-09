# Python bindings

The `batchfit` package uses pybind11 3.0.1 and scikit-build-core 0.11.6. Python
3.9–3.12 is supported by the pinned NumPy/SciPy environment. Everything installs
inside the repository's supplied `.venv`; no global Python package is needed.

```sh
export PATH="$PWD/.venv/bin:$PATH"
uv pip install -e .
# Development/test dependencies, if not already bootstrapped:
uv pip install -e '.[dev]'
```

`scripts/bootstrap.sh` preserves an existing environment and installs the editable
package plus exact dev pins. It never downloads a Python interpreter. Keep the
provided project-local HOME/TMPDIR/cache settings. After C++ edits, reinstall with
`uv pip install --reinstall-package batchfit -e .`; pure Python edits are editable
immediately. Automatic rebuild-on-import is intentionally disabled.

## API

```python
import numpy as np
import batchfit

x = np.linspace(-5, 5, 129)
y = np.stack([2 / (1 + ((x - center) / 0.7)**2) for center in (-0.2, 0.3)])
result = batchfit.fit(
    x, y, model="lorentzian", p0=None,
    bounds=([0, -5, 1e-6], [np.inf, 5, np.inf]), device="auto",
)
params, status, iterations, chi2 = result
assert np.all(status == batchfit.FitStatus.converged)
```

Exact signature:
`fit(x, y, model="lorentzian", p0=None, bounds=None, device="auto", *, uncertainties=False)`.

- Shared `x[M]`, fit-major `y[N,M]`. A single `y[M]` is treated as N=1; outputs
  **always** have batch dimensions. Empty N is supported; M must be positive.
- Models: `gaussian` (height, center, sigma), `lorentzian` (height, center, HWHM),
  `pseudo_voigt` (height, center, common FWHM, mixing), `two_lorentzian`
  (height1, center1, HWHM1, height2, center2, HWHM2, intercept, slope).
  Formulas/width conventions are unchanged; see DESIGN.md.
- `p0[P]` broadcasts, `p0[N,P]` supplies independent initials. None runs the
  existing data-only feature guesses in native code. No truth parameters or
  SciPy fallback are used. Failed guesses remain invalid rows, not removed fits.
- `bounds=(lower,upper)` accepts a scalar or `[P]` on either side, common to all
  fits. Per-fit bound matrices are not accepted. Equal finite bounds fix a
  coordinate. None means the existing C++ unbounded defaults; positive width
  bounds and `[0,1]` mixing bounds are recommended. Finite unrepresentable bounds
  raise ValueError rather than becoming silently unbounded after conversion.
- `device="cpu"` preserves common float32/float64 precision. Other real numeric
  inputs are converted using NumPy promotion, then float32/float64. Complex,
  object, string and boolean arrays are rejected instead of losing information.
- `auto` uses Metal for common float32 inputs when `metal_available()` is true;
  float64 stays on CPU to preserve precision. `metal` explicitly requests safe/
  precise float32 GPU fitting, including conversion of float64 input. It raises
  RuntimeError when unavailable; backend errors are never silently retried on CPU.

`FitResult` is a named tuple: attributes and ordinary four-array unpacking work.

| Field | Shape | dtype |
|---|---|---|
| params | N,P | float32 or float64 (always float32 for Metal) |
| status | N | int32 |
| iterations | N | uint32 |
| chi2 | N | same floating dtype as params; unweighted residual sum of squares |

`FitStatus` codes: 0 converged, 1 iteration budget, 2 singular matrix, 3 invalid
input, 4 degenerate data, 5 unresolved peak. Nonfinite observations/initials are
per-row failures, not a whole-batch crash; nonfinite shared x invalidates all
rows. Shape/model/device/global-bound errors raise exceptions. With p0=None,
flat/unresolved input can fail initial guessing and return invalid_input (NaN
parameters); with a valid p0 the fitter's spectral diagnostics remain available.
Always inspect status before interpreting parameters.

## Optional uncertainties

Use `uncertainties=True` to receive `FitResultWithUncertainties`: the original
four fields followed by `covariance[N,P,P]`, `stderr[N,P]` and
`uncertainty_status[N]`. With the default False, the four-array result/unpacking
is unchanged and no uncertainty output storage is allocated.

The undamped analytic `J.T @ J` is recomputed at the accepted solution and its
inverse scaled by `chi2/(M-P_free)`. Fixed coordinates have zero conditional
variance and do not consume residual degrees of freedom. Singular information,
failed fits, insufficient DOF and active inequality bounds return NaN estimates
with a separate status; they never overwrite the optimization status. Status
codes are `UncertaintyStatus.available` (0), `not_converged` (1),
`insufficient_dof` (2), `singular_matrix` (3), `active_bound` (4), `invalid_input`
(5). Inspect both status arrays before interpreting error bars.

```python
result = batchfit.fit(x, y, p0=p0, bounds=bounds, device="auto", uncertainties=True)
valid = result.uncertainty_status == batchfit.UncertaintyStatus.available
lower = result.params[valid] - result.stderr[valid]
upper = result.params[valid] + result.stderr[valid]
```

Metal computes covariance on GPU in a separate optional pass before command
completion. The output dtype matches params; output ownership/GIL rules are the
same as ordinary results. These are marginal local Gaussian errors, not a joint
confidence region or a guarantee for misspecified/poorly identified models.
NUMERICS.md reports the 10,000-realization coverage test for every model/backend.

## Copies, ownership and threads

The CPU path borrows aligned, native-endian, C-contiguous arrays with the common
float32/float64 dtype. `x`, `y` and full `[N,P]` p0 are not copied in that case;
private pybind overloads prohibit force-casting. Strided/Fortran/unaligned arrays,
dtype/endian conversions, and broadcast initials require preparation copies.
Readonly inputs are supported and never mutated.

Metal necessarily fills the existing shared-buffer layout: observations transpose
from fit-major to point-major, with float conversion fused into that pass when
needed. That required backend layout copy is not advertised as zero-copy. Input
preparation avoids an additional host copy for already suitable arrays. Small
GPU output vectors are copied out so keeping a result does not pin the large
transposed observation buffers.

NumPy outputs use a native capsule owner for stable result-vector storage. Views
survive dropping the tuple, other outputs and original inputs. They are writable
and independent of later fits. Input array owners remain alive while the GIL is
released, but callers must **not mutate or resize them during a call**.

The GIL is released around initialization, CPU fitting, Metal preparation/dispatch
and waiting. Native fitting calls serialize through a mutex and share a two-worker
CPU pool / lazily initialized Metal context. This bounds resources for concurrent
Python calls; it is not a promise of parallel fits from multiple Python threads.
Other Python work can run during either backend. No Python callbacks run on native
workers. Public and extension `.pyi` stubs plus `py.typed` are included in wheels.

## Build and verification

Native CMake builds keep `BATCHFIT_BUILD_PYTHON=OFF` unless requested. Packaging
turns it on, disables native test/benchmark targets for the wheel, links the same
CPU/Metal library, and installs `_core` under `batchfit`. Build tools/dependencies
are pinned; CMake/Ninja are explicitly pinned even though scikit-build-core can
otherwise inject unpinned tool versions. Native ABI/architecture follows the
local interpreter/toolchain; this turn verified CPython 3.9.6 on macOS arm64,
not a cross-platform wheel matrix.

```sh
sh scripts/check_python.sh
# Explicit CPU-only editable build (also tested here):
uv pip install --reinstall-package batchfit -e . \
  -Cbuild-dir=build/python-cpu-check -Ccmake.define.BATCHFIT_METAL=OFF
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
# Restore/verify the normal macOS Metal-enabled editable package:
sh scripts/check_python.sh
```

The check script uses preinstalled dev dependencies and `--no-build-isolation`
so its persistent compile database refers to stable `.venv` headers, not an
already-deleted isolated build environment. Ordinary isolated `uv pip install
-e .` was also verified. Native binding clang-tidy uses that compile database.
`scripts/check.sh` includes the Python gate after the existing C++ checks.

Pytest compares **1,000 noisy fits per model** against independent NumPy model
functions and `scipy.optimize.curve_fit`: CPU float64, CPU float32 and Metal
float32, **12,000 batchfit fits / 8,000 SciPy fits**. Stored float32 inputs get
their own float64 SciPy references; no precision mismatch is hidden. SNR is 30,
M=129, fixed seeds, positive resolved peaks, separated paired components and
fixed non-truth initial guesses. Every row must converge; no failures are dropped.

Parameter tolerances are `2e-7*(1+abs(SciPy))` for float64 and
`2e-4*(1+abs(SciPy))` for float32; objectives are checked separately. Tests also
cover automatic guesses, pointer identity/no-forcecast, strided/unaligned/endian
inputs, result lifetime, dtype/shape failures, row isolation, fixed bounds, both
GIL-release paths, concurrent calls, explicit unavailable Metal, and shipped
typing files. Metal is required on the macOS gate, not silently skipped; the
deliberate CPU-only build reports its expected Metal skips.

This does not establish SciPy agreement for every ill-conditioned blend, SNR or
parameter scale. NUMERICS.md's float32 limits still apply. Normalization and
advanced numerical options are not silently introduced into this minimal API;
callers choose appropriate units/bounds. Weights and per-fit grids
are not part of this interface. Covariance/standard errors are opt-in as described
above, with full assumptions and coverage in NUMERICS.md.

### Recorded turn-12 verification

- Full `scripts/check.sh` passed: 23 CPU and 32 Metal CTests, existing numerical/
  profiling/pipeline regressions, native formatting/tidy, editable installation,
  and **46 Python tests**. Every one of the 12,000 batchfit/SciPy comparisons
  converged and passed its stated parameter/objective tolerance.
- The explicit `BATCHFIT_METAL=OFF` extension passed **41 tests**, with five
  expected Metal-only skips; the Metal-enabled install was restored afterward.
- The Python suite also passed with `PYTHONMALLOC=debug`; both CPU and Metal GIL
  tests and concurrent-call/lifetime tests passed. Python document diagnostics
  report zero errors/warnings. Native indexed diagnostics remain partial without
  a root compile-database marker; actual compiler and binding clang-tidy pass.
- An isolated ordinary editable build and a standalone wheel were built with uv.
  The wheel contains `_core`, public/private stubs and `py.typed`, and CPU/Metal
  fitting passed without editable import hooks. Temporary wheel/extraction output
  is removed; the requested editable package remains installed in `.venv`.

Turn 13 adds optional uncertainty estimation and all-denominator coverage checks:
the combined Python suite is now **73 passing tests**, with CPU-only builds
reporting 59 passes / 14 expected Metal-only skips. See NUMERICS.md for the
per-model/device coverage table, failure diagnostics and sanitizer results.
