"""Batched Gaussian, Lorentzian, pseudo-Voigt and paired-Lorentzian fitting."""

from enum import IntEnum
from typing import NamedTuple

import numpy as np

# The compiled extension's typing contract lives in _core.pyi.
from . import _core  # pyright: ignore[reportMissingModuleSource]

__version__ = _core.__version__
__all__ = ["fit", "FitResult", "FitResultWithUncertainties", "FitStatus", "UncertaintyStatus", "metal_available", "__version__"]
_PARAMETERS = {"gaussian": 3, "lorentzian": 3, "pseudo_voigt": 4, "two_lorentzian": 8}


class FitStatus(IntEnum):
    converged = 0
    max_iterations = 1
    singular_matrix = 2
    invalid_input = 3
    degenerate_data = 4
    unresolved_peak = 5


class FitResult(NamedTuple):
    params: np.ndarray
    status: np.ndarray
    iterations: np.ndarray
    chi2: np.ndarray


class UncertaintyStatus(IntEnum):
    available = 0
    not_converged = 1
    insufficient_dof = 2
    singular_matrix = 3
    active_bound = 4
    invalid_input = 5


class FitResultWithUncertainties(NamedTuple):
    params: np.ndarray
    status: np.ndarray
    iterations: np.ndarray
    chi2: np.ndarray
    covariance: np.ndarray
    stderr: np.ndarray
    uncertainty_status: np.ndarray


def metal_available():
    """Whether this build has Metal and a supported device (no shader compilation)."""
    return _core.metal_available()


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    return array


def fit(x, y, model="lorentzian", p0=None, bounds=None, device="auto", *, uncertainties=False):
    """Fit shared x[M] and y[N,M]; return params[N,P], status/iterations/chi2[N].

    A single y[M] is accepted as N=1. CPU preserves float32/float64 precision;
    auto uses Metal for float32 when available, CPU otherwise. Explicit Metal
    uses float32 and necessarily transposes observations into shared GPU storage.
    Native aligned C-contiguous arrays are borrowed by CPU without copying.
    Do not mutate/resize inputs until this synchronous call returns.
    """
    if model not in _PARAMETERS:
        raise ValueError(f"unknown model {model!r}; choose one of {tuple(_PARAMETERS)}")
    if device not in ("auto", "cpu", "metal"):
        raise ValueError("device must be 'auto', 'cpu', or 'metal'")
    if not isinstance(uncertainties,(bool,np.bool_)):
        raise TypeError("uncertainties must be a boolean")
    x, y = _real_array(x, "x"), _real_array(y, "y")
    if x.ndim != 1 or not x.size:
        raise ValueError("x must have nonempty shape [M]")
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.ndim != 2 or y.shape[1] != x.size:
        raise ValueError("y must have shape [N,M] matching x[M]")
    dtype = np.dtype("float32" if np.result_type(x.dtype, y.dtype) == np.dtype("float32") else "float64")
    metal = device == "metal" or (device == "auto" and dtype == np.dtype("float32") and metal_available())
    x = np.require(x, dtype=dtype, requirements=["C", "A"])
    y = np.require(y, dtype=dtype, requirements=["C", "A"])
    n, p = y.shape[0], _PARAMETERS[model]
    if p0 is not None:
        p0 = _real_array(p0, "p0")
        if p0.shape == (p,):
            p0 = np.broadcast_to(p0, (n, p))
        if p0.shape != (n, p):
            raise ValueError(f"p0 must have shape [{p}] or [{n},{p}]")
        p0 = np.require(p0, dtype=dtype, requirements=["C", "A"])
    if bounds is None:
        lower, upper = np.full(p, -np.inf, dtype=dtype), np.full(p, np.inf, dtype=dtype)
    else:
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError("bounds must be (lower, upper), each scalar or [P]")
        lower, upper = (_real_array(value, "bounds") for value in bounds)
        if lower.shape not in ((), (p,)) or upper.shape not in ((), (p,)):
            raise ValueError(f"each bound must be scalar or have shape [{p}]")
        maximum = np.finfo(np.float32 if metal else dtype).max
        if any(np.any(np.isfinite(bound) & (np.abs(bound) > maximum)) for bound in (lower, upper)):
            raise ValueError("finite bounds must be representable in the fitting dtype")
        lower = np.require(np.broadcast_to(lower, (p,)), dtype=dtype, requirements=["C", "A"])
        upper = np.require(np.broadcast_to(upper, (p,)), dtype=dtype, requirements=["C", "A"])
    if uncertainties:
        return FitResultWithUncertainties(*_core.fit(x,y,p0,lower,upper,model,metal,True))
    return FitResult(*_core.fit(x, y, p0, lower, upper, model, metal))
