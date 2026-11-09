from enum import IntEnum
from typing import Literal, NamedTuple, Union, overload
import numpy as np
from numpy.typing import ArrayLike, NDArray

__version__: str
_Model = Literal["gaussian", "lorentzian", "pseudo_voigt", "two_lorentzian"]
_Device = Literal["auto", "cpu", "metal"]
_FloatingArray = Union[NDArray[np.float32], NDArray[np.float64]]

class FitStatus(IntEnum):
    converged = 0
    max_iterations = 1
    singular_matrix = 2
    invalid_input = 3
    degenerate_data = 4
    unresolved_peak = 5

class FitResult(NamedTuple):
    params: _FloatingArray
    status: NDArray[np.int32]
    iterations: NDArray[np.uint32]
    chi2: _FloatingArray

class UncertaintyStatus(IntEnum):
    available = 0
    not_converged = 1
    insufficient_dof = 2
    singular_matrix = 3
    active_bound = 4
    invalid_input = 5

class FitResultWithUncertainties(NamedTuple):
    params: _FloatingArray
    status: NDArray[np.int32]
    iterations: NDArray[np.uint32]
    chi2: _FloatingArray
    covariance: _FloatingArray
    stderr: _FloatingArray
    uncertainty_status: NDArray[np.int32]

def metal_available() -> bool: ...
@overload
def fit(x: ArrayLike, y: ArrayLike, model: _Model = "lorentzian",
        p0: Union[ArrayLike, None] = None,
        bounds: Union[tuple[ArrayLike, ArrayLike], None] = None,
        device: _Device = "auto", *, uncertainties: Literal[False] = False) -> FitResult: ...
@overload
def fit(x: ArrayLike, y: ArrayLike, model: _Model = "lorentzian",
        p0: Union[ArrayLike, None] = None, bounds: Union[tuple[ArrayLike, ArrayLike], None] = None,
        device: _Device = "auto", *, uncertainties: Literal[True]) -> FitResultWithUncertainties: ...
@overload
def fit(x: ArrayLike, y: ArrayLike, model: _Model = "lorentzian",
        p0: Union[ArrayLike, None] = None, bounds: Union[tuple[ArrayLike, ArrayLike], None] = None,
        device: _Device = "auto", *, uncertainties: bool) -> Union[FitResult,FitResultWithUncertainties]: ...
