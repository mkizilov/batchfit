from typing import Any
import numpy as np
from numpy.typing import NDArray

__version__: str
def metal_available() -> bool: ...
def fit(x: NDArray[Any], y: NDArray[Any], p0: NDArray[Any] | None,
        lower: NDArray[Any], upper: NDArray[Any], model: str,
        metal: bool) -> tuple[NDArray[Any], NDArray[np.int32], NDArray[np.uint32], NDArray[Any]]: ...
