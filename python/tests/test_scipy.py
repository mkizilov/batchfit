import numpy as np
import pytest

import batchfit
from .conftest import MODELS, comparison_data


@pytest.mark.parametrize("device,dtype", [("cpu", "float64"), ("cpu", "float32"), ("metal", "float32")])
def test_thousand_fits_per_model_agree_with_scipy(model_name, device, dtype):
    if device == "metal" and not batchfit.metal_available():
        pytest.skip("this build/device has no Metal; CPU comparison remains required")
    x, y, initial, bounds, expected, chi = comparison_data(model_name, dtype)
    result = batchfit.fit(x, y, model=model_name, p0=initial, bounds=bounds, device=device)
    assert result.params.shape == expected.shape
    assert result.params.dtype == np.dtype(dtype)
    assert result.status.shape == result.iterations.shape == result.chi2.shape == (1000,)
    assert np.all(result.status == batchfit.FitStatus.converged)
    assert np.all(result.iterations <= 100)
    tolerance = 2e-7 if dtype == "float64" else 2e-4
    np.testing.assert_allclose(result.params, expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(result.chi2, chi, atol=1e-8 if dtype == "float64" else 1e-6, rtol=2e-5)
    scaled = np.max(np.abs(result.params - expected) / (tolerance * (1 + np.abs(expected))))
    print(f"{model_name} {device} {dtype}: 1000/1000 converged, max tolerance fraction={scaled:.6g}")


@pytest.mark.parametrize("device,dtype", [("cpu", "float64"), ("auto", "float32")])
def test_automatic_initial_guesses(model_name, device, dtype):
    x, y, _, bounds, _, _ = comparison_data(model_name, dtype)
    result = batchfit.fit(x, y[:16], model=model_name, bounds=bounds, device=device)
    assert np.all(result.status == batchfit.FitStatus.converged)
    assert np.all(np.isfinite(result.params))
    for row, parameters, cost in zip(y[:16], result.params, result.chi2):
        np.testing.assert_allclose(np.sum((row - MODELS[model_name](x, *parameters)) ** 2), cost, rtol=1e-12 if dtype=="float64" else 2e-5, atol=1e-12 if dtype=="float64" else 1e-6)
