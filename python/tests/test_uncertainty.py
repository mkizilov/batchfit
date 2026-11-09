import gc
from typing import Any, cast

import numpy as np
import pytest
from scipy.optimize import curve_fit

import batchfit
from .conftest import MODELS, comparison_data


TRUTH = {
    "gaussian": [2, .2, .7],
    "lorentzian": [2, .2, .7],
    "pseudo_voigt": [2, .2, 1.4, .5],
    "two_lorentzian": [2, -1.5, .5, 1.7, 1.5, .6, .25, .01],
}
INITIAL = {
    "gaussian": [1.8, 0, .8],
    "lorentzian": [1.8, 0, .8],
    "pseudo_voigt": [1.8, 0, 1.6, .45],
    "two_lorentzian": [1.8, -1.3, .55, 1.5, 1.3, .7, .2, 0],
}


def domain(name, dtype):
    p = len(TRUTH[name])
    lower, upper = np.full(p, -np.inf, dtype=dtype), np.full(p, np.inf, dtype=dtype)
    lower[0], lower[2] = 0, 1e-4
    lower[1], upper[1] = -5, 5
    if p == 4:
        lower[3], upper[3] = 0, 1
    if p == 8:
        lower[3], lower[5], lower[4], upper[4] = 0, 1e-4, -5, 5
    return lower, upper


@pytest.mark.parametrize("device,dtype", [("cpu", "float64"), ("cpu", "float32"), ("metal", "float32")])
def test_covariance_matches_scipy_at_solution(model_name, device, dtype):
    if device == "metal" and not batchfit.metal_available():
        pytest.skip("Metal is unavailable")
    x, y, initial, bounds, _, _ = comparison_data(model_name, dtype)
    basic = batchfit.fit(x, y[:8], model_name, initial, bounds, device)
    result = batchfit.fit(x, y[:8], model_name, initial, bounds, device, uncertainties=True)
    assert type(basic) is batchfit.FitResult and len(basic) == 4
    assert not hasattr(basic, "covariance")
    assert isinstance(result, batchfit.FitResultWithUncertainties) and len(result) == 7
    for old, new in zip(basic, result[:4]):
        np.testing.assert_array_equal(old, new)
    p = len(initial)
    assert result.covariance.shape == (8, p, p) and result.stderr.shape == (8, p)
    assert result.covariance.dtype == result.stderr.dtype == result.params.dtype
    assert result.uncertainty_status.dtype == np.int32
    assert np.all(result.uncertainty_status == batchfit.UncertaintyStatus.available)
    for i, row in enumerate(y[:8]):
        _, expected = curve_fit(MODELS[model_name], x, row, p0=initial, bounds=bounds, max_nfev=1000, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        relative = 2e-5 if dtype == "float64" else 3e-3
        np.testing.assert_allclose(result.covariance[i], expected, rtol=relative, atol=1e-10 if dtype == "float64" else 1e-8)
    np.testing.assert_array_equal(result.covariance, result.covariance.transpose(0, 2, 1))
    np.testing.assert_allclose(result.stderr**2, np.diagonal(result.covariance, axis1=1, axis2=2), rtol=2e-7, atol=1e-16)
    view = result.stderr[:, 0]
    del result, basic
    gc.collect()
    assert np.all(np.isfinite(view))


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_uncertainty_invalidity_and_fixed_parameters(device):
    if device == "metal" and not batchfit.metal_available():
        pytest.skip("Metal is unavailable")
    x = np.linspace(-5, 5, 129)
    y = MODELS["lorentzian"](x, *TRUTH["lorentzian"])
    initial = TRUTH["lorentzian"]
    failed = y.copy()
    failed[17] = np.nan
    result = batchfit.fit(x, failed, p0=initial, device=device, uncertainties=True)
    assert result.status[0] == batchfit.FitStatus.invalid_input
    assert result.uncertainty_status[0] == batchfit.UncertaintyStatus.not_converged
    assert np.all(np.isnan(result.covariance)) and np.all(np.isnan(result.stderr))
    small_x = np.array([-1., 0., 1.])
    small_y = MODELS["lorentzian"](small_x, 2, 0, 2)
    small = batchfit.fit(small_x, small_y, p0=[2, 0, 2], device=device, uncertainties=True)
    assert small.uncertainty_status[0] == batchfit.UncertaintyStatus.insufficient_dof
    singular = batchfit.fit(np.zeros(20), np.full(20, 2.), p0=[2, 0, 1], device=device, uncertainties=True)
    assert singular.status[0] == batchfit.FitStatus.converged
    assert singular.uncertainty_status[0] == batchfit.UncertaintyStatus.singular_matrix
    boundary = batchfit.fit(x, y, p0=initial, bounds=([2, -5, .01], [np.inf, 5, 5]), device=device, uncertainties=True)
    assert boundary.uncertainty_status[0] == batchfit.UncertaintyStatus.active_bound
    noisy = y + np.random.default_rng(101).normal(0, .03, len(x))
    fixed = batchfit.fit(x, noisy, p0=initial, bounds=([0, .2, .7], [np.inf, .2, .7]), device=device, uncertainties=True)
    assert fixed.uncertainty_status[0] == batchfit.UncertaintyStatus.available
    np.testing.assert_array_equal(fixed.covariance[0, 1:], 0)
    np.testing.assert_array_equal(fixed.stderr[0, 1:], 0)
    shape = MODELS["lorentzian"](x, 1, .2, .7)
    expected = fixed.chi2[0] / (len(x) - 1) / np.sum(shape**2)
    np.testing.assert_allclose(fixed.covariance[0, 0, 0], expected, rtol=1e-5, atol=1e-12)
    empty = batchfit.fit(x, np.empty((0, len(x))), device=device, uncertainties=True)
    assert empty.covariance.shape == (0, 3, 3) and empty.stderr.shape == (0, 3)


@pytest.mark.parametrize("device,dtype", [("cpu", "float64"), ("cpu", "float32"), ("metal", "float32")])
def test_monte_carlo_10000_coverage(model_name, device, dtype):
    if device == "metal" and not batchfit.metal_available():
        pytest.skip("Metal is unavailable")
    n = 10000
    random = np.random.default_rng(20261110 + list(MODELS).index(model_name))
    truth = np.asarray(TRUTH[model_name])
    x = np.linspace(-5, 5, 129)
    clean = MODELS[model_name](x, *truth)
    y = (clean + random.normal(0, .03, (n, len(x)))).astype(dtype)
    result = batchfit.fit(x.astype(dtype), y, model_name, INITIAL[model_name], domain(model_name, dtype), device, uncertainties=True)
    available = (result.status == batchfit.FitStatus.converged) & (result.uncertainty_status == batchfit.UncertaintyStatus.available)
    inside = available[:, None] & np.isfinite(result.stderr) & (np.abs(result.params - truth) <= result.stderr)
    coverage = np.count_nonzero(inside, axis=0) / n  # Failed/unavailable rows are misses.
    ratio = np.std(result.params.astype(np.float64), axis=0, ddof=1) / np.mean(result.stderr.astype(np.float64), axis=0)
    print(f"COVERAGE {model_name} {device} {dtype} available={np.count_nonzero(available)}/{n} fractions={coverage.tolist()} empirical_std_over_mean_se={ratio.tolist()}")
    assert np.all(available), "No failed fits or unavailable uncertainties may be discarded"
    assert np.all(np.abs(coverage - 0.682689492137) <= .02), coverage


def test_uncertainty_flag_is_explicit_boolean():
    with pytest.raises(TypeError, match="boolean"):
        batchfit.fit([0, 1, 2], [1, 2, 1], uncertainties=cast(Any, "yes"))
