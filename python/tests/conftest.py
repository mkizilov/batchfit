import functools
import os

import numpy as np
import pytest
from scipy.optimize import curve_fit

import batchfit


@pytest.fixture(scope="session", autouse=True)
def required_backend():
    if os.environ.get("BATCHFIT_REQUIRE_METAL") == "1":
        assert batchfit.metal_available(), "Metal Python tests are required on this macOS gate"


def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def lorentzian(x, amplitude, center, hwhm):
    return amplitude / (1 + ((x - center) / hwhm) ** 2)


def pseudo_voigt(x, amplitude, center, fwhm, mixing):
    z = (x - center) / fwhm
    return amplitude * ((1 - mixing) * np.exp(-4 * np.log(2) * z**2) + mixing / (1 + 4 * z**2))


def two_lorentzian(x, a1, c1, w1, a2, c2, w2, intercept, slope):
    return lorentzian(x, a1, c1, w1) + lorentzian(x, a2, c2, w2) + intercept + slope * x


MODELS = {"gaussian": gaussian, "lorentzian": lorentzian, "pseudo_voigt": pseudo_voigt, "two_lorentzian": two_lorentzian}


@functools.lru_cache(None)
def comparison_data(name, dtype):
    random = np.random.default_rng(20261109 + list(MODELS).index(name))
    n, m = 1000, 129
    x = np.linspace(-5, 5, m)
    if name == "two_lorentzian":
        truth = np.column_stack([random.uniform(1.5, 3, n), random.uniform(-2, -1.2, n), random.uniform(.4, .7, n), random.uniform(1.5, 3, n), random.uniform(1.2, 2, n), random.uniform(.4, .7, n), random.uniform(.2, .5, n), random.uniform(-.02, .02, n)])
        initial = np.array([2, -1.5, .6, 2, 1.5, .6, .3, 0])
        bounds = (np.array([0, -5, .01, 0, -5, .01, -np.inf, -np.inf]), np.array([np.inf, 5, 5, np.inf, 5, 5, np.inf, np.inf]))
    else:
        truth = np.column_stack([random.uniform(1.5, 3, n), random.uniform(-.5, .5, n), random.uniform(.5, 1, n)])
        initial = np.array([2, 0, .8])
        bounds = (np.array([0, -5, .01]), np.array([np.inf, 5, 5]))
        if name == "pseudo_voigt":
            truth[:, 2] *= 2
            truth = np.column_stack([truth, random.uniform(.2, .8, n)])
            initial = np.array([2, 0, 1.6, .5])
            bounds = (np.r_[bounds[0], 0], np.r_[bounds[1], 1])
    clean = np.stack([MODELS[name](x, *parameters) for parameters in truth])
    deviation = np.sqrt(np.mean(clean**2, axis=1)) / 30
    y = clean + random.normal(size=clean.shape) * deviation[:, None]
    x, y, initial = x.astype(dtype), y.astype(dtype), initial.astype(dtype)
    bounds = (bounds[0].astype(dtype),bounds[1].astype(dtype))
    reference = np.stack([curve_fit(MODELS[name], x, row, p0=initial, bounds=bounds, max_nfev=1000, ftol=1e-11, xtol=1e-11, gtol=1e-11)[0] for row in y])
    chi = np.array([np.sum((row.astype(np.float64) - MODELS[name](x.astype(np.float64), *parameters)) ** 2) for row, parameters in zip(y, reference)])
    return x, y, initial, bounds, reference, chi


@pytest.fixture(params=list(MODELS))
def model_name(request):
    return request.param
