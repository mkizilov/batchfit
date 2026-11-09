import gc
import importlib.metadata
import importlib.resources
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.typing import DTypeLike

import batchfit
import batchfit._core as native  # pyright: ignore[reportMissingModuleSource]


def data(dtype: DTypeLike=np.float64, n=8):
    x = np.linspace(-4, 4, 129, dtype=dtype)
    y = np.tile(2 / (1 + ((x - .2) / .7) ** 2), (n, 1))
    initial = np.tile(np.array([1.8, 0, .8], dtype=dtype), (n, 1))
    return x, y, initial


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_contiguous_inputs_are_borrowed(monkeypatch, dtype):
    x, y, initial = data(dtype)
    original = native.fit
    pointers = (x.ctypes.data, y.ctypes.data, initial.ctypes.data)

    def inspect(x_arg, y_arg, p0_arg, *rest):
        assert (x_arg.ctypes.data, y_arg.ctypes.data, p0_arg.ctypes.data) == pointers
        return original(x_arg, y_arg, p0_arg, *rest)

    monkeypatch.setattr(native, "fit", inspect)
    before = x.copy(), y.copy(), initial.copy()
    result = batchfit.fit(x, y, p0=initial, device="cpu")
    for array, snapshot in zip((x, y, initial), before):
        np.testing.assert_array_equal(array, snapshot)
    assert result.status.dtype == np.int32
    assert result.iterations.dtype == np.uint32
    assert np.all(result.status == 0)
    assert result.params.flags.writeable and result.chi2.flags.writeable


def test_strided_unaligned_and_non_native_arrays():
    x, y, initial = data()
    expected = batchfit.fit(x, y, p0=initial, device="cpu")
    strided_x = np.repeat(x, 2)[::2]
    strided_y = np.repeat(y, 2, axis=1)[:, ::2]
    unaligned = np.ndarray(y.shape, dtype=np.float64, buffer=bytearray(y.nbytes + 1), offset=1)
    unaligned[:] = y
    for target in (strided_y, unaligned, y.astype(">f8"), np.asfortranarray(y)):
        result = batchfit.fit(strided_x, target, p0=initial, device="cpu")
        np.testing.assert_allclose(result.params, expected.params, rtol=0, atol=0)


def test_readonly_input_and_output_lifetimes():
    x, y, initial = data()
    x.flags.writeable = y.flags.writeable = initial.flags.writeable = False
    refs = weakref.ref(x), weakref.ref(y), weakref.ref(initial)
    result = batchfit.fit(x, y, p0=initial, device="cpu")
    view = result.params[:, 0]
    del x, y, initial, result
    gc.collect()
    assert all(ref() is None for ref in refs)
    np.testing.assert_allclose(view, 2, rtol=1e-8)


def test_single_empty_and_tuple_outputs():
    x, y, initial = data(n=1)
    params, status, iterations, chi2 = batchfit.fit(x.tolist(), y[0].tolist(), p0=initial[0], device="cpu")
    assert params.shape == (1, 3)
    assert status.shape == iterations.shape == chi2.shape == (1,)
    result = batchfit.fit(x, np.empty((0, len(x))), device="cpu")
    assert result.params.shape == (0, 3)
    assert result.chi2.shape == (0,)


def test_row_failure_isolation_and_fixed_bounds():
    x, y, initial = data(n=5)
    y[1, 10] = np.nan
    y[2, 10] = np.inf
    y[3] = 0
    result = batchfit.fit(x, y, p0=initial, device="cpu")
    np.testing.assert_array_equal(result.status, [0, 3, 3, 4, 0])
    constrained = batchfit.fit(x, y[[0]], p0=initial[0], bounds=([0, .2, .7], [np.inf, .2, .7]), device="cpu")
    np.testing.assert_allclose(constrained.params[0], [2, .2, .7], atol=1e-8)


@pytest.mark.parametrize("change,error", [({"model": "missing"}, ValueError), ({"device": "gpu"}, ValueError), ({"p0": [1, 2]}, ValueError), ({"bounds": ([0, 0], [1, 1])}, ValueError), ({"bounds": (3, 2)}, ValueError), ({"bounds": (np.nan, np.inf)}, ValueError)])
def test_invalid_options(change, error):
    x, y, _ = data()
    with pytest.raises(error):
        batchfit.fit(x, y, **{"device":"cpu",**change})


@pytest.mark.parametrize("x,y,error", [(np.zeros((2, 2)), np.zeros((2, 2)), ValueError), ([], [], ValueError), ([1, 2], [[1, 2, 3]], ValueError), ([1j, 2j], [[1, 2]], TypeError), ([1, 2], [["a", "b"]], TypeError)])
def test_invalid_arrays(x, y, error):
    with pytest.raises(error):
        batchfit.fit(x, y, device="cpu")


def test_auto_policy(monkeypatch):
    original = native.fit
    selected = []

    def inspect(x, y, p0, lower, upper, model, metal):
        selected.append(metal)
        return original(x, y, p0, lower, upper, model, False)

    monkeypatch.setattr(native, "fit", inspect)
    monkeypatch.setattr(batchfit, "metal_available", lambda: True)
    for dtype in (np.float64, np.float32):
        x, y, initial = data(dtype)
        batchfit.fit(x, y, p0=initial)
    assert selected == [False, True]


def test_explicit_metal_contract():
    x, y, initial = data()
    if not batchfit.metal_available():
        with pytest.raises(RuntimeError, match="Metal.*unavailable"):
            batchfit.fit(x, y, p0=initial, device="metal")
    else:
        result = batchfit.fit(x, y, p0=initial, device="metal")
        assert result.params.dtype == np.float32
        assert np.all(result.status == 0)
        np.testing.assert_allclose(result.params, np.tile([2, .2, .7], (y.shape[0], 1)), atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_gil_is_released(device):
    if device == "metal" and not batchfit.metal_available():
        pytest.skip("Metal is unavailable in this build/device")
    x, y, initial = data(n=20000)
    ready, proceed, progressed = threading.Event(), threading.Event(), threading.Event()

    def python_work():
        ready.set()
        proceed.wait()
        progressed.set()

    worker = threading.Thread(target=python_work)
    worker.start()
    ready.wait()
    old = sys.getswitchinterval()
    try:
        sys.setswitchinterval(10)
        proceed.set()
        batchfit.fit(x, y, p0=initial, device=device)
        assert progressed.is_set(), "Python thread could not run during native fitting"
    finally:
        sys.setswitchinterval(old)
        proceed.set()
        worker.join(timeout=5)
    assert not worker.is_alive()


def test_type_stubs_are_shipped():
    package = importlib.resources.files("batchfit")
    assert package.joinpath("py.typed").is_file()
    assert "def fit(" in package.joinpath("__init__.pyi").read_text()


def test_distribution_license_is_shipped():
    distribution = importlib.metadata.distribution("batchfit")
    assert distribution.metadata["License-Expression"] == "MIT"
    license_files = distribution.metadata.get_all("License-File", [])
    assert "LICENSE" in license_files
    assert "third_party/licenses/NOTICES.txt" in license_files
    license_path = next(path for path in distribution.files or ()
                        if str(path).endswith(".dist-info/licenses/LICENSE"))
    text = license_path.read_text()
    assert text.startswith("MIT License\n\nCopyright (c) 2026 Mykyta Kizilov\n")
    assert "Permission is hereby granted, free of charge" in text
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in text


def test_native_entry_never_forcecasts():
    x, y, initial = data()
    lower, upper = np.full(3, -np.inf), np.full(3, np.inf)
    with pytest.raises(TypeError):
        native.fit(x[::2], y[:, ::2], initial, lower, upper, "lorentzian", False)
    with pytest.raises(ValueError, match="dtype"):
        native.fit(x, y, initial.astype(np.float32), lower, upper, "lorentzian", False)


def test_unrepresentable_float32_bounds_rejected():
    x, y, initial = data(np.float32)
    with pytest.raises(ValueError, match="representable"):
        batchfit.fit(x, y, p0=initial, bounds=(-np.inf, 1e300), device="cpu")


def test_concurrent_calls_do_not_deadlock():
    x, y, initial = data(n=256)
    with ThreadPoolExecutor(max_workers=2) as callers:
        futures = [callers.submit(batchfit.fit, x, y, p0=initial, device="cpu") for _ in range(2)]
        for future in futures:
            assert np.all(future.result(timeout=10).status == 0)


def test_unresolved_peak_status_is_exposed():
    x = np.linspace(-1, 1, 129)
    y = 2 / (1 + (x / 1e-4) ** 2)
    result = batchfit.fit(x, y, p0=[2, 0, 1e-4], device="cpu")
    assert result.status[0] == batchfit.FitStatus.unresolved_peak
    assert np.isfinite(result.chi2[0])
