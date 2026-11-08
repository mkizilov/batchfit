"""Exercise the CLI and inspect its NPY files independently using Python's stdlib."""

import ast
import math
import pathlib
import struct
import subprocess
import sys
import tempfile

binary = pathlib.Path(sys.argv[1]).resolve()
root = pathlib.Path(__file__).resolve().parents[1]


def read_npy(path):
    with path.open("rb") as stream:
        assert stream.read(8) == b"\x93NUMPY\x01\x00"
        length = struct.unpack("<H", stream.read(2))[0]
        header = stream.read(length)
        assert (10 + length) % 16 == 0 and header.endswith(b"\n")
        metadata = ast.literal_eval(header.decode("ascii"))
        assert metadata["descr"] == "<f4" and metadata["fortran_order"] is False
        payload = stream.read()
        assert len(payload) == math.prod(metadata["shape"]) * 4
        return metadata["shape"], struct.unpack("<" + "f" * (len(payload) // 4), payload)


def run(*args, success=True):
    result = subprocess.run([str(binary), *map(str, args)], capture_output=True, text=True)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result


with tempfile.TemporaryDirectory(prefix="generator-cli-", dir=root / ".agent-local") as temporary:
    scratch = pathlib.Path(temporary)
    for model, count in (("gaussian", 3), ("lorentzian", 3), ("pseudo_voigt", 4), ("two_lorentzian", 8)):
        for noise in ("none", "gaussian", "poisson"):
            output = scratch / (model + "-" + noise)
            run("--output", output, "--fits", 7, "--points", 9, "--chunk-fits", 3,
                "--model", model, "--noise", noise, "--snr", 10, "--seed", 123)
            assert read_npy(output / "x.npy")[0] == (9,)
            assert read_npy(output / "y.npy")[0] == (7, 9)
            assert read_npy(output / "true_params.npy")[0] == (7, count)
    # Range options intentionally precede --model: parsing must not depend on flag order.
    fixed = scratch / "fixed"
    args = ["--range", "amplitude=4:4", "--range", "center=1:1", "--range", "sigma=2:2",
            "--model", "gaussian", "--noise", "none", "--fits", 7, "--points", 9]
    run("--output", fixed, *args)
    _, x = read_npy(fixed / "x.npy")
    _, y = read_npy(fixed / "y.npy")
    _, truth = read_npy(fixed / "true_params.npy")
    assert truth == (4.0, 1.0, 2.0) * 7
    assert all(abs(value - 4 * math.exp(-0.5 * ((x[j % 9] - 1) / 2) ** 2)) < 1e-6
               for j, value in enumerate(y))
    for arguments in (("--unknown", "x"), ("--fits", "-1"), ("--points", "0"),
                      ("--snr", "nan"), ("--range", "sigma=0:1"),
                      ("--range", "no_such_parameter=1:2"), ("--model", "invalid"), ("--seed",)):
        output = scratch / "must-not-exist"
        run("--output", output, *arguments, success=False)
        assert not output.exists()
    before = (fixed / "y.npy").read_bytes()
    run("--output", fixed, success=False)
    assert (fixed / "y.npy").read_bytes() == before
    run(success=False)
print("CLI/independent Python NPY checks passed")
