"""Independent stdlib NPY/CLI checks; all fixtures stay in the repository."""

import ast
import math
import pathlib
import struct
import subprocess
import sys
import tempfile

root = pathlib.Path(__file__).resolve().parents[1]
binary = (root / sys.argv[1]).resolve()


def write(path, shape, values, dtype="f8"):
    text = "{'descr': '<%s', 'fortran_order': False, 'shape': %r, }" % (dtype, shape)
    text += " " * ((64 - (10 + len(text) + 1) % 64) % 64) + "\n"
    code = {"f8": "d", "f4": "f"}[dtype]
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(text)) + text.encode() + struct.pack("<" + str(len(values)) + code, *values))


def read(path, dtype, shape):
    data = path.read_bytes()
    assert data[:8] == b"\x93NUMPY\x01\x00"
    offset = 10 + struct.unpack_from("<H", data, 8)[0]
    header = ast.literal_eval(data[10:offset].decode().strip())
    assert header == {"descr": "<" + dtype, "fortran_order": False, "shape": shape}
    count = math.prod(shape)
    assert len(data) == offset + count * 4
    return struct.unpack_from("<" + str(count) + {"f4": "f", "u4": "I"}[dtype], data, offset)


def invoke(args, success=True):
    result = subprocess.run([str(binary), *map(str, args)], cwd=root, text=True, capture_output=True)
    assert (result.returncode == 0) == success, (result.returncode, result.stdout, result.stderr)


with tempfile.TemporaryDirectory(prefix="pipeline-cli-", dir=root / ".agent-local") as name:
    directory = pathlib.Path(name)
    n, m = 71, 65
    x = [-5 + 10 * j / (m - 1) for j in range(m)]
    truth = [(2 + i / n, -0.2 + i / n * 0.4, 0.7 + i / n * 0.1) for i in range(n)]
    y = [a * math.exp(-0.5 * ((value - center) / width) ** 2) for a, center, width in truth for value in x]
    write(directory / "x.npy", (m,), x)
    write(directory / "y.npy", (n, m), y)
    write(directory / "initial.npy", (n, 3), [value for a, center, width in truth for value in (a * 0.9, center + 0.1, width * 1.1)])
    args = ["--x", directory / "x.npy", "--y", directory / "y.npy", "--chunk-fits", 17]
    for automatic in (False, True):
        output = directory / ("auto" if automatic else "explicit")
        invoke(args + ["--output", output] + ([] if automatic else ["--initial", directory / "initial.npy"]))
        parameters = read(output / "parameters.npy", "f4", (n, 3))
        assert all(abs(actual - expected) <= 2e-4 + 2e-4 * abs(expected) for actual, expected in zip(parameters, sum((list(row) for row in truth), [])))
        assert set(read(output / "status.npy", "u4", (n,))) == {0}
        assert all(value <= 100 for value in read(output / "iterations.npy", "u4", (n,)))
        assert max(read(output / "chi_squared.npy", "f4", (n,))) < 1e-6
        snapshot = {path.name: path.read_bytes() for path in output.iterdir()}
        invoke(args + ["--output", output], False)
        assert snapshot == {path.name: path.read_bytes() for path in output.iterdir()}
    (directory / "truncated.npy").write_bytes((directory / "y.npy").read_bytes()[:-4])
    invoke(["--x", directory / "x.npy", "--y", directory / "truncated.npy", "--output", directory / "bad"], False)
    assert not (directory / "bad").exists()
    write(directory / "wrong_initial.npy", (n, 4), [1.0] * (n * 4))
    invoke(args + ["--initial", directory / "wrong_initial.npy", "--output", directory / "rollback"], False)
    assert not (directory / "rollback").exists()
    (directory / "existing.json").write_text("preserve me")
    invoke(args + ["--timeline", directory / "existing.json", "--output", directory / "timeline_rollback"], False)
    assert (directory / "existing.json").read_text() == "preserve me"
    assert not (directory / "timeline_rollback").exists()
    write(directory / "empty.npy", (0, m), [])
    invoke(["--x", directory / "x.npy", "--y", directory / "empty.npy", "--output", directory / "empty"])
    assert read(directory / "empty" / "parameters.npy", "f4", (0, 3)) == ()
print("Pipeline CLI: float64 conversion, explicit/automatic guesses, typed NPY outputs, empty input, overwrite refusal and rollback passed")
