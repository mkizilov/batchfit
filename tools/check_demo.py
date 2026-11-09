"""Execute the full demo and its real-NPY branch using project-local IPC kernels."""
import argparse
import copy
import json
import pathlib
import tempfile

import nbformat
import numpy as np
from nbclient import NotebookClient
from traitlets.config import Config

root = pathlib.Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--write", action="store_true", help="store actual default execution outputs in the deliverable notebook")
parser.add_argument("--device", choices=("auto", "cpu", "metal"), default="auto")
args = parser.parse_args()
if args.write and args.device != "auto":
    parser.error("--write retains the default auto-device notebook; device overrides are verification-only")
path = root / "examples/hyperspectral_demo.ipynb"
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)


def execute(book, directory):
    config = Config()
    config.KernelManager.transport = "ipc"
    config.KernelManager.ip = str(directory / "k")
    config.KernelManager.connection_file = str(directory / "connection.json")
    check = nbformat.v4.new_code_cell(
        "from ipykernel.kernelapp import IPKernelApp\n"
        "app = IPKernelApp.instance()\n"
        "assert app.transport == 'ipc'\n"
        f"assert Path(app.ip).resolve().is_relative_to(Path({str(directory)!r}))\n"
    )
    book.cells.append(check)
    client = NotebookClient(book, config=config, kernel_name="python3", timeout=240, allow_errors=False,
                            extra_arguments=["--HistoryManager.hist_file=:memory:"])
    try:
        result = client.execute(cwd=str(root))
    finally:
        assert client.km is None or not client.km.has_kernel, "notebook kernel did not shut down"
    result.cells.pop()
    return result


with tempfile.TemporaryDirectory(prefix="nb-", dir=root / ".agent-local/tmp") as temporary:
    directory = pathlib.Path(temporary)
    first = directory / "synthetic"
    first.mkdir()
    book = copy.deepcopy(notebook)
    for cell in book.cells:
        if "parameters" in cell.metadata.get("tags", []):
            cell.source += f"\nDEVICE = {args.device!r}\n"
    executed = execute(book, first)
    images = 0
    summary = None
    for cell in executed.cells:
        for output in cell.get("outputs", []):
            assert output.output_type != "error"
            images += "image/png" in output.get("data", {})
            for line in output.get("text", "").splitlines():
                if line.startswith("DEMO_SUMMARY "):
                    summary = json.loads(line.removeprefix("DEMO_SUMMARY "))
    assert images >= 2 and summary is not None
    assert summary["shape"] == [512, 512, 129] and summary["submitted"] == 512 * 512
    assert summary["defects"] == summary["flagged_defects"] == 12
    assert summary["healthy_converged"] >= .999 * summary["healthy_pixels"]
    print(json.dumps(summary, indent=2))
    second = directory / "real"
    second.mkdir()
    x = np.linspace(-5, 5, 65)
    cube = np.tile(2 / (1 + ((x-.1)/.8)**2), (4, 5, 1))
    cube[0, 0] = np.nan
    np.save(second / "x.npy", x, allow_pickle=False)
    np.save(second / "cube.npy", cube, allow_pickle=False)
    selected = {tag: copy.deepcopy(cell) for cell in notebook.cells for tag in cell.metadata.get("tags", [])}
    real = nbformat.v4.new_notebook(cells=[selected["setup"], selected["fit-function"],
        nbformat.v4.new_code_cell(f"REAL_DATA_DIR = Path({str(second)!r})\nDEVICE = 'cpu'\nCHUNK_PIXELS = 7"), selected["real-data"],
        nbformat.v4.new_code_cell(
            "assert real_maps['submitted'] == 20\n"
            "assert real_maps['params'].shape == (4,5,3)\n"
            "assert real_maps['status'][0,0] != batchfit.FitStatus.converged\n"
            "assert np.count_nonzero(real_maps['status'] == batchfit.FitStatus.converged) == 19\n"
            "np.testing.assert_allclose(real_maps['params'].reshape(-1,3)[1:], np.tile([2,.1,.8],(19,1)),atol=2e-5,rtol=2e-5)\n"
            "for bad_x,bad_cube in [(real_x[:-1],real_cube), (real_x,np.zeros(real_cube.shape,dtype=np.int32)), (real_x[::-1],real_cube), (real_x,np.asfortranarray(real_cube))]:\n"
            "    try:\n"
            "        fit_cube(bad_x,bad_cube,'cpu',7)\n"
            "    except (ValueError,TypeError):\n"
            "        pass\n"
            "    else:\n"
            "        raise AssertionError('malformed cube/axis was accepted')\n"
        )])
    execute(real, second)
    if args.write:
        # Persist only the default sources; the injected IPC check and test override are not deliverables.
        for original, cell in zip(notebook.cells, executed.cells):
            cell.source = original.source
        nbformat.write(executed, path)
print(f"Notebook verified: full 512x512 fit, {images} embedded maps, real read-only NPY input/tails/errors, IPC kernel cleanup")
