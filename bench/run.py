"""Bounded-memory, measured (never extrapolated) CPU/Metal/SciPy benchmark."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import pathlib
import platform
import resource
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".agent-local/cache/matplotlib"))

import numpy as np
import scipy
from scipy.optimize import curve_fit

BACKENDS = ("scipy", "cpu_single", "cpu_multi", "metal_v1", "metal_v2_simd", "metal_v2_group", "metal_auto")
MODELS = ("gaussian", "lorentzian", "pseudo_voigt", "two_lorentzian")


def gaussian(x, a, c, w):
    return a * np.exp(-.5 * ((x-c)/w)**2)


def lorentzian(x, a, c, w):
    return a / (1 + ((x-c)/w)**2)


def pseudo_voigt(x, a, c, w, eta):
    z = (x-c)/w
    return a * ((1-eta)*np.exp(-4*np.log(2)*z*z) + eta/(1+4*z*z))


def two_lorentzian(x, a, c, w, b, d, v, intercept, slope):
    return lorentzian(x,a,c,w) + lorentzian(x,b,d,v) + intercept + slope*x


def build(directory):
    import pybind11
    subprocess.run(["cmake", "-S", str(ROOT), "-B", str(directory), "-G", "Ninja",
                    "-DCMAKE_BUILD_TYPE=Release", "-DBATCHFIT_BUILD_TESTS=OFF",
                    "-DBATCHFIT_BUILD_BENCHMARKS=ON", "-DBATCHFIT_BUILD_PYTHON=ON",
                    f"-DPython_EXECUTABLE={sys.executable}", f"-Dpybind11_DIR={pybind11.get_cmake_dir()}"], check=True)
    subprocess.run(["cmake", "--build", str(directory), "--target", "_benchmark", "--parallel", "2"], check=True)


def load_native(directory):
    candidates = list(directory.glob("_benchmark*.so")) + list(directory.glob("_benchmark*.pyd"))
    if len(candidates) != 1:
        raise RuntimeError("build the Release benchmark module with --build first")
    spec = importlib.util.spec_from_file_location("_benchmark", candidates[0])
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load native benchmark module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scipy_run(name, data, n, chunk, budget):
    allocation = time.perf_counter()
    x = data["x"].astype(np.float64)
    bounds = data["lower"].astype(float), data["upper"].astype(float)
    allocation = time.perf_counter() - allocation
    model = globals()[name]
    setup = time.perf_counter()
    curve_fit(model, x, data["y"][0].astype(float), p0=data["p0"][0].astype(float), bounds=bounds, max_nfev=1000)
    setup = time.perf_counter() - setup
    out = dict(setup_seconds=setup, allocation_seconds=allocation, prepare_seconds=0., fit_wall_seconds=0., readback_seconds=0., gpu_seconds=0., validation_seconds=0., completed=0, converged=0, status_errors=0, parameter_errors=0, cost_errors=0, chunks=0, selected=0, mean_iterations=0., shared_or_staging_bytes=0)
    start = time.perf_counter()
    stop = False
    while out["completed"] < n and not stop:
        count = min(chunk, n-out["completed"])
        prepared = time.perf_counter()
        indices = (np.arange(count) + out["completed"]) % len(data["y"])
        y = np.ascontiguousarray(data["y"][indices], dtype=float)
        p0 = np.ascontiguousarray(data["p0"][indices], dtype=float)
        out["prepare_seconds"] += time.perf_counter()-prepared
        out["shared_or_staging_bytes"] = max(out["shared_or_staging_bytes"], x.nbytes+y.nbytes+p0.nbytes)
        out["chunks"] += 1
        for i, pattern in enumerate(indices):
            if budget > 0 and out["completed"] > 0 and time.perf_counter()-start >= budget:
                stop = True
                break
            fitted = time.perf_counter()
            try:
                params, _ = curve_fit(model, x, y[i], p0=p0[i], bounds=bounds, max_nfev=1000)
                success = True
            except (RuntimeError, ValueError):
                params, success = np.full(len(p0[i]), np.nan), False
            out["fit_wall_seconds"] += time.perf_counter()-fitted
            checked = time.perf_counter()
            expected = data["parameters"][pattern].astype(float)
            out["converged"] += success
            out["status_errors"] += success != (data["status"][pattern] == 0)
            if data["status"][pattern] == 0:
                out["parameter_errors"] += not np.all(np.abs(params-expected) <= 2e-4*(1+np.abs(expected)))
                chi = np.sum((y[i]-model(x,*params))**2)
                out["cost_errors"] += not abs(chi-data["chi2"][pattern]) <= 1e-6+2e-5*abs(data["chi2"][pattern])
            out["validation_seconds"] += time.perf_counter()-checked
            out["completed"] += 1
    out["wall_seconds"] = allocation + time.perf_counter()-start-out["validation_seconds"]
    return out


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["model"],row["M"],row["N"],row["backend"]), []).append(row)
    result = []
    for samples in groups.values():
        selected = sorted(samples,key=lambda row:float(row["wall_seconds"]))[len(samples)//2].copy()
        selected["repetitions"] = len(samples)
        selected["wall_min"] = min(float(row["wall_seconds"]) for row in samples)
        selected["wall_max"] = max(float(row["wall_seconds"]) for row in samples)
        result.append(selected)
    return result


def write_csv(path, rows):
    with path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name in sorted({row["model"] for row in rows}):
        points = sorted({int(row["M"]) for row in rows if row["model"] == name})
        fig, axes = plt.subplots(1,len(points),figsize=(4.6*len(points),4.4),squeeze=False)
        phases, phase_axes = plt.subplots(1,len(points),figsize=(4.6*len(points),4.4),squeeze=False)
        for column,m in enumerate(points):
            ax, phase = axes[0,column], phase_axes[0,column]
            for backend in BACKENDS:
                cells = sorted([row for row in rows if row["model"]==name and int(row["M"])==m and row["backend"]==backend and row["state"]=="complete"],key=lambda row:int(row["N"]))
                if cells:
                    ax.loglog([int(row["N"]) for row in cells],[float(row["fits_per_second"]) for row in cells],"o-",label=backend)
                if backend == "scipy":
                    limited=[row for row in rows if row["model"]==name and int(row["M"])==m and row["backend"]==backend and row["state"]=="time_limit"]
                    if limited:
                        ax.scatter([int(row["completed"]) for row in limited],[float(row["fits_per_second"]) for row in limited],marker="x",label="scipy time-limited prefix")
            ax.set(title=f"{name}, M={m}",xlabel="fits actually completed (N)",ylabel="fits / second (wall)")
            ax.grid(True,which="both",alpha=.25)
            cells=sorted([row for row in rows if row["model"]==name and int(row["M"])==m and row["backend"]=="metal_auto" and row["state"]=="complete"],key=lambda row:int(row["N"]))
            if cells:
                x=[int(row["N"]) for row in cells]
                fractions=[[float(row[key])/float(row["wall_seconds"]) for row in cells] for key in ["allocation_seconds","prepare_seconds","gpu_seconds"]]
                fractions.append([max(0,1-sum(values)) for values in zip(*fractions)])
                phase.stackplot(x,*fractions,labels=["allocation","packing / transpose","GPU kernel","host / readback / other"])
                phase.set_xscale("log")
            phase.set(title=f"automatic Metal, M={m}",xlabel="N",ylabel="fraction of wall time",ylim=(0,1))
            phase.grid(True,alpha=.25)
        axes[0,-1].legend(fontsize=7)
        phase_axes[0,-1].legend(fontsize=7)
        fig.suptitle("Measured throughput; time-limited SciPy prefixes are not extrapolated")
        phases.suptitle("Warm automatic GPU phase split (validation excluded)")
        for figure,suffix in [(fig,"throughput"),(phases,"phases")]:
            figure.tight_layout()
            for extension in ("png","svg"):
                path=directory/f"{name}_{suffix}.{extension}"
                figure.savefig(path,dpi=150)
                if extension=="svg":
                    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
            plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build",action="store_true")
    parser.add_argument("--build-dir",type=pathlib.Path,default=ROOT/"build/benchmark-suite")
    parser.add_argument("--output",type=pathlib.Path,default=ROOT/"bench/results")
    parser.add_argument("--models",nargs="+",choices=MODELS,default=["lorentzian"])
    parser.add_argument("--ns",nargs="+",type=int,default=[1,10,100,300,1000,3000,10000,100000,1000000,10000000])
    parser.add_argument("--points",nargs="+",type=int,default=[32,128,512,2048])
    parser.add_argument("--workers",type=int,default=min(2,os.cpu_count() or 1))
    parser.add_argument("--chunk-fits",type=int,default=16384)
    parser.add_argument("--patterns",type=int,default=1024)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--large-repeats",type=int,default=1)
    parser.add_argument("--scipy-budget-seconds",type=float,default=30.)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--plot-only",action="store_true",help="regenerate figures from saved summary.csv without native fitting")
    args=parser.parse_args()
    if args.smoke:
        args.models=list(MODELS);args.ns=[17,65];args.points=[32];args.workers=2;args.patterns=32;args.chunk_fits=31;args.repeats=1;args.scipy_budget_seconds=10
    if min(args.ns)<=0 or min(args.points)<2 or min(args.workers,args.patterns,args.chunk_fits,args.repeats,args.large_repeats)<=0 or args.scipy_budget_seconds<0:
        parser.error("positive dimensions/repeats/workers and nonnegative SciPy budget required")
    for path in (args.build_dir,args.output):
        if not path.resolve().is_relative_to(ROOT):
            parser.error("build/output paths must stay inside this repository")
    if args.plot_only:
        with (args.output/"summary.csv").open() as stream:
            plot(list(csv.DictReader(stream)),args.output)
        return
    if args.build:
        build(args.build_dir)
    native=load_native(args.build_dir)
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"raw.csv").exists():
        parser.error("refusing to overwrite an existing benchmark result set")
    metadata={key:str(value.resolve().relative_to(ROOT)) if isinstance(value,pathlib.Path) else value for key,value in vars(args).items()}
    metadata.update(platform=platform.platform(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    metadata["git_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if sys.platform=="darwin":
        metadata["chip"]=subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"],text=True).strip()
        metadata["mac_model"]=subprocess.check_output(["sysctl","-n","hw.model"],text=True).strip()
    (args.output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    rows=[]
    for model in args.models:
        for m in args.points:
            data=native.inputs(model,m,args.patterns)
            for n in args.ns:
                repeats=args.large_repeats if n>=1000000 else args.repeats
                limited=set()
                for repeat in range(repeats):
                    order=BACKENDS[repeat%len(BACKENDS):]+BACKENDS[:repeat%len(BACKENDS)]
                    for backend in order:
                        if backend in limited:
                            continue
                        measured=scipy_run(model,data,n,min(args.chunk_fits,1024),args.scipy_budget_seconds) if backend=="scipy" else native.run(model,data,n,backend,args.workers,args.chunk_fits)
                        row=dict(model=model,N=n,M=m,backend=backend,repeat=repeat,workers=args.workers if backend=="cpu_multi" else 1,patterns=args.patterns,compute_dtype="float64" if backend=="scipy" else "float32",dataset_setup_seconds=data["dataset_setup_seconds"],**measured)
                        errors=sum(int(row[key]) for key in ["status_errors","parameter_errors","cost_errors"])
                        row["state"]="numerical_error" if errors else "complete" if row["completed"]==n else "time_limit"
                        row["fits_per_second"]=int(row["completed"])/float(row["wall_seconds"])
                        row["gpu_fits_per_second"]=int(row["completed"])/float(row["gpu_seconds"]) if float(row["gpu_seconds"])>0 else 0
                        rows.append(row)
                        write_csv(args.output/"raw.csv",rows)
                        print(f"{model} N={n} M={m} {backend} {row['state']} completed={row['completed']} wall={row['wall_seconds']:.6g}s gpu={row['gpu_seconds']:.6g}s",flush=True)
                        if row["state"]=="time_limit":limited.add(backend)
                        if errors:raise RuntimeError(f"numerical failure retained in raw.csv: {row}")
    summary=summarize(rows)
    write_csv(args.output/"summary.csv",summary)
    plot(summary,args.output)
    metadata["finished_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    metadata["peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)
    (args.output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(f"Measured {len(rows)} runs; figures and raw/summary CSVs in {args.output}")


if __name__=="__main__":
    main()
