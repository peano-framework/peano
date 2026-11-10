# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

Convergence study of the gauge wave benchmark

The script builds and runs gauge-wave.py for several numerical schemes and
mesh sizes, measures the maximum error of the lapse along the line y=z=0.5
at the end of the simulation against the exact solution, estimates the
convergence orders, and writes

- convergence.png (error over mesh size),
- convergence.md (table with errors, orders, runtimes and memory footprint),
- results.json (everything in machine-readable form, in the work directory)

into the output directory. The study is the integration test for the
correctness of the numerical schemes behind the benchmark: The script exits
with a non-zero status if the estimated convergence order of a scheme falls
below the one in ExpectedOrders. Runtimes and memory footprints are recorded,
but are not an acceptance criterion.

     python3 convergence-study.py --work-directory /scratch/gauge-wave

Runs which have completed before in the work directory are reused. Remove
the run's directory (or results.json) to rerun it.


## Scheduling

Every scheme is studied by a worker of its own, and all workers run at the same
time. A worker walks through the meshes from coarse to fine. Runs on coarse meshes
cannot exploit many cores: With 27 mesh cells (the minimum, as Peano refines the
root cell at least once) the scaling stagnates, and these runs use one thread.
They are done long before the runs on the fine meshes get going, i.e. the biggest
setups do not start at the same time. A run on a fine mesh gets its share of the
cores, which grows as workers finish. At most two of them (--concurrent-large-runs)
run at the same time, as they are memory hungry.


## Memory

The memory footprint grows with the number of volumes. A watchdog
observes the footprint of each run. If less than 1.5 GB remain available on the machine,
the run with the biggest footprint is killed (typically right after its start, as the
meshes allocate their memory early) and the study continues without that data point.

"""
import argparse
import concurrent.futures
import contextlib
import glob
import io
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time

Here = os.path.dirname(os.path.abspath(__file__))
PeanoRoot = os.path.abspath(os.path.join(Here, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(PeanoRoot, "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tarch.plotter.api.output

EndTime = 0.1
Amplitude = 0.1
PatchFile = "solution-CCZ4.peano-patch-file"


## The patch size does not influence the solution. Peano refines the root cell
## at least once, i.e. the coarsest mesh with 9 volumes per axis needs patches
## of size 3. All other meshes use patches of size 9.
def patch_size_of(volumes_per_axis):
    return 3 if volumes_per_axis == 9 else 9


Schemes = ["fv", "rk1fd4", "rk2fd4", "rk3fd4", "rk4fd4"]

## Number of volumes per axis. The meshes are regular and refined by factors
## of three, and the patch size is 9.
Meshes = [9, 27, 81]

## Peano refines the root cell at least once, i.e. there are at least 27 mesh
## cells. Runs up to this number of mesh cells do not scale to many cores.
StagnationCells = 27

## Minimum estimated order between the two finest meshes that the schemes
## have to reach. The Runge-Kutta schemes are limited by the order of their
## time integration, the Finite Volume scheme by its first-order spatial
## discretisation. The values include a safety margin.
ExpectedOrders = {
    "fv": 0.8,
    "rk1fd4": 0.8,
    "rk2fd4": 1.8,
    "rk3fd4": 3.2,
    "rk4fd4": 3.2,
}


## Runs on fine meshes are memory hungry, too. Only a few of them run at once;
## main() sets the number.
LargeRunSlots = threading.Semaphore(2)

## Compiling is memory hungry, i.e. only a few builds run at the same time
BuildSlots = threading.Semaphore(3)


class Workers:
    """! Number of scheme workers that are still running, and the lock to protect it"""

    alive = 1
    lock = threading.Lock()


def exact_lapse(x, t):
    """! Lapse of the gauge wave, i.e. sqrt(H) with H = 1 - A sin(2 pi (x-t))"""
    return np.sqrt(1.0 - Amplitude * np.sin(2.0 * np.pi * (np.asarray(x) - t)))


def mesh_cells(volumes_per_axis):
    """! Number of mesh cells (patches) of a run"""
    return (volumes_per_axis // patch_size_of(volumes_per_axis)) ** 3


def name_of(scheme, volumes_per_axis):
    return f"{scheme}-{volumes_per_axis}"


def available_memory_in_kilobytes():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1])
    return sys.maxsize


def build_and_run(scheme, volumes_per_axis, threads, work_directory):
    """!

    Generate, compile and run gauge-wave.py

    @return dict
      Runtime, memory footprint, threads and the error against the exact
      solution at the last snapshot.

    """
    directory = os.path.join(work_directory, name_of(scheme, volumes_per_axis))
    # Start from scratch: A previous attempt that was killed leaves generated
    # files behind which clash with the new ones.
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory)
    print(f"build {name_of(scheme, volumes_per_axis)} ({threads} threads)", flush=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        os.path.join(PeanoRoot, "src") + os.pathsep + environment.get("PYTHONPATH", "")
    )
    with BuildSlots:
        subprocess.run(
            [
                sys.executable,
                os.path.join(Here, "gauge-wave.py"),
                "--mode",
                "release",
                "--scheme",
                scheme,
                "--patch-size",
                str(patch_size_of(volumes_per_axis)),
                "--mesh-size",
                repr(1.0 / volumes_per_axis),
                "--end-time",
                repr(EndTime),
                "--directory",
                directory,
                "--threads",
                str(threads),
                "--plot-along-x-axis",
            ],
            cwd=directory,
            env=environment,
            check=True,
            stdout=open(os.path.join(directory, "generate.log"), "w"),
            stderr=subprocess.STDOUT,
        )
    print(f"run {name_of(scheme, volumes_per_axis)}", flush=True)
    time_file = os.path.join(directory, "time.txt")
    with open(os.path.join(directory, "run.log"), "w") as log:
        process = subprocess.Popen(
            ["/usr/bin/time", "-f", "%e %M", "-o", time_file, "./gauge-wave"],
            cwd=directory,
            stdout=log,
        )
        exceeded = watch_memory(process, name_of(scheme, volumes_per_axis))
    if exceeded is not None:
        return {
            "scheme": scheme,
            "volumes": volumes_per_axis,
            "skipped": f"killed at {exceeded / 1e6:.1f} GB, as the machine ran out of memory",
        }
    if process.returncode in (-9, 137):
        # SIGKILL: the kernel's out-of-memory killer was faster than our watchdog
        return {
            "scheme": scheme,
            "volumes": volumes_per_axis,
            "skipped": "killed by the operating system, i.e. the machine ran out of memory",
        }
    if process.returncode != 0:
        raise RuntimeError(
            f"{name_of(scheme, volumes_per_axis)} failed, see {directory}/run.log"
        )
    with open(time_file) as f:
        runtime, memory = f.read().split()[-2:]
    t, error = measure_error(directory, volumes_per_axis)
    return {
        "scheme": scheme,
        "volumes": volumes_per_axis,
        "threads": threads,
        "runtime": float(runtime),
        "memory": int(memory),
        "time": t,
        "error": error,
    }


def measure_error(directory, volumes_per_axis):
    """!

    Maximum error of the lapse along y=z=0.5 in the last snapshot

    Peano's PointProbe samples the lapse in the centre of each volume.

    @return time stamp of the snapshot, error

    """
    snapshots = len(
        glob.glob(os.path.join(directory, "solution-CCZ4-*.peano-patch-file"))
    )
    x = (np.arange(volumes_per_axis) + 0.5) / volumes_per_axis
    lapse = []
    for i, x_i in enumerate(x):
        output_file = os.path.join(directory, f"probe-{i}.csv")
        point_probe = tarch.plotter.api.output.PointProbe(
            file_name=os.path.join(directory, PatchFile),
            point=[x_i, 0.5, 0.5],
            output_file=output_file,
            start=snapshots - 1,
            end=snapshots - 1,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            point_probe.display()
        with open(output_file) as f:
            header = f.readline().strip().split(",")
            row = f.readline().strip().split(",")
        os.remove(output_file)
        t = float(row[header.index("time")])
        lapse.append(float(row[header.index("lapse")]))
    error = np.max(np.abs(np.array(lapse) - exact_lapse(x, t)))
    return t, float(error)


def estimated_order(coarse, fine):
    """! Order from two meshes whose volume sizes differ by a factor of three"""
    return math.log(coarse["error"] / fine["error"]) / math.log(3.0)


## Minimum memory (kilobytes) that has to stay available on the machine
MemoryFloor = 1500000

## Memory footprints (kilobytes) of the runs that currently run
Footprints = {}


def footprint_of(process):
    """! Resident memory in kilobytes of a process and its children (/usr/bin/time forks the executable)"""
    footprint = 0
    for status in glob.glob("/proc/[0-9]*/status"):
        try:
            with open(status) as f:
                content = f.read()
        except OSError:
            continue
        fields = dict(
            line.split(":", 1) for line in content.splitlines() if ":" in line
        )
        if process.pid in (int(fields.get("PPid", "0")), int(fields.get("Pid", "0"))):
            for line in content.splitlines():
                if line.startswith("VmRSS"):
                    footprint += int(line.split()[1])
    return footprint


def watch_memory(process, name):
    """!

    Wait for a process, but kill it if the machine runs out of memory

    Several runs go on at the same time. If less than MemoryFloor is available
    on the machine, the run with the biggest footprint of all runs is killed.
    The study then continues without that data point. Runs on fine meshes
    allocate their memory early, i.e. we do not waste hours before we learn
    that a run does not fit.

    @return None if the process terminated on its own, otherwise the
      footprint in kilobytes when it was killed.

    """
    try:
        while process.poll() is None:
            footprint = footprint_of(process)
            Footprints[name] = footprint
            if available_memory_in_kilobytes() < MemoryFloor and footprint >= max(
                Footprints.values()
            ):
                subprocess.run(["pkill", "-9", "-P", str(process.pid)])
                process.kill()
                process.wait()
                return footprint
            time.sleep(1.0)
    finally:
        Footprints.pop(name, None)
    return None


def plot(results, schemes, meshes, output_file):
    """!

    Max error over the volume size

    The gray lines are guides for the orders 1 to 4. The Runge-Kutta schemes
    of order three and four yield almost the same error, as the error of the
    fourth-order spatial discretisation dominates. We therefore use different
    markers.

    """
    figure, axes = plt.subplots(figsize=(7, 5))
    markers = {"fv": "o", "rk1fd4": "s", "rk2fd4": "^", "rk3fd4": "D", "rk4fd4": "x"}
    sizes = {"rk3fd4": 9}
    volumes = set()
    errors = []
    for scheme in schemes:
        points = sorted(
            (r for r in results.values() if r["scheme"] == scheme and "error" in r),
            key=lambda r: r["volumes"],
        )
        if points:
            axes.loglog(
                [1.0 / r["volumes"] for r in points],
                [r["error"] for r in points],
                marker=markers[scheme],
                markersize=sizes.get(scheme, 6),
                label=scheme,
            )
            volumes.update(r["volumes"] for r in points)
            errors.extend(r["error"] for r in points)
    volumes = sorted(volumes)
    h = np.array([1.0 / volumes[0], 1.0 / volumes[-1]])
    for order in [1, 2, 3, 4]:
        guide = max(errors) * (h / h[0]) ** order
        axes.loglog(h, guide, color="gray", linestyle=":", linewidth=0.8)
        axes.annotate(f"order {order}", (h[-1], guide[-1]), color="gray", fontsize=8)
    axes.set_xticks([1.0 / v for v in volumes])
    axes.set_xticklabels([f"1/{v}" for v in volumes])
    axes.minorticks_off()
    axes.set_xlabel("volume size h")
    axes.set_ylabel(f"max error of the lapse at t={EndTime}")
    axes.set_title("Gauge wave convergence")
    axes.grid(True, which="major", alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_file, dpi=150)
    plt.close(figure)


def table(results, schemes, meshes):
    lines = [
        "| scheme | volumes per axis | max error of lapse | order | runtime [s] | threads | memory [MB] |",
        "|---|---|---|---|---|---|---|",
    ]
    for scheme in schemes:
        previous = None
        for volumes in meshes:
            r = results.get(name_of(scheme, volumes))
            if r is None:
                continue
            if "error" not in r:
                lines.append(
                    f"| {scheme} | {volumes} | skipped: {r['skipped']} | | | | |"
                )
                previous = None
                continue
            order = "" if previous is None else f"{estimated_order(previous, r):.2f}"
            lines.append(
                f"| {scheme} | {volumes} | {r['error']:.3e} | {order} | {r['runtime']:.0f} | {r['threads']} | {r['memory'] / 1024:.0f} |"
            )
            previous = r
    return "\n".join(lines) + "\n"


def check_orders(results, schemes, meshes):
    """! Return the list of schemes whose order between the two finest meshes is too low"""
    failures = []
    for scheme in schemes:
        done = [
            results[name_of(scheme, v)]
            for v in meshes
            if name_of(scheme, v) in results and "error" in results[name_of(scheme, v)]
        ]
        if len(done) < 2:
            failures.append(f"{scheme}: fewer than two meshes")
            continue
        order = estimated_order(done[-2], done[-1])
        if order < ExpectedOrders[scheme]:
            failures.append(f"{scheme}: order {order:.2f} < {ExpectedOrders[scheme]}")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convergence study of the gauge wave benchmark"
    )
    parser.add_argument(
        "--work-directory",
        required=True,
        help="Directory for builds, outputs and results.json",
    )
    parser.add_argument(
        "--output-directory",
        default=os.path.join(Here, "example-output"),
        help="Directory for convergence.png and convergence.md",
    )
    parser.add_argument("--schemes", nargs="+", default=Schemes, choices=Schemes)
    parser.add_argument(
        "--meshes",
        nargs="+",
        type=int,
        default=Meshes,
        help="Volumes per axis (9, 27, 81, 243, ...)",
    )
    parser.add_argument("--cores", type=int, default=os.cpu_count())
    parser.add_argument(
        "--concurrent-large-runs",
        type=int,
        default=2,
        help="Maximum number of runs on meshes with more than 27 cells that run at the same time",
    )
    args = parser.parse_args()

    work_directory = os.path.abspath(args.work_directory)
    os.makedirs(work_directory, exist_ok=True)
    results_file = os.path.join(work_directory, "results.json")
    results = {}
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)

    def store():
        with open(results_file, "w") as f:
            json.dump(results, f, indent=1)

    meshes = sorted(args.meshes)
    todo = [
        (s, v)
        for s in args.schemes
        for v in meshes
        if name_of(s, v) not in results or "skipped" in results[name_of(s, v)]
    ]

    ## One worker per scheme, all of them at the same time. Each worker runs
    ## its meshes from coarse to fine. The runs on the coarse meshes do not
    ## scale and run with one thread. They are done long before the runs on the
    ## fine meshes get going, i.e. the biggest setups do not start at the same
    ## time. The cores are shared among the workers that still run.
    def run_scheme(scheme):
        try:
            for v in meshes:
                if (
                    name_of(scheme, v) in results
                    and "skipped" not in results[name_of(scheme, v)]
                ):
                    continue
                with Workers.lock:
                    share = max(1, Workers.alive)
                threads = (
                    1
                    if mesh_cells(v) <= StagnationCells
                    else max(1, math.ceil(args.cores / share))
                )
                if mesh_cells(v) > StagnationCells:
                    with LargeRunSlots:
                        results[name_of(scheme, v)] = build_and_run(
                            scheme, v, threads, work_directory
                        )
                else:
                    results[name_of(scheme, v)] = build_and_run(
                        scheme, v, threads, work_directory
                    )
                store()
        finally:
            with Workers.lock:
                Workers.alive -= 1

    Workers.alive = len(args.schemes)
    LargeRunSlots = threading.Semaphore(args.concurrent_large_runs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.schemes)) as pool:
        for future in [pool.submit(run_scheme, scheme) for scheme in args.schemes]:
            future.result()

    os.makedirs(args.output_directory, exist_ok=True)
    plot(
        results,
        args.schemes,
        meshes,
        os.path.join(args.output_directory, "convergence.png"),
    )
    markdown = table(results, args.schemes, meshes)
    with open(os.path.join(args.output_directory, "convergence.md"), "w") as f:
        f.write(markdown)
    print(markdown)

    failures = check_orders(results, args.schemes, meshes)
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("PASSED")
