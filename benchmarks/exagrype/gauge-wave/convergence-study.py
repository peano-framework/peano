# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

Convergence study of the gauge wave benchmark

The script builds and runs gauge-wave.py for several task graph compilers,
numerical schemes, both CCZ4 formulations (first-order, fo, and first-order
second-order, so) and mesh sizes, measures the maximum error of the lapse along the line y=z=0.5
at the end of the simulation against the exact solution, estimates the
convergence orders, and writes

- convergence.png (error over mesh size),
- convergence.md (table with errors, orders, runtimes and memory footprint),
- results.json (everything in machine-readable form, in the work directory)

into a subdirectory <load balancing>/<task graph compiler> of the output directory
(results.json goes into the same subdirectory of the work directory). The study is the integration test for the
correctness of the numerical schemes behind the benchmark: The script exits
with a non-zero status if the estimated convergence order of a scheme falls
below the one in ExpectedOrders, or if any build or run fails, for any of the
compilers and load balancings. Runtimes and memory footprints are recorded,
but are not an acceptance criterion.

A set, i.e. one task graph compiler with one numerical scheme, stops at its first
failure: The study skips its remaining meshes and formulations, and it does not run
it with the load balancings that follow. This includes sets which miss their order.
All other sets continue. The study ends with a table of the verdicts of all sets,
which it also writes into summary.md in the output directory.

     python3 convergence-study.py --work-directory /scratch/gauge-wave

Runs which have completed before in the work directory are reused. Remove
the run's directory (or results.json) to rerun it.


## Load balancing

The study first runs everything with a single tree (none), and then again with
the load balancing toolbox::loadbalancing::strategies::SpreadOut (spread-out),
so that errors which the domain decomposition introduces show up as such. A run
whose log does not report the expected trees fails: more than one tree without
load balancing, or a single tree only with it. The coarse runs, which otherwise
use one thread, get MinimumThreadsWithLoadBalancing threads with load balancing,
as the load balancing does not split on one thread. --load-balancing restricts
or extends the configurations. All results of a configuration live in
<directory>/<load balancing>/<compiler>.


## Task graph compilers

For each load balancing, the study repeats all runs with each task graph compiler of
the registry @ref peano4.api.taskgraph.compiler.Compilers, i.e. a new compiler is
studied without any change to this script. All of them have to reach the same orders.
The compilers are studied one after another. --compilers restricts the list.
The script has no timeout: If a run hangs (the sequential compilers can, see ADR 0001
of the task graph compilers), terminate its executable with kill (SIGTERM), so the
study records a failure and continues.


## Scheduling

Every scheme is studied by a worker of its own, and all workers run at the same
time. A worker walks through the meshes from coarse to fine, and runs all
formulations on one mesh before it continues with the next one. The formulations
share a worker, as twice as many workers would leave each large run with half
the cores. Runs on coarse meshes
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
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time

Here = os.path.dirname(os.path.abspath(__file__))
PeanoRoot = os.path.abspath(os.path.join(Here, "..", "..", ".."))
sys.path.insert(0, os.path.join(PeanoRoot, "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import peano4.api.taskgraph.compiler.Compilers
import tarch.plotter.api.merge
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

## Task graph compilers the study checks, i.e. all of them, see gauge-wave.py
Compilers = list(peano4.api.taskgraph.compiler.Compilers.COMPILERS)

## CCZ4 formulations, see gauge-wave.py
Formulations = ["fo", "so"]

## Load balancing configurations, in the order in which they are studied: first
## a single tree, then several trees, so errors which stem from the load balancing
## show up as such
LoadBalancings = ["none", "spread-out"]

## Minimum threads of a run with load balancing, as there is no split on one thread
MinimumThreadsWithLoadBalancing = 4

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

## redirect_stdout() affects the whole process, i.e. only one worker may use it at
## a time. Otherwise, stdout can stay redirected, and the tables and the verdict
## of the study get lost.
StdoutRedirectLock = threading.Lock()

## Compiling is memory hungry, i.e. only a few builds run at the same time
BuildSlots = threading.Semaphore(3)


## Sets (task graph compiler, scheme) which have failed, mapped onto the load
## balancing under which they have failed. The study does not continue them.
FailedSets = {}


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


def name_of(scheme, formulation, volumes_per_axis):
    return f"{scheme}-{formulation}-{volumes_per_axis}"


def available_memory_in_kilobytes():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1])
    return sys.maxsize


def number_of_trees(log_file):
    """!
    Maximum number of trees which a run reports

    The load balancing statistics write lines such as "8 tree(s): ..." into the
    log. Without load balancing, there are no such lines and there is one tree.

    @param log_file: str
      Log of the run.
    @return int
      Maximum number of trees.
    """
    with open(log_file, errors="ignore") as log:
        return max(
            (int(n) for line in log for n in re.findall(r"(\d+) tree\(s\):", line)),
            default=1,
        )


def tree_failure(load_balancing, trees):
    """!
    Check that a run has used the trees which its load balancing stands for

    A run with load balancing which never splits the domain tests nothing.

    @param load_balancing: str
      Name of the load balancing, "none" stands for a single tree.
    @param trees: int
      Number of trees of the run, see number_of_trees().
    @return str
      The failure, or None.
    """
    if load_balancing == "none" and trees != 1:
        return f"{trees} trees although there is no load balancing"
    if load_balancing != "none" and trees < 2:
        return "the load balancing never split the domain"
    return None


def build_and_run(
    load_balancing,
    compiler,
    scheme,
    formulation,
    volumes_per_axis,
    threads,
    work_directory,
):
    """!

    Generate, compile and run gauge-wave.py

    @return dict
      Runtime, memory footprint, threads and the error against the exact
      solution at the last snapshot. If the build or the run fails, or if the
      run gets killed, the entry "failed" or "skipped" holds the reason instead.

    """
    name = name_of(scheme, formulation, volumes_per_axis)
    run_key = {
        "scheme": scheme,
        "formulation": formulation,
        "volumes": volumes_per_axis,
    }
    directory = os.path.join(work_directory, name)
    # Start from scratch: A previous attempt that was killed leaves generated
    # files behind which clash with the new ones.
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory)
    print(
        f"build {name} with {compiler}, load balancing {load_balancing} ({threads} threads)",
        flush=True,
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        os.path.join(PeanoRoot, "src") + os.pathsep + environment.get("PYTHONPATH", "")
    )
    with BuildSlots:
        generator = subprocess.run(
            [
                sys.executable,
                os.path.join(Here, "gauge-wave.py"),
                "--mode",
                "release",
                "--scheme",
                scheme,
                "--formulation",
                formulation,
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
                "--compiler",
                compiler,
                "--load-balancing",
                load_balancing,
            ],
            cwd=directory,
            env=environment,
            stdout=open(os.path.join(directory, "generate.log"), "w"),
            stderr=subprocess.STDOUT,
        )
    # A failed compilation does not change the exit code of the generator
    if generator.returncode != 0 or not os.path.exists(
        os.path.join(directory, "gauge-wave")
    ):
        print(f"build of {name} failed", flush=True)
        return {**run_key, "failed": f"build failed, see {directory}/generate.log"}
    print(f"run {name}", flush=True)
    time_file = os.path.join(directory, "time.txt")
    with open(os.path.join(directory, "run.log"), "w") as log:
        process = subprocess.Popen(
            ["/usr/bin/time", "-f", "%e %M", "-o", time_file, "./gauge-wave"],
            cwd=directory,
            stdout=log,
        )
        exceeded = watch_memory(process, name)
    if exceeded is not None:
        return {
            **run_key,
            "skipped": f"killed at {exceeded / 1e6:.1f} GB, as the machine ran out of memory",
        }
    if process.returncode in (-9, 137):
        # SIGKILL: the kernel's out-of-memory killer was faster than our watchdog
        return {
            **run_key,
            "skipped": "killed by the operating system, i.e. the machine ran out of memory",
        }
    if process.returncode != 0:
        print(f"run of {name} failed", flush=True)
        return {**run_key, "failed": f"run failed, see {directory}/run.log"}
    trees = number_of_trees(os.path.join(directory, "run.log"))
    failure = tree_failure(load_balancing, trees)
    if failure:
        print(f"run of {name} failed: {failure}", flush=True)
        return {**run_key, "failed": f"{failure}, see {directory}/run.log"}
    with open(time_file) as f:
        runtime, memory = f.read().split()[-2:]
    # A run which terminates without a usable snapshot must not stop the study
    try:
        t, error = measure_error(directory, volumes_per_axis)
    except Exception as exception:
        print(f"evaluation of {name} failed", flush=True)
        return {
            **run_key,
            "failed": f"evaluation failed ({exception!r}), see {directory}/run.log",
        }
    return {
        **run_key,
        "threads": threads,
        "trees": trees,
        "runtime": float(runtime),
        "memory": int(memory),
        "time": t,
        "error": error,
    }


def measure_error(directory, volumes_per_axis):
    """!

    Maximum error of the lapse along y=z=0.5 in the last snapshot

    A snapshot consists of one patch file per spacetree. We merge the files of
    the last snapshot first (@ref tarch.plotter.api.merge). Peano's PointProbe
    then samples the lapse in the centre of each volume.

    @return time stamp of the snapshot, error

    """
    meta_file = os.path.join(os.path.abspath(directory), PatchFile)
    last_snapshot = tarch.plotter.api.merge.count_total_snapshot_number(meta_file) - 1
    # The script writes the merged files next to the meta file
    with StdoutRedirectLock, contextlib.redirect_stdout(io.StringIO()):
        tarch.plotter.api.merge.read_meta_file(meta_file, last_snapshot, last_snapshot)
    merged_meta_file = os.path.join(
        os.path.dirname(meta_file), "merged_meta_file.peano-patch-file"
    )
    x = (np.arange(volumes_per_axis) + 0.5) / volumes_per_axis
    lapse = []
    for i, x_i in enumerate(x):
        output_file = os.path.join(directory, f"probe-{i}.csv")
        point_probe = tarch.plotter.api.output.PointProbe(
            file_name=merged_meta_file,
            point=[x_i, 0.5, 0.5],
            output_file=output_file,
            start=0,
            end=0,
        )
        with StdoutRedirectLock, contextlib.redirect_stdout(io.StringIO()):
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


def plot(results, schemes, formulations, meshes, configuration, output_file):
    """!

    Max error over the volume size

    The gray lines are guides for the orders 1 to 4. The Runge-Kutta schemes
    of order three and four yield almost the same error, as the error of the
    fourth-order spatial discretisation dominates. We therefore use different
    markers.

    """
    figure, axes = plt.subplots(figsize=(7, 5))
    # Solid lines for the first formulation, dashed ones for the second
    line_styles = {"fo": "-", "so": "--"}
    markers = {"fv": "o", "rk1fd4": "s", "rk2fd4": "^", "rk3fd4": "D", "rk4fd4": "x"}
    sizes = {"rk3fd4": 9}
    volumes = set()
    errors = []
    for scheme in schemes:
        colour = f"C{Schemes.index(scheme)}"
        for formulation in formulations:
            points = sorted(
                (
                    r
                    for r in results.values()
                    if r["scheme"] == scheme
                    and r.get("formulation") == formulation
                    and "error" in r
                ),
                key=lambda r: r["volumes"],
            )
            if points:
                axes.loglog(
                    [1.0 / r["volumes"] for r in points],
                    [r["error"] for r in points],
                    marker=markers[scheme],
                    markersize=sizes.get(scheme, 6),
                    color=colour,
                    linestyle=line_styles[formulation],
                    label=f"{scheme} {formulation}",
                )
                volumes.update(r["volumes"] for r in points)
                errors.extend(r["error"] for r in points)
    if not volumes:
        # All runs have failed, i.e. there is nothing to plot
        plt.close(figure)
        return
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
    axes.set_title(f"Gauge wave convergence ({configuration})")
    axes.grid(True, which="major", alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_file, dpi=150)
    plt.close(figure)


def table(results, schemes, formulations, meshes):
    lines = [
        "| scheme | formulation | volumes per axis | max error of lapse | order | runtime [s] | threads | trees | memory [MB] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for scheme, formulation in itertools.product(schemes, formulations):
        previous = None
        for volumes in meshes:
            r = results.get(name_of(scheme, formulation, volumes))
            if r is None:
                continue
            prefix = f"| {scheme} | {formulation} | {volumes} |"
            if "error" not in r:
                reason = (
                    f"failed: {r['failed']}"
                    if "failed" in r
                    else f"skipped: {r['skipped']}"
                )
                lines.append(f"{prefix} {reason} | | | | | |")
                previous = None
                continue
            order = "" if previous is None else f"{estimated_order(previous, r):.2f}"
            lines.append(
                f"{prefix} {r['error']:.3e} | {order} | {r['runtime']:.0f} | {r['threads']} | {r['trees']} | {r['memory'] / 1024:.0f} |"
            )
            previous = r
    return "\n".join(lines) + "\n"


def check_orders(results, schemes, formulations, meshes):
    """!

    Return the list of failures: failed builds or runs, and combinations of
    scheme and formulation whose order between the two finest meshes is too low.
    Both formulations have to reach the order of the scheme.

    """
    failures = []
    for scheme, formulation in itertools.product(schemes, formulations):
        label = f"{scheme} {formulation}"
        runs = [
            results[name_of(scheme, formulation, v)]
            for v in meshes
            if name_of(scheme, formulation, v) in results
        ]
        broken = [r for r in runs if "failed" in r]
        if broken:
            failures.extend(
                f"{label}, {r['volumes']} volumes: {r['failed']}" for r in broken
            )
            continue
        done = [r for r in runs if "error" in r]
        if len(done) < 2:
            failures.append(f"{label}: fewer than two meshes")
            continue
        order = estimated_order(done[-2], done[-1])
        # Negated, so that an order which is not a number fails
        if not order >= ExpectedOrders[scheme]:
            failures.append(f"{label}: order {order:.2f} < {ExpectedOrders[scheme]}")
    return failures


def study(load_balancing, compiler, args, work_directory, output_directory):
    """!

    Run the convergence study with one load balancing and one task graph compiler

    @param load_balancing: str
      Name of the load balancing, see gauge-wave.py.

    @param compiler: str
      Name of the task graph compiler, see gauge-wave.py.

    @param args: argparse.Namespace
      Command line arguments of the study.

    @param work_directory: str
      Directory of the configuration's runs and its results.json.

    @param output_directory: str
      Directory for the configuration's plot and table.

    @return {str: str}
      Verdict per scheme.

    """
    print(
        f"load balancing {load_balancing}, task graph compiler {compiler}", flush=True
    )
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
    ## Sets which have failed under a load balancing studied before
    not_run = {
        scheme: FailedSets[(compiler, scheme)]
        for scheme in args.schemes
        if (compiler, scheme) in FailedSets
    }

    ## Skipped and failed runs are repeated
    def is_completed(name):
        return name in results and "error" in results[name]

    ## One worker per scheme, all of them at the same time. Each worker runs
    ## its meshes from coarse to fine. The runs on the coarse meshes do not
    ## scale and run with one thread. They are done long before the runs on the
    ## fine meshes get going, i.e. the biggest setups do not start at the same
    ## time. The cores are shared among the workers that still run.
    def run_scheme(scheme):
        try:
            for v, formulation in itertools.product(meshes, args.formulations):
                if (compiler, scheme) in FailedSets:
                    break
                name = name_of(scheme, formulation, v)
                if is_completed(name):
                    continue
                with Workers.lock:
                    share = max(1, Workers.alive)
                threads = (
                    1
                    if mesh_cells(v) <= StagnationCells
                    else max(1, math.ceil(args.cores / share))
                )
                if load_balancing != "none":
                    threads = max(threads, MinimumThreadsWithLoadBalancing)
                large_run = mesh_cells(v) > StagnationCells
                with LargeRunSlots if large_run else contextlib.nullcontext():
                    results[name] = build_and_run(
                        load_balancing,
                        compiler,
                        scheme,
                        formulation,
                        v,
                        threads,
                        work_directory,
                    )
                store()
                if "failed" in results[name]:
                    FailedSets[(compiler, scheme)] = load_balancing
                    break
        finally:
            with Workers.lock:
                Workers.alive -= 1

    Workers.alive = len(args.schemes)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.schemes)) as pool:
        for future in [pool.submit(run_scheme, scheme) for scheme in args.schemes]:
            future.result()

    os.makedirs(output_directory, exist_ok=True)
    plot(
        results,
        args.schemes,
        args.formulations,
        meshes,
        f"{compiler}, load balancing {load_balancing}",
        os.path.join(output_directory, "convergence.png"),
    )
    markdown = table(results, args.schemes, args.formulations, meshes)
    with open(os.path.join(output_directory, "convergence.md"), "w") as f:
        f.write(markdown)
    print(markdown)

    verdicts = {}
    for scheme in args.schemes:
        if scheme in not_run:
            verdicts[scheme] = f"not run (failed with {not_run[scheme]})"
            continue
        runs = [
            results.get(name_of(scheme, formulation, v), {})
            for formulation, v in itertools.product(args.formulations, meshes)
        ]
        # A broken run ends the set, i.e. there are no orders to check
        failures = [
            f"{r['formulation']}, {r['volumes']} volumes: {r['failed']}"
            for r in runs
            if "failed" in r
        ] or check_orders(results, [scheme], args.formulations, meshes)
        if failures:
            FailedSets[(compiler, scheme)] = load_balancing
            verdicts[scheme] = "FAILED: " + "; ".join(failures)
        else:
            verdicts[scheme] = "PASSED"
    return verdicts


def summary(verdicts, compilers, schemes, load_balancings):
    """!

    Table of the verdicts of all sets

    @param verdicts: {(str, str): {str: str}}
      Maps (load balancing, task graph compiler) onto the verdict per scheme,
      see study().

    @return str
      Table in Markdown with one row per set and one column per load balancing.

    """
    lines = [
        "| compiler | scheme | " + " | ".join(load_balancings) + " |",
        "|---" * (len(load_balancings) + 2) + "|",
    ]
    for compiler, scheme in itertools.product(compilers, schemes):
        columns = [verdicts[(lb, compiler)][scheme] for lb in load_balancings]
        lines.append(f"| {compiler} | {scheme} | " + " | ".join(columns) + " |")
    return "\n".join(lines) + "\n"


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
        "--formulations",
        nargs="+",
        default=Formulations,
        choices=Formulations,
        help="CCZ4 formulations (default: both)",
    )
    parser.add_argument(
        "--meshes",
        nargs="+",
        type=int,
        default=Meshes,
        help="Volumes per axis (9, 27, 81, 243, ...)",
    )
    parser.add_argument("--cores", type=int, default=os.cpu_count())
    parser.add_argument(
        "--end-time",
        type=float,
        default=EndTime,
        help=f"Simulated time of all runs (default: {EndTime}). ExpectedOrders stem "
        "from the default. Use a work directory of its own per end time, as "
        "completed runs are reused.",
    )
    parser.add_argument(
        "--load-balancing",
        nargs="+",
        default=LoadBalancings,
        help=f"Load balancings in the order they are studied, see gauge-wave.py (default: {' '.join(LoadBalancings)})",
    )
    parser.add_argument(
        "--compilers",
        nargs="+",
        default=Compilers,
        help=f"Task graph compilers, see gauge-wave.py (default: {' '.join(Compilers)})",
    )
    parser.add_argument(
        "--concurrent-large-runs",
        type=int,
        default=2,
        help="Maximum number of runs on meshes with more than 27 cells that run at the same time",
    )
    args = parser.parse_args()
    EndTime = args.end_time
    LargeRunSlots = threading.Semaphore(args.concurrent_large_runs)

    verdicts = {
        (load_balancing, compiler): study(
            load_balancing,
            compiler,
            args,
            os.path.join(
                os.path.abspath(args.work_directory), load_balancing, compiler
            ),
            os.path.join(args.output_directory, load_balancing, compiler),
        )
        for load_balancing in args.load_balancing
        for compiler in args.compilers
    }
    markdown = summary(verdicts, args.compilers, args.schemes, args.load_balancing)
    with open(os.path.join(args.output_directory, "summary.md"), "w") as f:
        f.write(markdown)
    print(markdown)
    if FailedSets:
        print("FAILED")
        sys.exit(1)
    print("PASSED")
