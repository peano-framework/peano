# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

Convergence study of the gauge wave with a scalar field

The script builds and runs gauge-wave-with-scalar-field.py for several couplings
and mesh sizes. It measures the maximum error of the scalar field along the line y=z=0.5
at the end of the simulation against the exact solution, estimates the convergence
orders, and writes convergence.png (error over mesh size) and convergence.md (table)
into the output directory. The study doubles as the integration test of the couplings:
The script exits with a non-zero status if the estimated order between the two finest meshes
falls below the one in ExpectedOrders.

     python3 convergence-study.py --work-directory /scratch/gauge-wave-with-scalar-field

Runs which have completed before in the work directory are reused. Remove the run's
directory (or results.json) to rerun it.

## Exact solution

The scalar field obeys @f$ \\partial_t q + H \\partial_x q = 0 @f$ with @f$ H = 1 - A \\sin(2 \\pi (x-t)) @f$,
the g_xx of the gauge wave. With @f$ \\theta = \\pi (x - t) @f$, the characteristics fulfil
@f$ \\tan \\theta(t) = \\tan \\theta_0 \\ e^{-2 \\pi A t} @f$. The initial data are q_m = sin(2 pi x + m pi/4),
m=0,...,3, i.e. q_m(x,t) = sin(2 theta_0 + m pi/4), where theta_0 is the foot point of the characteristic.
See DESIGN.md.

"""
import argparse
import contextlib
import glob
import io
import json
import math
import os
import shutil
import subprocess
import sys

Here = os.path.dirname(os.path.abspath(__file__))
PeanoRoot = os.path.abspath(os.path.join(Here, "..", "..", ".."))
sys.path.insert(0, os.path.join(PeanoRoot, "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tarch.plotter.api.output

EndTime = 0.1
Amplitude = 0.1
PatchFile = "solution-ScalarField.peano-patch-file"

## Number of volumes per axis. The meshes are regular and refined by factors of
## three. Peano refines the root cell at least once, i.e. the coarsest mesh needs patches of size 3.
Meshes = [9, 27, 81]


def patch_size_of(volumes_per_axis):
    return 3 if volumes_per_axis == 9 else 9


## The scheme of both solvers, and the couplings that are studied for it
Studies = [
    ("rk4fd4", "outer-multiplicative"),
    ("rk4fd4", "inner-additive"),
    ("rk4fd4", "inner-multiplicative"),
    ("fv", "outer-multiplicative"),
    ("fv", "inner-additive"),
    ("fv", "inner-multiplicative"),
]

## Minimum estimated order between the two finest meshes. The outer coupling freezes the spacetime
## within the time step (first order). The inner couplings with rk4fd4 keep the order of the time
## integrator, which the fourth-order discretisation in space can not
## exceed. Finite Volumes are first order. The values include a safety margin.
ExpectedOrders = {
    ("rk4fd4", "outer-multiplicative"): 0.8,
    ("rk4fd4", "inner-additive"): 3.2,
    ("rk4fd4", "inner-multiplicative"): 3.2,
    ("fv", "outer-multiplicative"): 0.7,
    ("fv", "inner-additive"): 0.7,
    ("fv", "inner-multiplicative"): 0.7,
}
## Orders which the couplings must not exceed (they would indicate that the exact solution is wrong)
MaximumOrders = {("rk4fd4", "outer-multiplicative"): 1.6}


def exact_solution(x, t, component):
    """! Component 0, ..., 3 of the scalar field: sin(2 pi x + m pi/4) transported along the characteristics """
    theta = np.pi * (np.asarray(x) - t)
    foot = np.arctan2(np.exp(2.0 * np.pi * Amplitude * t) * np.sin(theta), np.cos(theta))
    return np.sin(2.0 * foot + 0.25 * np.pi * component)


def name_of(study, volumes_per_axis):
    scheme, coupling = study
    return f"{scheme}-{coupling}-{volumes_per_axis}"


def build_and_run(study, volumes_per_axis, threads, work_directory):
    """!

    Generate, compile and run gauge-wave-with-scalar-field.py

    @return dict
      Runtime, memory footprint, threads and the error against the exact
      solution at the last snapshot.

    """
    scheme, coupling = study
    name = name_of(study, volumes_per_axis)
    directory = os.path.join(work_directory, name)
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory)
    print(f"build {name} ({threads} threads)", flush=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = (
        os.path.join(PeanoRoot, "src") + os.pathsep + environment.get("PYTHONPATH", "")
    )
    subprocess.run(
        [
            sys.executable,
            os.path.join(Here, "gauge-wave-with-scalar-field.py"),
            "--mode", "release",
            "--scheme", scheme,
            "--coupling", coupling,
            "--patch-size", str(patch_size_of(volumes_per_axis)),
            "--mesh-size", repr(1.0 / volumes_per_axis),
            "--end-time", repr(EndTime),
            "--directory", directory,
            "--threads", str(threads),
            "--plot-along-x-axis",
        ],
        cwd=directory,
        env=environment,
        check=True,
        stdout=open(os.path.join(directory, "generate.log"), "w"),
        stderr=subprocess.STDOUT,
    )
    print(f"run {name}", flush=True)
    time_file = os.path.join(directory, "time.txt")
    with open(os.path.join(directory, "run.log"), "w") as log:
        process = subprocess.run(
            ["/usr/bin/time", "-f", "%e %M", "-o", time_file, "./gauge-wave-with-scalar-field"],
            cwd=directory,
            stdout=log,
        )
    if process.returncode != 0:
        raise RuntimeError(f"{name} failed, see {directory}/run.log")
    with open(time_file) as f:
        runtime, memory = f.read().split()[-2:]
    t, error = measure_error(directory, volumes_per_axis)
    return {
        "scheme": scheme,
        "coupling": coupling,
        "volumes": volumes_per_axis,
        "threads": threads,
        "runtime": float(runtime),
        "memory": int(memory),
        "time": t,
        "error": error,
    }


def measure_error(directory, volumes_per_axis):
    """!

    Maximum error of the four scalar fields along y=z=0.5 in the last snapshot

    Peano's PointProbe samples the unknowns in the centre of each volume.

    @return time stamp of the snapshot, error

    """
    snapshots = len(glob.glob(os.path.join(directory, "solution-ScalarField-*.peano-patch-file")))
    x = (np.arange(volumes_per_axis) + 0.5) / volumes_per_axis
    values = []
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
        values.append([float(row[header.index(f"phi[{c}]")]) for c in range(4)])
    values = np.array(values)
    error = max(
        np.max(np.abs(values[:, c] - exact_solution(x, t, c))) for c in range(4)
    )
    return t, float(error)


def estimated_order(coarse, fine):
    """! Order from two meshes whose volume sizes differ by a factor of three"""
    return math.log(coarse["error"] / fine["error"]) / math.log(3.0)


def plot(results, studies, output_file):
    figure, axes = plt.subplots(figsize=(7, 5))
    volumes = set()
    errors = []
    for study in studies:
        points = sorted(
            (r for r in results.values() if (r["scheme"], r["coupling"]) == study),
            key=lambda r: r["volumes"],
        )
        if points:
            axes.loglog(
                [1.0 / r["volumes"] for r in points],
                [r["error"] for r in points],
                marker="o",
                label=" ".join(study),
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
    axes.set_ylabel(f"max error of the scalar field at t={EndTime}")
    axes.set_title("Gauge wave with a scalar field: convergence")
    axes.grid(True, which="major", alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_file, dpi=150)
    plt.close(figure)


def table(results, studies, meshes):
    lines = [
        "| scheme | coupling | volumes per axis | max error of the scalar field | order | runtime [s] | threads | memory [MB] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for study in studies:
        previous = None
        for volumes in meshes:
            r = results.get(name_of(study, volumes))
            if r is None:
                continue
            order = "" if previous is None else f"{estimated_order(previous, r):.2f}"
            lines.append(
                f"| {study[0]} | {study[1]} | {volumes} | {r['error']:.3e} | {order} | {r['runtime']:.0f} | {r['threads']} | {r['memory'] / 1024:.0f} |"
            )
            previous = r
    return "\n".join(lines) + "\n"


def check_orders(results, studies, meshes):
    """! The studies whose order between the two finest meshes is outside the expected range """
    failures = []
    for study in studies:
        done = [
            results[name_of(study, v)] for v in meshes if name_of(study, v) in results
        ]
        if len(done) < 2:
            failures.append(f"{' '.join(study)}: fewer than two meshes")
            continue
        order = estimated_order(done[-2], done[-1])
        if order < ExpectedOrders[study]:
            failures.append(f"{' '.join(study)}: order {order:.2f} < {ExpectedOrders[study]}")
        if order > MaximumOrders.get(study, math.inf):
            failures.append(f"{' '.join(study)}: order {order:.2f} > {MaximumOrders[study]}")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convergence study of the gauge wave with a scalar field"
    )
    parser.add_argument("--work-directory", required=True, help="Directory for builds, outputs and results.json")
    parser.add_argument(
        "--output-directory",
        default=os.path.join(Here, "example-output"),
        help="Directory for convergence.png and convergence.md",
    )
    parser.add_argument(
        "--studies",
        nargs="+",
        default=[f"{s}:{c}" for s, c in Studies],
        help="scheme:coupling pairs, e.g. rk4fd4:inner-additive",
    )
    parser.add_argument("--meshes", nargs="+", type=int, default=Meshes, help="Volumes per axis (9, 27, 81, 243, ...)")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    args = parser.parse_args()

    studies = [tuple(s.split(":")) for s in args.studies]
    meshes = sorted(args.meshes)
    work_directory = os.path.abspath(args.work_directory)
    os.makedirs(work_directory, exist_ok=True)
    results_file = os.path.join(work_directory, "results.json")
    results = {}
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)

    for study in studies:
        for v in meshes:
            if name_of(study, v) not in results:
                results[name_of(study, v)] = build_and_run(
                    study, v, 1 if patch_size_of(v) == 3 or v == 27 else args.threads, work_directory
                )
                with open(results_file, "w") as f:
                    json.dump(results, f, indent=1)

    os.makedirs(args.output_directory, exist_ok=True)
    plot(results, studies, os.path.join(args.output_directory, "convergence.png"))
    markdown = table(results, studies, meshes)
    with open(os.path.join(args.output_directory, "convergence.md"), "w") as f:
        f.write(markdown)
    print(markdown)

    failures = check_orders(results, studies, meshes)
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("PASSED")
