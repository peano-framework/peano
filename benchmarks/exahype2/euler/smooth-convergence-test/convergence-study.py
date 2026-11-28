# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

Convergence study of the entropy wave benchmark

Generates, builds and runs smooth-convergence-test.py for the three
standard mesh widths (@ref benchmarks_exahype2_euler_smooth_setup), and
hands the outcome to compute-errors.py. The script exits with a non-zero
status if the order of the maximum error between the two finest meshes
falls below the minimum order of the scheme (e.g. 3.5 for rk4fd4, which yields 4.0),
or if the last snapshot of any run does not reach the end time.
By default, the study runs the Finite Difference schemes rk1fd4, rk2fd4, rk3fd4 and
rk4fd4 one after another (each in a subdirectory of the work directory) and fails
if any of them misses its minimum order.

All of this is repeated for each task graph compiler of the registry
@ref peano4.api.taskgraph.compiler.Compilers. The runs of a
compiler live in <work directory>/<load balancing>/<compiler>/<scheme>.

A set, i.e. one task graph compiler with one scheme, stops at its first failure:
The study skips its remaining mesh widths, and it does not run it with the load
balancings that follow. This includes sets which miss their order. All other sets
continue. The study ends with a table of the verdicts of all sets.

The study first runs all of this with a single tree (none), and then again with
the load balancing toolbox::loadbalancing::strategies::SpreadOut (spread-out),
so that errors which the domain decomposition introduces show up as such. A run
whose log does not report the expected trees fails: more than one tree without
load balancing, or a single tree only with it.

This is part of the ExaHyPE 2 integration test (skill exahype2:integration-test).

"""
import argparse
import os
import re
import subprocess
import sys

import peano4.api.taskgraph.compiler.Compilers

HERE = os.path.dirname(os.path.abspath(__file__))
## Orders the scheme has to reach at least, i.e. slightly below the formal ones
MINIMUM_ORDERS = {
    "fv": 0.8,
    "fv-optimistic": 0.8,
    "rk1fd4": 0.8,
    "rk2fd4": 1.8,
    "rk3fd4": 2.8,
    "rk4fd4": 3.5,
}
VOLUME_SIZES = ["0.0075", "0.0025", "0.00083"]
END_TIME = 0.05
## Load balancing configurations, in the order in which they are studied: first
## a single tree, then several trees, so errors which stem from the load balancing
## show up as such
LOAD_BALANCINGS = ["none", "spread-out"]
## Task graph compilers the study checks, i.e. all of them, see
## smooth-convergence-test.py
COMPILERS = list(peano4.api.taskgraph.compiler.Compilers.COMPILERS)

parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
parser.add_argument(
    "--work-directory",
    required=True,
    help="Directory the runs are built in (outside of the repository)",
)
parser.add_argument(
    "--schemes",
    nargs="+",
    default=["rk1fd4", "rk2fd4", "rk3fd4", "rk4fd4"],
    choices=list(MINIMUM_ORDERS.keys()),
    help="Numerical schemes, see smooth-convergence-test.py (default: rk1fd4 rk2fd4 rk3fd4 rk4fd4)",
)
parser.add_argument(
    "--end-time",
    type=float,
    default=END_TIME,
    help=f"Simulated time of all runs (default: {END_TIME}). The minimum orders stem from the default. Use a work directory of its own per end time, as finished runs are reused",
)
parser.add_argument(
    "--minimum-order",
    type=float,
    default=None,
    help="Minimum order between the two finest meshes, for all schemes (default: slightly below the scheme's formal order, e.g. 3.5 for rk4fd4)",
)
parser.add_argument(
    "-j",
    "--build-jobs",
    type=int,
    default=0,
    help="Parallel make jobs per build; 0 or less uses all cores (default: all cores)",
)
parser.add_argument(
    "--load-balancing",
    nargs="+",
    default=LOAD_BALANCINGS,
    help=f"Load balancings in the order they are studied, see smooth-convergence-test.py (default: {' '.join(LOAD_BALANCINGS)})",
)
parser.add_argument(
    "--compilers",
    nargs="+",
    default=COMPILERS,
    help=f"Task graph compilers, see smooth-convergence-test.py (default: {' '.join(COMPILERS)})",
)
args = parser.parse_args()

work = os.path.abspath(args.work_directory)
os.makedirs(work, exist_ok=True)


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


def compute_errors(runs, extra_arguments=[]):
    """!
    Run compute-errors.py on run directories

    @param runs: [str]
      Run directories, from the coarsest to the finest mesh.
    @param extra_arguments: [str]
      Further arguments of compute-errors.py besides the end time.
    @return subprocess.CompletedProcess
      Exit code and output of the script.
    """
    return subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "compute-errors.py"),
            "--end-time",
            str(args.end_time),
        ]
        + extra_arguments
        + runs,
        capture_output=True,
        text=True,
    )


def failure_of(process):
    """! Reason why compute-errors.py has failed, from its output"""
    lines = (process.stdout + process.stderr).strip().splitlines()
    return next(
        (line for line in lines if line.startswith("FAILED: ")), lines[-1]
    ).replace("FAILED: ", "")


def study(load_balancing, compiler, scheme):
    """!
    Run one set with one load balancing, and stop at its first failure

    @return str
      The failure, or None.
    """
    minimum_order = (
        args.minimum_order if args.minimum_order is not None else MINIMUM_ORDERS[scheme]
    )
    print(
        f"scheme {scheme} with {compiler}, load balancing {load_balancing}", flush=True
    )
    runs = []
    for volume_size in VOLUME_SIZES:
        run = os.path.join(work, load_balancing, compiler, scheme, f"run-{volume_size}")
        runs.append(run)
        executable = os.path.join(run, "smooth-convergence-test")
        if not os.path.exists(
            os.path.join(run, "solution-EulerSmooth.peano-patch-file")
        ):
            print(f"build and run {scheme} {volume_size}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    os.path.join(HERE, "smooth-convergence-test.py"),
                    "--output-directory",
                    run,
                    "--volume-size",
                    volume_size,
                    "--scheme",
                    scheme,
                    "--end-time",
                    str(args.end_time),
                    "--build-jobs",
                    str(args.build_jobs),
                    "--verbosity",
                    "text",
                    "--compiler",
                    compiler,
                    "--load-balancing",
                    load_balancing,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            with open(os.path.join(run, "run.log"), "w") as log:
                # Wall-clock time [s] and peak memory [kB] of the run, see time.txt
                subprocess.run(
                    [
                        "/usr/bin/time",
                        "-f",
                        "%e %M",
                        "-o",
                        os.path.join(run, "time.txt"),
                        executable,
                    ],
                    cwd=run,
                    check=True,
                    stdout=log,
                    stderr=log,
                )
        trees = number_of_trees(os.path.join(run, "run.log"))
        print(f"{run}: {trees} tree(s)", flush=True)
        failure = tree_failure(load_balancing, trees)
        if failure is None:
            process = compute_errors([run])
            if process.returncode != 0:
                failure = failure_of(process)
        if failure:
            return f"{volume_size}: {failure}"
    process = compute_errors(runs, ["--minimum-order", str(minimum_order)])
    print(process.stdout + process.stderr, end="", flush=True)
    return failure_of(process) if process.returncode != 0 else None


def verdict_of(load_balancing, compiler, scheme):
    """! A failed build or run must not stop the study of the other sets"""
    try:
        failure = study(load_balancing, compiler, scheme)
    except (subprocess.CalledProcessError, OSError) as exception:
        failure = str(exception)
    if failure:
        print(f"{load_balancing} {compiler} {scheme}: FAILED: {failure}", flush=True)
        return f"FAILED: {failure}"
    return "PASSED"


## Sets (task graph compiler, scheme) which have failed, mapped onto the load
## balancing under which they have failed. The study does not continue them.
failed_sets = {}
verdicts = {}
for load_balancing in args.load_balancing:
    for compiler in args.compilers:
        for scheme in args.schemes:
            key = (load_balancing, compiler, scheme)
            if (compiler, scheme) in failed_sets:
                verdicts[
                    key
                ] = f"not run (failed with {failed_sets[(compiler, scheme)]})"
                continue
            verdicts[key] = verdict_of(*key)
            if verdicts[key] != "PASSED":
                failed_sets[(compiler, scheme)] = load_balancing

print("| compiler | scheme | " + " | ".join(args.load_balancing) + " |")
print("|---" * (len(args.load_balancing) + 2) + "|")
for compiler in args.compilers:
    for scheme in args.schemes:
        columns = [verdicts[(lb, compiler, scheme)] for lb in args.load_balancing]
        print(f"| {compiler} | {scheme} | " + " | ".join(columns) + " |")
print("FAILED" if failed_sets else "PASSED")
sys.exit(1 if failed_sets else 0)
