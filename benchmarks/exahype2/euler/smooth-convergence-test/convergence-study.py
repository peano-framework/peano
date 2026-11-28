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

This is part of the ExaHyPE 2 integration test (skill exahype2:integration-test).

"""
import argparse
import os
import subprocess
import sys

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
    "--compiler",
    default=None,
    help="Task graph compiler, see smooth-convergence-test.py (default: the script's default)",
)
args = parser.parse_args()

work = os.path.abspath(args.work_directory)
os.makedirs(work, exist_ok=True)


def study(scheme):
    minimum_order = args.minimum_order if args.minimum_order is not None else MINIMUM_ORDERS[scheme]
    print(f"scheme {scheme}", flush=True)
    runs = []
    for volume_size in VOLUME_SIZES:
        run = os.path.join(work, scheme, f"run-{volume_size}")
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
                    str(END_TIME),
                    "--build-jobs",
                    str(args.build_jobs),
                    "--verbosity",
                    "text",
                ]
                + (["--compiler", args.compiler] if args.compiler else []),
                check=True,
                stdout=subprocess.DEVNULL,
            )
            with open(os.path.join(run, "run.log"), "w") as log:
                # Wall-clock time [s] and peak memory [kB] of the run, see time.txt
                subprocess.run(
                    ["/usr/bin/time", "-f", "%e %M", "-o", os.path.join(run, "time.txt"), executable],
                    cwd=run,
                    check=True,
                    stdout=log,
                    stderr=log,
                )
    return subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "compute-errors.py"),
            "--minimum-order",
            str(minimum_order),
            "--end-time",
            str(END_TIME),
        ]
        + runs
    ).returncode


failed = [scheme for scheme in args.schemes if study(scheme) != 0]
if failed:
    print(f"FAILED: {', '.join(failed)}")
sys.exit(1 if failed else 0)
