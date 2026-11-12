# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

Convergence study of the entropy wave benchmark

Generates, builds and runs smooth-convergence-test.py for the three
standard mesh widths (@ref benchmarks_exahype2_euler_smooth_setup), and
hands the outcome to compute-errors.py. The script exits with a non-zero
status if the order of the maximum error between the two finest meshes
falls below the minimum order (default 3.5 for rk4fd4, which yields 4.0).

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

parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
parser.add_argument(
    "--work-directory",
    required=True,
    help="Directory the runs are built in (outside of the repository)",
)
parser.add_argument(
    "--scheme",
    default="rk4fd4",
    choices=list(MINIMUM_ORDERS.keys()),
    help="Numerical scheme, see smooth-convergence-test.py (default: rk4fd4)",
)
parser.add_argument(
    "--minimum-order",
    type=float,
    default=None,
    help="Minimum order between the two finest meshes (default: slightly below the scheme's formal order, e.g. 3.5 for rk4fd4)",
)
parser.add_argument(
    "-j",
    "--build-jobs",
    type=int,
    default=0,
    help="Parallel make jobs per build; 0 or less uses all cores (default: all cores)",
)
args = parser.parse_args()
minimum_order = args.minimum_order if args.minimum_order is not None else MINIMUM_ORDERS[args.scheme]

work = os.path.abspath(args.work_directory)
os.makedirs(work, exist_ok=True)
runs = []
for volume_size in VOLUME_SIZES:
    run = os.path.join(work, f"run-{volume_size}")
    runs.append(run)
    executable = os.path.join(run, "smooth-convergence-test")
    if not os.path.exists(
        os.path.join(run, "solution-EulerSmooth.peano-patch-file")
    ):
        print(f"build and run {volume_size}", flush=True)
        subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "smooth-convergence-test.py"),
                "--output-directory",
                run,
                "--volume-size",
                volume_size,
                "--scheme",
                args.scheme,
                "--build-jobs",
                str(args.build_jobs),
                "--verbosity",
                "text",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        with open(os.path.join(run, "run.log"), "w") as log:
            subprocess.run([executable], cwd=run, check=True, stdout=log, stderr=log)

sys.exit(
    subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "compute-errors.py"),
            "--minimum-order",
            str(minimum_order),
        ]
        + runs
    ).returncode
)
