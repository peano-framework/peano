# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

Convergence study of the entropy wave benchmark

Generates, builds and runs smooth-convergence-test-fd4.py for the three
standard mesh widths (@ref benchmarks_exahype2_euler_smooth_fd4), and
hands the outcome to compute-errors.py. The script exits with a non-zero
status if the order of the maximum error between the two finest meshes
falls below the minimum order (default 3.5; the scheme yields 4.0).

This is part of the ExaHyPE 2 integration test (skill exahype2:integration-test).

"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VOLUME_SIZES = ["0.0075", "0.0025", "0.00083"]

parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
parser.add_argument(
    "--work-directory",
    required=True,
    help="Directory the runs are built in (outside of the repository)",
)
parser.add_argument(
    "--minimum-order",
    type=float,
    default=3.5,
    help="Minimum order between the two finest meshes (default: 3.5)",
)
parser.add_argument(
    "--rk-order",
    type=int,
    default=4,
    help="Order of the Runge-Kutta scheme (default: 4)",
)
args = parser.parse_args()

work = os.path.abspath(args.work_directory)
os.makedirs(work, exist_ok=True)
runs = []
for volume_size in VOLUME_SIZES:
    run = os.path.join(work, f"run-{volume_size}")
    runs.append(run)
    executable = os.path.join(run, "smooth-convergence-test-fd4")
    if not os.path.exists(
        os.path.join(run, "solution-EulerFD4Smooth.peano-patch-file")
    ):
        print(f"build and run {volume_size}", flush=True)
        subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "smooth-convergence-test-fd4.py"),
                "--output-directory",
                run,
                "--volume-size",
                volume_size,
                "--rk-order",
                str(args.rk_order),
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
            str(args.minimum_order),
        ]
        + runs
    ).returncode
)
