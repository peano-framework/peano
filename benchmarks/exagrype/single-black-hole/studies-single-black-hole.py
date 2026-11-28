"""

Run the numerical scheme and the performance study of the single black hole

The script drives single-black-hole.py: For each configuration of a study, it
generates the code into its own subdirectory of --work-directory, builds it,
runs the executable and appends one row to results-<study>.csv in the work
directory. The rows hold the configuration, the wall time of the build, the
wall time of the run, the exit code, the seconds in the setup and in the time
stepping sweeps (from Peano's step report), the number of sweeps, the last
time stamp and the log file of the run. The console output of every run goes
into <configuration>/run-<threads>-threads-<repetition>.log.

    python3 studies-single-black-hole.py schemes     --variant profile
    python3 studies-single-black-hole.py performance --variant profile --threads 1 2 4 8

The studies are:

- schemes: Which numerical schemes get through the evolution of the black
  hole? Runs the Finite Volumes on the regular mesh and with one
  additional level of adaptive mesh refinement around the puncture (two levels
  need more than 11 GB and hours even for this radius), and the fourth-order Finite
  Differences with Runge-Kutta orders 2 to 4, with and without lopsided
  advection and Kreiss-Oliger dissipation. A configuration which blows up
  yields a non-zero exit code or the wall time of the run is much shorter
  than the others. The Finite Differences do not support adaptive mesh
  refinement yet.
- performance: How fast is the code? Runs the Finite Volumes on the regular
  mesh and with one level of adaptive mesh refinement for all the task
  graph compilers, and for every thread count passed in --threads. The
  thread count is set through OMP_NUM_THREADS, i.e. it is meaningful for
  OpenMP builds of Peano. Use --repetitions to see the noise.

The studies write binary checkpoints every 10 time units and visualisation
snapshots every 1 time unit, as single-black-hole.py does by default. Analyse
the snapshots with probe-single-black-hole.py afterwards. Pass
--extra-arguments "--checkpoint-interval 0 --plot-interval 0" to switch the
output off, e.g. for timings.

Both studies need a Peano build with GSL, as the initial data comes from
TwoPunctures. Use --dry-run to only list the configurations.

"""

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "src")))

import peano4.api.taskgraph.compiler.Compilers

## Task graph compilers of the performance study, i.e. all of them
Compilers = list(peano4.api.taskgraph.compiler.Compilers.COMPILERS)

# Name and arguments of single-black-hole.py for each configuration
SchemeStudy = {
    "fv-regular": "--scheme fv",
    "fv-amr-1": "--scheme fv --amr-levels 1 --amr-radius 1",
    "fd4-rk2": "--scheme fd4 --rk-order 2",
    "fd4-rk3": "--scheme fd4 --rk-order 3",
    "fd4-rk4": "--scheme fd4 --rk-order 4",
    "fd4-rk4-lopsided": "--scheme fd4 --rk-order 4 --lopsided",
    "fd4-rk4-lopsided-ko": "--scheme fd4 --rk-order 4 --lopsided --ko-sigma 8.0",
    "fd4-rk4-amr-1": "--scheme fd4 --rk-order 4 --amr-levels 1 --amr-radius 1",
}

PerformanceStudy = {
    f"{mesh}-{compiler}": f"--scheme fv {amr} --compiler {compiler}"
    for compiler in Compilers
    for mesh, amr in [("regular", ""), ("amr-1", "--amr-levels 1 --amr-radius 1")]
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Numerical scheme and performance study of the single black hole",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("study", choices=["schemes", "performance"])
    parser.add_argument(
        "--variant",
        choices=["debug", "profile", "production"],
        default="profile",
        help="Build variant of single-black-hole.py.",
    )
    parser.add_argument("--work-directory", default="studies")
    parser.add_argument(
        "--peano-directory",
        default=os.path.abspath(os.path.join(HERE, "..", "..", "..")),
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[0],
        help="Thread counts of the runs. 0 leaves the thread count to the runtime.",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Seconds after which a run is killed.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Names of the configurations to run. Default: all.",
    )
    parser.add_argument(
        "--extra-arguments",
        default="",
        help="Additional arguments for single-black-hole.py, e.g. --end-time 1.",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build(name, configuration, arguments, directory):
    """! Generate and compile one configuration. Returns the wall time or None if it failed."""
    command = (
        ["python3", os.path.join(HERE, "single-black-hole.py")]
        + shlex.split(configuration)
        + [
            "--variant", arguments.variant,
            "--verbosity", "text",
            "--output-directory", directory,
            "--peano-directory", arguments.peano_directory,
        ]
        + shlex.split(arguments.extra_arguments)
    )
    start = time.time()
    with open(os.path.join(directory, "build.log"), "w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    return time.time() - start if result.returncode == 0 else None


def run(arguments, directory, threads, repetition):
    """! Run the executable once. Returns (exit code, wall time)."""
    environment = dict(os.environ)
    if threads > 0:
        environment["OMP_NUM_THREADS"] = str(threads)
    log_name = f"run-{threads}-threads-{repetition}.log"
    start = time.time()
    with open(os.path.join(directory, log_name), "w") as log:
        try:
            result = subprocess.run(
                ["./single-black-hole"],
                cwd=directory,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=arguments.timeout,
            )
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code = "timeout"
    return exit_code, time.time() - start, log_name


def parse_log(path):
    """! Metrics Peano's step report writes at the end of a run

    Returns the seconds of the setup, the seconds spent in all time stepping
    sweeps, the number of sweeps, and the final time stamp. Entries are None
    if the run did not get that far.
    """
    setup = sweeps = sweep_seconds = final_time = None
    with open(path, errors="replace") as log:
        for line in log:
            match = re.search(r"reportFinalResults\(\)\s+info\s+(\w+): ([0-9.e+-]+)s\s+\(avg=([0-9.e+-]+),#measurements=(\d+)", line)
            if match:
                if match.group(1) == "SetupSimulation":
                    setup = float(match.group(2))
                elif match.group(1) == "TimeStep":
                    sweeps = int(match.group(4))
                    sweep_seconds = float(match.group(3)) * sweeps
            match = re.search(r"t=([0-9.e+-]+), new dt=", line)
            if match:
                final_time = float(match.group(1))
    return setup, sweep_seconds, sweeps, final_time


def main():
    arguments = parse_arguments()
    study = SchemeStudy if arguments.study == "schemes" else PerformanceStudy
    if arguments.only is not None:
        unknown = set(arguments.only) - set(study)
        if unknown:
            raise SystemExit(f"Unknown configurations {sorted(unknown)}, available: {sorted(study)}")
        study = {name: study[name] for name in arguments.only}

    if arguments.dry_run:
        for name, configuration in study.items():
            print(f"{name}: single-black-hole.py {configuration} --variant {arguments.variant}")
        return

    os.makedirs(arguments.work_directory, exist_ok=True)
    result_file = os.path.join(arguments.work_directory, f"results-{arguments.study}.csv")
    with open(result_file, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["name", "arguments", "threads", "repetition", "build_seconds", "exit_code", "run_seconds",
             "setup_seconds", "time_stepping_seconds", "sweeps", "final_time", "log"]
        )
        for name, configuration in study.items():
            directory = os.path.abspath(os.path.join(arguments.work_directory, name))
            os.makedirs(directory, exist_ok=True)
            build_seconds = None
            if arguments.skip_build:
                build_seconds = 0.0
            else:
                print(f"Build {name}")
                build_seconds = build(name, configuration, arguments, directory)
            if build_seconds is None:
                print(f"  build failed, see {directory}/build.log")
                writer.writerow([name, configuration, "", "", "", "build failed", "", "", "", "", "", "build.log"])
                handle.flush()
                continue
            for threads in arguments.threads:
                for repetition in range(arguments.repetitions):
                    print(f"Run {name}, {threads} threads, repetition {repetition}")
                    exit_code, run_seconds, log_name = run(arguments, directory, threads, repetition)
                    print(f"  exit code {exit_code}, {run_seconds:.1f}s")
                    metrics = parse_log(os.path.join(directory, log_name))
                    writer.writerow(
                        [name, configuration, threads, repetition, f"{build_seconds:.1f}",
                         exit_code, f"{run_seconds:.1f}", *metrics, log_name]
                    )
                    handle.flush()
    print(f"Wrote {result_file}")


if __name__ == "__main__":
    main()
