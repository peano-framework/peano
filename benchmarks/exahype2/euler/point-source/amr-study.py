# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

AMR check of the point source benchmark

Generates, builds and runs the point source benchmark
(@ref benchmarks_exahype2_euler_point_source) without AMR (levels 0) and with AMR
(levels 1 by default), and checks that

- the dynamic AMR works: the mesh has coarsened, i.e. it has fewer cells than the
  regular start grid at the end of the run, and
- the AMR does not harm the quality: the final density and energy differ from the
  solution on the regular grid by at most --tolerance times the range of the
  latter.

Prints a table of the mesh over time, the differences and PASSED or FAILED, and exits
non-zero on FAILED. Finished runs in the work directory are reused.

"""
import argparse
import math
import os
import re
import subprocess
import sys

import numpy as np

from tarch.plotter.api.input.PatchFileParser import PatchFileParser
from tarch.plotter.api.notebook import read_snapshots

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER_NAME = "EulerPointSource"
VOLUMES_PER_PATCH_AXIS = 9


def run(work_directory: str, levels: int, arguments):
    """!
    Generate, build and run the benchmark with the given number of AMR levels

    @param work_directory: str
      Directory of all runs. The run of the AMR levels lives in a directory of its own.
    @param levels: int
      Number of AMR levels.
    @param arguments: [str]
      Further arguments of point-source.py.
    @return str
      The run directory.
    """
    directory = os.path.join(work_directory, f"levels-{levels}")
    if os.path.exists(os.path.join(directory, "run.log")):
        return directory
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "generate.log"), "w") as log:
        subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "point-source.py"),
                "--output-directory",
                directory,
                "--verbosity",
                "text",
                "--amr-levels",
                str(levels),
                "--patch-size",
                str(VOLUMES_PER_PATCH_AXIS),
            ]
            + arguments,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    with open(os.path.join(directory, "run.log.tmp"), "w") as log:
        subprocess.run(
            ["./point-source"],
            cwd=directory,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    os.rename(
        os.path.join(directory, "run.log.tmp"), os.path.join(directory, "run.log")
    )
    return directory


def mesh_over_time(run_directory: str):
    """!
    Number of leaf cells after each time step

    @return [(float, int)]
      Time stamp and number of cells per time step.
    """
    result = []
    time_stamp = None
    cells = None
    with open(os.path.join(run_directory, "run.log"), errors="ignore") as log:
        for line in log:
            match = re.search(r"t=([0-9.e+-]+), new dt=", line)
            if match:
                if time_stamp is not None:
                    result.append((time_stamp, cells))
                time_stamp = float(match.group(1))
            match = re.search(r"#local cells=\((\d+)/\d+\)", line)
            if match:
                cells = int(match.group(1))
    result.append((time_stamp, cells))
    return result


def final_solution(run_directory: str, volumes_per_axis: int):
    """!
    Last snapshot, injected piecewise constant onto the regular grid

    @return (float, np.ndarray, int)
      Time stamp, array [unknown][y][x] and the number of mesh cells of the snapshot.
    """
    snapshots = read_snapshots(
        os.path.join(run_directory, f"solution-{SOLVER_NAME}.peano-patch-file")
    )
    time_stamp, files = snapshots[-1]
    volume_size = 1.0 / volumes_per_axis
    result = np.full((4, volumes_per_axis, volumes_per_axis), np.nan)
    cells = 0
    for file_name in files:
        parser = PatchFileParser(file_name, "", 0)
        parser.parse_file()
        for attributes in parser.unknown_attributes.values():
            dofs = attributes.dof
            for patch in attributes.cell_data:
                cells += 1
                values = patch.values.reshape(dofs, dofs, attributes.unknowns)
                factor = int(round(patch.size[0] / dofs / volume_size))
                x = int(round((patch.offset[0] + 0.5) / volume_size))
                y = int(round((patch.offset[1] + 0.5) / volume_size))
                block = np.repeat(np.repeat(values, factor, axis=0), factor, axis=1)
                for unknown in range(4):
                    result[
                        unknown, y : y + dofs * factor, x : x + dofs * factor
                    ] = block[..., unknown]
    return time_stamp, result, cells


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument(
        "--work-directory", required=True, help="Directory for the runs"
    )
    parser.add_argument(
        "--amr-levels",
        type=int,
        default=1,
        help="AMR levels of the AMR run (default: 1)",
    )
    parser.add_argument(
        "--patch-count",
        type=int,
        default=81,
        help="Mesh cells per axis of the regular start grid, a power of three (default: 81)",
    )
    parser.add_argument(
        "--end-time", type=float, default=0.1, help="Simulated time (default: 0.1)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Allowed maximum difference relative to the range (default: 0.02)",
    )
    args = parser.parse_args()

    volumes_per_axis = args.patch_count * VOLUMES_PER_PATCH_AXIS
    extra = [
        "--volume-size",
        str(1.001 / volumes_per_axis),
        "--end-time",
        str(args.end_time),
        "--plots",
        "5",
    ]
    os.makedirs(args.work_directory, exist_ok=True)
    regular = run(args.work_directory, 0, extra)
    adaptive = run(args.work_directory, args.amr_levels, extra)

    start_cells = args.patch_count**2
    timeline = mesh_over_time(adaptive)
    print(
        f"Mesh of the run with {args.amr_levels} AMR level(s) (regular start grid: {start_cells} cells)"
    )
    print("| time stamp | cells | volumes | coarsened volumes | refined volumes |")
    print("|------------|-------|---------|-------------------|-----------------|")
    previous = start_cells
    last_printed = None
    coarsened = refined = 0
    for time_stamp, cells in timeline:
        delta = (cells - previous) * VOLUMES_PER_PATCH_AXIS**2
        coarsened += max(-delta, 0)
        refined += max(delta, 0)
        if delta != 0:
            print(
                f"| {time_stamp:10.4f} | {cells:5d} | {cells * VOLUMES_PER_PATCH_AXIS**2:7d} | {max(-delta, 0):17d} | {max(delta, 0):15d} |"
            )
        previous = cells
    print(f"total coarsened {coarsened} volumes, refined {refined} volumes")

    _, reference, reference_cells = final_solution(regular, volumes_per_axis)
    time_stamp, solution, cells = final_solution(adaptive, volumes_per_axis)
    failures = []
    if reference_cells != start_cells:
        failures.append(
            f"the run without AMR holds {reference_cells} instead of {start_cells} cells"
        )
    if not cells < start_cells:
        failures.append(
            f"the mesh has not coarsened ({cells} cells at the end, {start_cells} at the start)"
        )
    if coarsened == 0:
        failures.append("the mesh never coarsened")
    print(
        f"final snapshot t={time_stamp:.4f}: {cells} cells ({100.0 * cells / start_cells:.0f}% of the regular grid)"
    )
    for name, unknown in [("density", 0), ("energy", 3)]:
        difference = np.abs(solution[unknown] - reference[unknown])
        value_range = np.nanmax(reference[unknown]) - np.nanmin(reference[unknown])
        maximum = np.nanmax(difference)
        rms = math.sqrt(np.nanmean(difference**2))
        print(
            f"{name}: range {value_range:.3f}, max difference {maximum:.3e}, rms difference {rms:.3e}"
        )
        if not np.isfinite(maximum) or maximum > args.tolerance * value_range:
            failures.append(
                f"{name} differs by {maximum:.3e}, which is more than {args.tolerance} of its range {value_range:.3f}"
            )
    if failures:
        print("FAILED")
        for failure in failures:
            print(" -", failure)
        sys.exit(1)
    print("PASSED")
