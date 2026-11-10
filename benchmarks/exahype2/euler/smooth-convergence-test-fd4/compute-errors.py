# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

Error and convergence order of the entropy wave benchmark

Reads the last snapshot of each run directory passed, compares the density
with the exact solution, and prints the maximum and the root mean square
error per run plus the convergence orders between subsequent runs. Pass the
runs from the coarsest to the finest mesh.

The patch files hold the patches' offsets and sizes with seven significant
digits only. Below an error of roughly 1e-7, this precision would dominate
the error. The script hence snaps the offsets and sizes onto the regular
mesh over the domain [-0.1,0.1]^d.

"""
import argparse
import math
import os
import sys

import numpy as np

from tarch.plotter.api.input.PatchFileParser import PatchFileParser
from tarch.plotter.api.notebook import read_snapshots

DOMAIN_OFFSET = -0.1
DOMAIN_WIDTH = 0.2
VELOCITY = 1.0


def exact_density(x, t):
    return 1.0 + 0.2 * np.sin(
        2.0 * np.pi * (x - VELOCITY * t - DOMAIN_OFFSET) / DOMAIN_WIDTH
    )


def snap(offset, size):
    """! Snap a patch's offset and size onto the regular mesh over the domain."""
    size = np.array([DOMAIN_WIDTH / round(DOMAIN_WIDTH / s) for s in size])
    offset = np.round((np.array(offset) - DOMAIN_OFFSET) / size) * size + DOMAIN_OFFSET
    return offset, size


def errors(run_directory):
    """!
    Return (time stamp, grid cells per axis, max error, rms error) of the last
    snapshot of a run.
    """
    snapshots = read_snapshots(
        os.path.join(run_directory, "solution-EulerFD4Smooth.peano-patch-file")
    )
    time_stamp, files = snapshots[-1]
    max_error = 0.0
    squared_error = 0.0
    number_of_grid_cells = 0
    min_grid_cell_size = math.inf
    for file_name in files:
        parser = PatchFileParser(file_name, "", 0)
        parser.parse_file()
        for attributes in parser.unknown_attributes.values():
            dofs = attributes.dof
            for patch in attributes.cell_data:
                offset, size = snap(patch.offset, patch.size)
                h = size / dofs
                min_grid_cell_size = min(min_grid_cell_size, h[0])
                shape = [dofs] * attributes.dimensions + [attributes.unknowns]
                # The fastest index is x, i.e. values[(z,) y, x, unknown]
                density = patch.values.reshape(shape)[..., 0]
                x = offset[0] + (np.arange(dofs) + 0.5) * h[0]
                error = np.abs(density - exact_density(x, time_stamp))
                if not np.all(np.isfinite(error)):
                    return time_stamp, None, math.nan, math.nan
                max_error = max(max_error, float(np.max(error)))
                squared_error += float(np.sum(error * error))
                number_of_grid_cells += error.size
    return (
        time_stamp,
        round(DOMAIN_WIDTH / min_grid_cell_size),
        max_error,
        math.sqrt(squared_error / number_of_grid_cells),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument(
        "run_directory", nargs="+", help="Runs, from the coarsest to the finest mesh"
    )
    parser.add_argument(
        "--minimum-order",
        type=float,
        default=None,
        help="Exit with a non-zero status if the order of the maximum error between the two finest runs is below this value or not finite",
    )
    args = parser.parse_args()

    print(
        f"{'run':40} {'t':>8} {'cells/axis':>10} {'max error':>12} {'rms error':>12} {'order (max)':>12} {'order (rms)':>12}"
    )
    previous = None
    for run_directory in args.run_directory:
        time_stamp, cells, max_error, rms_error = errors(run_directory)
        orders = ["", ""]
        if previous is not None and cells is not None and previous[0] is not None:
            ratio = cells / previous[0]
            orders = [
                f"{math.log(previous[1] / max_error) / math.log(ratio):.2f}",
                f"{math.log(previous[2] / rms_error) / math.log(ratio):.2f}",
            ]
        print(
            f"{run_directory:40} {time_stamp:8.4f} {str(cells):>10} {max_error:12.4e} {rms_error:12.4e} {orders[0]:>12} {orders[1]:>12}"
        )
        previous = (cells, max_error, rms_error)
        last_order = float(orders[0]) if orders[0] != "" else math.nan

    if args.minimum_order is not None:
        if not last_order >= args.minimum_order:
            print(f"FAILED: order {last_order} is below {args.minimum_order}")
            sys.exit(1)
        print(f"PASSED: order {last_order} is at least {args.minimum_order}")
