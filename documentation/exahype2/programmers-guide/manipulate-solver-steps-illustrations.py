# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

Create the illustrations of the page "Manipulate an existing solver's steps"

Run from anywhere via

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python3 documentation/exahype2/programmers-guide/manipulate-solver-steps-illustrations.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The script writes two PNGs next to it:

- manipulate-solver-steps-rusanov-original.png: the time stepping task graph
  of exahype2.api.fv.RusanovWithExplicitEuler as the scheme builds it;
- manipulate-solver-steps-rusanov-with-step.png: the same graph built by
  the subclass RusanovWithPostprocessing, which inserts one step after the
  task carrying exahype2.api.Solver.SolutionUpdateTaskTag. The subclass is
  the example of the page.

Both graphs are constructed from the actual numerical scheme, so the
pictures always match the code. The kernels this requires are generated
into a temporary directory.

"""
import os
import sys
import tempfile
import types

Here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(Here, "..", "..", "..", "src")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import peano4.api
import peano4.api.taskgraph
import exahype2.api
import toolbox.blockstructured.api


class RusanovWithPostprocessing(exahype2.api.fv.RusanovWithExplicitEuler):
    """! The example of the page: insert a step after each solution update"""

    def construct_output(self, exahype2_project, peano_project, time_step, add_global_tasks=True):
        super().construct_output(exahype2_project, peano_project, time_step, add_global_tasks)
        solver = self._solver
        for anchor in time_step.find_tasks(exahype2.api.Solver.SolutionUpdateTaskTag):
            if not anchor.name.startswith(solver.name + "_"):
                continue
            task = peano4.api.taskgraph.Task(
                name=f"{solver.name}_Postprocess solution",
                entity=peano4.api.taskgraph.Task.GridEntity.CELL,
            )
            task.use_cell(
                solver._patch, peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE
            )
            task.add_action_set(
                toolbox.blockstructured.api.ApplyPointWiselyToPatch(
                    patch=solver._patch,
                    implementation="Q[0] = std::max(Q[0], 0.9);",
                    guard=solver.grid_selector.createCellGuard(),
                    additional_includes=solver._get_default_includes()
                    + solver.user_solver_includes,
                )
            )
            time_step.insert_after(anchor, task)


def plot_time_step(scheme, file_name, title):
    # The Python API writes a log file when the program terminates. Keep it
    # out of the working directory.
    peano4.api.Logger().file_name = os.path.join(
        tempfile.gettempdir(), "manipulate-solver-steps-illustrations-log.md"
    )
    with tempfile.TemporaryDirectory() as directory:
        peano_project = peano4.api.Project(
            ["illustration"], "illustration", directory=directory
        )
        solver = exahype2.api.Solver("Euler", 4, 0)
        solver.add_pde_term(exahype2.api.Flux(stateless=False))
        solver.add_pde_term(exahype2.api.MaxEigenvalue(stateless=False))
        solver.numerical_scheme = scheme(solver, 9)
        exahype2_project = types.SimpleNamespace(periodic_BC=[True, True])
        time_step = peano4.api.taskgraph.Graph()
        solver.numerical_scheme.construct_output(
            exahype2_project, peano_project, time_step
        )
    time_step.visualise(title=title)
    output = os.path.join(Here, file_name)
    plt.gcf().savefig(output, dpi=90)
    plt.close(plt.gcf())
    print("Wrote", output)


if __name__ == "__main__":
    plot_time_step(
        exahype2.api.fv.RusanovWithExplicitEuler,
        "manipulate-solver-steps-rusanov-original.png",
        "RusanovWithExplicitEuler",
    )
    plot_time_step(
        RusanovWithPostprocessing,
        "manipulate-solver-steps-rusanov-with-step.png",
        "RusanovWithPostprocessing",
    )
