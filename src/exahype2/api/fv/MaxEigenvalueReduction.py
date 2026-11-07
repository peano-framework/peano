# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org


def add_max_eigenvalue_reduction(solver):
    """!

    Let the generated abstract solver class host the maximum eigenvalue of a time step

    The class collects the maximum over all patches and threads. This is
    a reduction over the whole mesh, hence it belongs to the solver and
    not to a cell. exahype2.api.fv.actionsets.ReduceMaxEigenvalue feeds it
    (reduceMaxEigenvalue()), while
    exahype2.api.fv.actionsets.ComputeGlobalCFLCondition reads and resets it
    (getAndResetMaxEigenvalue()) in a different place of the time step.
    Reading and resetting is one operation to avoid races.

    @param solver: exahype2.api.Solver
      The solver whose abstract class hosts the reduction.

    """
    solver.add_user_solver_includes(
        """
#include "tarch/multicore/BooleanSemaphore.h"
#include "tarch/multicore/Lock.h"
"""
    )
    # Brace initialisation: transient data, so the checkpoint meta data
    # (which lists attributes of the form "double name = value;") skips it
    solver.add_abstract_solver_attribute("double _maxEigenvalueOfTimeStep{0.0};")
    solver.add_abstract_solver_attribute(
        "tarch::multicore::BooleanSemaphore _maxEigenvalueSemaphore;"
    )
    solver.add_abstract_solver_method(
        "void reduceMaxEigenvalue(double eigenvalue);",
        lambda qualified_class_name: "void "
        + qualified_class_name
        + "::reduceMaxEigenvalue(double eigenvalue) {\n"
        + "  tarch::multicore::Lock lock(_maxEigenvalueSemaphore);\n"
        + "  _maxEigenvalueOfTimeStep = std::max(_maxEigenvalueOfTimeStep, eigenvalue);\n"
        + "}\n",
    )
    solver.add_abstract_solver_method(
        "double getAndResetMaxEigenvalue();",
        lambda qualified_class_name: "double "
        + qualified_class_name
        + "::getAndResetMaxEigenvalue() {\n"
        + "  tarch::multicore::Lock lock(_maxEigenvalueSemaphore);\n"
        + "  const double result       = _maxEigenvalueOfTimeStep;\n"
        + "  _maxEigenvalueOfTimeStep  = 0.0;\n"
        + "  return result;\n"
        + "}\n",
    )
