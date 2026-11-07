# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import os

import exahype2.api.dslhype as dslhype


## DSLHyPE kernel for the linear combinations of Runge-Kutta
#
# The function below is never executed as Python. DSLHyPE parses its source
# (@ref exahype2.api.dslhype.Parser) and translates it into a C++ function
# template (see generate_linear_combination_kernel()). It refers to symbols
# the parser sets rather than Python imports:
#
# - Template parameters: NumberOfDoFs, NumberOfUnknowns,
#   NumberOfAuxiliaryVariables, NumberOfEstimates.
#
# The kernel computes
#
# @f$ Q_{out} = Q_{in} + \Delta t \sum _{j} w_j K_j @f$
#
# per degree of freedom. The auxiliary variables are copied from the input.
# It does not know anything about the layout of the degrees of freedom in
# space: it requires only that each cell holds NumberOfDoFs degrees of
# freedom with the unknowns followed by the auxiliary variables per degree of
# freedom (AoS), and that the estimates hold the unknowns only. Finite
# Differences and DG schemes alike can hence use it.
#
# Each DataBlock's last dimension is the one DSLHyPE realises through an
# array of pointers. For the estimates, it enumerates the NumberOfEstimates
# estimates which enter the sum, i.e. estimates[j] points to the j-th of them,
# and weights[j] is its weight. For the input and the output, it is of size
# one. Input and output may point to the same array.
#
# This kernel replaces the legacy computeLinearCombination() and
# computeFinalLinearCombination() of exahype2.solvers.rkfd.fd4, which took
# the weights from the solver and assumed a three-dimensional patch.
def linearCombinationOfRungeKuttaEstimates(
    solution: "double**",
    estimates: "double**",
    output: "double**",
    weights: "const double*",
    timeStepSize: "double",
) -> "void":
    dt = Float("timeStepSize")
    Q = DataBlock(
        [[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [0, NumberOfDoFs], [0, 1]],
        "solution",
    )
    K = DataBlock(
        [[0, NumberOfUnknowns], [0, NumberOfDoFs], [0, NumberOfEstimates]], "estimates"
    )
    out = DataBlock(
        [[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [0, NumberOfDoFs], [0, 1]],
        "output",
    )
    W = DiagonalMatrix(NumberOfEstimates, "weights")
    out = Q[:, :, :]
    dot(W, K[0:NumberOfUnknowns, :, :], contribution)
    out[0:NumberOfUnknowns, :, 0] = out[0:NumberOfUnknowns, :, 0] + dt * contribution


LINEAR_COMBINATION_KERNEL_NAME = "linearCombinationOfRungeKuttaEstimates"
LINEAR_COMBINATION_KERNEL_FILE = "rungekutta_linear_combination"


def generate_linear_combination_kernel(namespace, output_directory):
    """!

    Translate linearCombinationOfRungeKuttaEstimates() into C++

    The kernel ends up in kernels/rungekutta_linear_combination.h in the
    output directory. It is a function template over NumberOfDoFs,
    NumberOfUnknowns, NumberOfAuxiliaryVariables and NumberOfEstimates within
    namespace. It does not depend on any solver, nor on the Runge-Kutta
    scheme (the weights and the number of estimates enter at the call site),
    so several solvers within one project can share it. Each of them calls
    this routine, but only the first call generates the file: The DSLHyPE
    translation appends to existing files, i.e. a second generation would
    define the function template twice.

    @param namespace: [str]
      Namespace of the generated function template, usually the Peano
      project's namespace.

    @param output_directory: str
      Directory the kernels/ subdirectory is created in, usually the Peano
      project's directory.

    """
    header = os.path.join(
        output_directory, "kernels", LINEAR_COMBINATION_KERNEL_FILE + ".h"
    )
    if os.path.exists(header):
        return

    dslhype.translate_dslhype_function(
        function=linearCombinationOfRungeKuttaEstimates,
        function_namespace=namespace,
        template_parameters=[
            dslhype.SyntaxTree.Argument("NumberOfDoFs", dslhype.SyntaxTree.TInteger()),
            dslhype.SyntaxTree.Argument(
                "NumberOfUnknowns", dslhype.SyntaxTree.TInteger()
            ),
            dslhype.SyntaxTree.Argument(
                "NumberOfAuxiliaryVariables", dslhype.SyntaxTree.TInteger()
            ),
            dslhype.SyntaxTree.Argument(
                "NumberOfEstimates", dslhype.SyntaxTree.TInteger()
            ),
        ],
        functor_parameters=[],
        pde_terms=[],
        output_file_name=LINEAR_COMBINATION_KERNEL_FILE,
        target=dslhype.Target.Serial,
        stateless=False,
        output_directory=output_directory,
    )
    include_implementation_into_header(
        os.path.join(output_directory, "kernels"), LINEAR_COMBINATION_KERNEL_FILE
    )


def include_implementation_into_header(directory: str, file_name: str):
    """!

    Make a generated kernel header pull in the template definitions

    The translation of a DSLHyPE function yields the declaration in
    file_name.h and the definition of the function template in
    file_name.cpph, which itself includes the header. We therefore guard the
    header and let it include the definitions at its end.

    @param directory: str
      Directory of the kernels.

    @param file_name: str
      Name of the kernel without suffix.

    """
    header = os.path.join(directory, file_name + ".h")
    with open(header, "r") as file:
        content = file.read()
    with open(header, "w") as file:
        file.write("#pragma once\n" + content + f'\n#include "{file_name}.cpph"\n')


def create_linear_combination_call(
    namespace,
    number_of_dofs,
    unknowns,
    auxiliary_variables,
    weights,
    solution,
    estimates,
    output,
    time_step_size,
):
    """!

    C++ code which invokes the kernel from generate_linear_combination_kernel()

    The code is a block of its own. It sets up the arrays of pointers the
    kernel expects and stores the weights as compile-time constants.

    Only the stages with a non-zero weight enter the kernel call. This is
    not an optimisation only: The estimates of the stages which a
    combination does not need may be uninitialised, e.g. the estimates of
    the later stages before the first time step has completed. Multiplying
    them with zero would still yield NaN if they hold NaN. If all weights
    are zero, the code copies the solution into the output.

    @param namespace: [str]
      Namespace of the kernel, see generate_linear_combination_kernel().

    @param number_of_dofs: int
      Degrees of freedom per cell.

    @param unknowns: int
      Number of unknowns per degree of freedom.

    @param auxiliary_variables: int
      Number of auxiliary variables per degree of freedom.

    @param weights: [float]
      One weight per Runge-Kutta stage, e.g. a row of
      exahype2.api.rungekutta.ButcherTableau.weight_matrix().

    @param solution: str
      C++ expression for the pointer to the input data.

    @param estimates: str
      C++ expression for the pointer to the estimates of all stages. They
      are stored one after the other, each with number_of_dofs*unknowns
      entries.

    @param output: str
      C++ expression for the pointer to the output data. May equal solution.

    @param time_step_size: str
      C++ expression for the time step size.

    """
    stages = [stage for stage, weight in enumerate(weights) if weight != 0.0]
    if not stages:
        if solution == output:
            return "{}"
        return f"""{{
    std::copy_n( {solution}, {number_of_dofs * (unknowns + auxiliary_variables)}, {output} );
  }}"""

    stage_pointers = ", ".join(
        f"{estimates} + {stage * number_of_dofs * unknowns}" for stage in stages
    )
    weight_values = ", ".join(repr(float(weights[stage])) for stage in stages)
    return f"""{{
    static constexpr double weights[{len(stages)}] = {{ {weight_values} }};
    double* solutionPointers[1] = {{ {solution} }};
    double* estimatePointers[{len(stages)}] = {{ {stage_pointers} }};
    double* outputPointers[1] = {{ {output} }};
    ::{"::".join(namespace)}::{LINEAR_COMBINATION_KERNEL_NAME}<
      {number_of_dofs},
      {unknowns},
      {auxiliary_variables},
      {len(stages)}
    >(solutionPointers, estimatePointers, outputPointers, weights, {time_step_size});
  }}"""
