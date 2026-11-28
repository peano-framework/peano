# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org

## DSLHyPE kernels of the fourth-order Finite Differences
#
# The functions below are never executed as Python. DSLHyPE parses their
# source (@ref exahype2.api.dslhype.Parser) and translates them into C++
# (@ref exahype2.api.rkfd4.CentralDifferencesWithRungeKutta). They hence
# refer to symbols the parser sets rather than Python imports:
#
# - Template parameters: NC (grid cells per axis within a patch), H (halo
#   size, at least two), NumberOfUnknowns, NumberOfAuxiliaryVariables.
# - Functors for the non-stateless code: flux, nonconservativeProduct,
#   volumetricSource, maxEigenvalue. An if over a functor's name holds if
#   and only if the solver has the corresponding PDE term.
#
# Do not add comments within the functions: DSLHyPE parses the comments, too,
# and gets confused by comments which mention the names of variables.
#
# ## Right-hand side
#
# rightHandSideWithFD42D() and rightHandSideWithFD43D() evaluate the
# right-hand side
#
# @f$ \partial _t Q = - \nabla \cdot F(Q) - \sum _d B_d(Q) \partial _d Q + S(Q) @f$
#
# over a patch with a halo of H grid cells (patchData.QIn) and write it
# into patchData.QOut, which holds the unknowns only. They do not scale with
# the time step size nor add the old solution, as the outcome is one estimate
# of a Runge-Kutta scheme.
#
# - The divergence of the flux and the derivatives in the non-conservative
#   product use the fourth-order central difference
#   @f$ \partial _x f_i \approx (f_{i-2} - 8 f_{i-1} + 8 f_{i+1} - f_{i+2}) / (12 \Delta x) @f$.
#   The mesh width is the cell size h over NC.
# - The non-conservative product is handed deltaQ, i.e. the derivative of
#   all unknowns and auxiliary variables times the mesh width, exactly like
#   the Finite Volumes of exahype2.api.fv hand over the jump across a face.
#   For auxiliary variables which are derivatives of unknowns, the scheme
#   may overwrite rows of deltaQ by compact second derivatives
#   (@ref exahype2.api.rkfd4.CompactSecondDerivatives). It relies on the
#   assignments of delta_x, delta_y and delta_z below.
# - The stencils reach two grid cells into the neighbouring patches, i.e.
#   they need a halo H of at least two. The kernels are written for an
#   arbitrary halo H, so subclasses of the numerical scheme can add wider
#   stencils (@ref exagrype.api.FD4WithKODissipation).
# - The PDE terms are handed the patch centre and the patch size rather
#   than the grid cell's centre and size, as DSLHyPE has no notion of the
#   individual grid cell's position. The Finite Volume kernels of
#   exahype2.api.fv share this simplification.
#
# The functions replace the legacy timeStepWithFD4() of
# exahype2.solvers.rkfd.fd4, which was three-dimensional only, passed
# deltaQ instead of the non-conservative product along x into the update,
# and had a (commented out) Kreiss-Oliger dissipation built in. There is no
# dissipation in these kernels anymore: @ref exagrype.api.FD4WithKODissipation
# adds it as a separate kernel on top.
#
# ## Maximum eigenvalue
#
# maxEigenvalueOverPatch2D() and maxEigenvalueOverPatch3D() reduce the
# maximum eigenvalue over all grid cells of a patch without halo
# (patchData.QIn) into patchData.maxEigenvalue. The adaptive time stepping
# derives the next time step size from it.


def rightHandSideWithFD42D(patchData: "::exahype2::CellData<double, double>&") -> "void":
    N = Integer("patchData.numberOfCells")
    t = DataBlock([[0, N]], "patchData.t", "double")
    dt = DataBlock([[0, N]], "patchData.dt", "double")
    h = DataBlock([[0, N]], "patchData.cellSize", "tarch::la::Vector<2, double>")
    x = DataBlock([[0, N]], "patchData.cellCentre", "tarch::la::Vector<2, double>")

    input = DataBlock([[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [-H, NC + H], [-H, NC + H], [0, N]], "patchData.QIn")
    output = DataBlock([[0, NumberOfUnknowns], [0, NC], [0, NC], [0, N]], "patchData.QOut")
    output = 0.0 * input[0:NumberOfUnknowns, 0:NC, 0:NC]

    if volumetricSource:
        source = DataBlock([[0, NumberOfUnknowns], [0, NC], [0, NC], [0, N]], eval("volumetricSource", input[:, 0:NC, 0:NC], x, t))
        output = output + source

    if flux:
        flux_x = DataBlock([[0, NumberOfUnknowns], [-2, NC + 2], [0, NC], [0, N]], eval("flux", input[:, -2:NC+2, 0:NC], x, 0))
        flux_y = DataBlock([[0, NumberOfUnknowns], [0, NC], [-2, NC + 2], [0, N]], eval("flux", input[:, 0:NC, -2:NC+2], x, 1))
        output = output - (1.0 / 12.0) * NC / h(0) * (flux_x[:, -2:NC-2, 0:NC] - 8.0 * flux_x[:, -1:NC-1, 0:NC] + 8.0 * flux_x[:, 1:NC+1, 0:NC] - flux_x[:, 2:NC+2, 0:NC])
        output = output - (1.0 / 12.0) * NC / h(1) * (flux_y[:, 0:NC, -2:NC-2] - 8.0 * flux_y[:, 0:NC, -1:NC-1] + 8.0 * flux_y[:, 0:NC, 1:NC+1] - flux_y[:, 0:NC, 2:NC+2])

    if nonconservativeProduct:
        delta_x = (1.0 / 12.0) * (input[:, -2:NC-2, 0:NC] - 8.0 * input[:, -1:NC-1, 0:NC] + 8.0 * input[:, 1:NC+1, 0:NC] - input[:, 2:NC+2, 0:NC])
        ncp_x = DataBlock([[0, NumberOfUnknowns], [0, NC], [0, NC], [0, N]], eval("nonconservativeProduct", input[:, 0:NC, 0:NC], delta_x, x, h, t, dt, 0))
        output = output - NC / h(0) * ncp_x

        delta_y = (1.0 / 12.0) * (input[:, 0:NC, -2:NC-2] - 8.0 * input[:, 0:NC, -1:NC-1] + 8.0 * input[:, 0:NC, 1:NC+1] - input[:, 0:NC, 2:NC+2])
        ncp_y = DataBlock([[0, NumberOfUnknowns], [0, NC], [0, NC], [0, N]], eval("nonconservativeProduct", input[:, 0:NC, 0:NC], delta_y, x, h, t, dt, 1))
        output = output - NC / h(1) * ncp_y


def rightHandSideWithFD43D(patchData: "::exahype2::CellData<double, double>&") -> "void":
    N = Integer("patchData.numberOfCells")
    t = DataBlock([[0, N]], "patchData.t", "double")
    dt = DataBlock([[0, N]], "patchData.dt", "double")
    h = DataBlock([[0, N]], "patchData.cellSize", "tarch::la::Vector<3, double>")
    x = DataBlock([[0, N]], "patchData.cellCentre", "tarch::la::Vector<3, double>")

    input = DataBlock([[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [-H, NC + H], [-H, NC + H], [-H, NC + H], [0, N]], "patchData.QIn")
    output = DataBlock([[0, NumberOfUnknowns], [0, NC], [0, NC], [0, NC], [0, N]], "patchData.QOut")
    output = 0.0 * input[0:NumberOfUnknowns, 0:NC, 0:NC, 0:NC]

    if volumetricSource:
        source = DataBlock([[0, NumberOfUnknowns], [0, NC], [0, NC], [0, NC], [0, N]], eval("volumetricSource", input[:, 0:NC, 0:NC, 0:NC], x, t))
        output = output + source

    if flux:
        flux_x = DataBlock([[0, NumberOfUnknowns], [-2, NC + 2], [0, NC], [0, NC], [0, N]], eval("flux", input[:, -2:NC+2, 0:NC, 0:NC], x, 0))
        flux_y = DataBlock([[0, NumberOfUnknowns], [0, NC], [-2, NC + 2], [0, NC], [0, N]], eval("flux", input[:, 0:NC, -2:NC+2, 0:NC], x, 1))
        flux_z = DataBlock([[0, NumberOfUnknowns], [0, NC], [0, NC], [-2, NC + 2], [0, N]], eval("flux", input[:, 0:NC, 0:NC, -2:NC+2], x, 2))
        output = output - (1.0 / 12.0) * NC / h(0) * (flux_x[:, -2:NC-2, 0:NC, 0:NC] - 8.0 * flux_x[:, -1:NC-1, 0:NC, 0:NC] + 8.0 * flux_x[:, 1:NC+1, 0:NC, 0:NC] - flux_x[:, 2:NC+2, 0:NC, 0:NC])
        output = output - (1.0 / 12.0) * NC / h(1) * (flux_y[:, 0:NC, -2:NC-2, 0:NC] - 8.0 * flux_y[:, 0:NC, -1:NC-1, 0:NC] + 8.0 * flux_y[:, 0:NC, 1:NC+1, 0:NC] - flux_y[:, 0:NC, 2:NC+2, 0:NC])
        output = output - (1.0 / 12.0) * NC / h(2) * (flux_z[:, 0:NC, 0:NC, -2:NC-2] - 8.0 * flux_z[:, 0:NC, 0:NC, -1:NC-1] + 8.0 * flux_z[:, 0:NC, 0:NC, 1:NC+1] - flux_z[:, 0:NC, 0:NC, 2:NC+2])

    if nonconservativeProduct:
        delta_x = (1.0 / 12.0) * (input[:, -2:NC-2, 0:NC, 0:NC] - 8.0 * input[:, -1:NC-1, 0:NC, 0:NC] + 8.0 * input[:, 1:NC+1, 0:NC, 0:NC] - input[:, 2:NC+2, 0:NC, 0:NC])
        ncp_x = DataBlock([[0, NumberOfUnknowns], [0, NC], [0, NC], [0, NC], [0, N]], eval("nonconservativeProduct", input[:, 0:NC, 0:NC, 0:NC], delta_x, x, h, t, dt, 0))
        output = output - NC / h(0) * ncp_x

        delta_y = (1.0 / 12.0) * (input[:, 0:NC, -2:NC-2, 0:NC] - 8.0 * input[:, 0:NC, -1:NC-1, 0:NC] + 8.0 * input[:, 0:NC, 1:NC+1, 0:NC] - input[:, 0:NC, 2:NC+2, 0:NC])
        ncp_y = DataBlock([[0, NumberOfUnknowns], [0, NC], [0, NC], [0, NC], [0, N]], eval("nonconservativeProduct", input[:, 0:NC, 0:NC, 0:NC], delta_y, x, h, t, dt, 1))
        output = output - NC / h(1) * ncp_y

        delta_z = (1.0 / 12.0) * (input[:, 0:NC, 0:NC, -2:NC-2] - 8.0 * input[:, 0:NC, 0:NC, -1:NC-1] + 8.0 * input[:, 0:NC, 0:NC, 1:NC+1] - input[:, 0:NC, 0:NC, 2:NC+2])
        ncp_z = DataBlock([[0, NumberOfUnknowns], [0, NC], [0, NC], [0, NC], [0, N]], eval("nonconservativeProduct", input[:, 0:NC, 0:NC, 0:NC], delta_z, x, h, t, dt, 2))
        output = output - NC / h(2) * ncp_z


def maxEigenvalueOverPatch2D(patchData: "::exahype2::CellData<double, double>&") -> "void":
    N = Integer("patchData.numberOfCells")
    x = DataBlock([[0, N]], "patchData.cellCentre", "tarch::la::Vector<2, double>")
    maxEigenvalues = DataBlock([[0, N]], "patchData.maxEigenvalue", "double")
    input = DataBlock([[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [0, NC], [0, NC], [0, N]], "patchData.QIn")

    if maxEigenvalue:
        eigenvalues_x = DataBlock([[0, 1], [0, NC], [0, NC], [0, N]], eval("maxEigenvalue", input, x, 0))
        eigenvalues_y = DataBlock([[0, 1], [0, NC], [0, NC], [0, N]], eval("maxEigenvalue", input, x, 1))
        eigenvaluesPerGridCell = max(eigenvalues_x, eigenvalues_y)
        maxEigenvalues = max(eigenvaluesPerGridCell)


def maxEigenvalueOverPatch3D(patchData: "::exahype2::CellData<double, double>&") -> "void":
    N = Integer("patchData.numberOfCells")
    x = DataBlock([[0, N]], "patchData.cellCentre", "tarch::la::Vector<3, double>")
    maxEigenvalues = DataBlock([[0, N]], "patchData.maxEigenvalue", "double")
    input = DataBlock([[0, NumberOfUnknowns + NumberOfAuxiliaryVariables], [0, NC], [0, NC], [0, NC], [0, N]], "patchData.QIn")

    if maxEigenvalue:
        eigenvalues_x = DataBlock([[0, 1], [0, NC], [0, NC], [0, NC], [0, N]], eval("maxEigenvalue", input, x, 0))
        eigenvalues_y = DataBlock([[0, 1], [0, NC], [0, NC], [0, NC], [0, N]], eval("maxEigenvalue", input, x, 1))
        eigenvalues_z = DataBlock([[0, 1], [0, NC], [0, NC], [0, NC], [0, N]], eval("maxEigenvalue", input, x, 2))
        eigenvaluesXY = max(eigenvalues_x, eigenvalues_y)
        eigenvaluesPerGridCell = max(eigenvaluesXY, eigenvalues_z)
        maxEigenvalues = max(eigenvaluesPerGridCell)
