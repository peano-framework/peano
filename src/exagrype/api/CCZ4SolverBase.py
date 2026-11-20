""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

import enum
from typing import Dict, List, Union

import exahype2.api
import peano4.api
import peano4.api.taskgraph
import toolbox.blockstructured.api


class CCZ4SolverBase(exahype2.api.Solver):
    """!

    Everything the first-order and the second-order CCZ4 solvers share

    The base class hosts the CCZ4 parameters, the PDE terms (which delegate
    to the kernels in exagrype::ccz4) and the optional enforcement of the
    CCZ4 constraints. The initial condition is up to the user. The kernels
    always work on the 58 entries of CCZ4, see
    exagrype.api.FOCCZ4Solver.Unknowns, which they take as two pointers:
    one to the 25 primary and one to the 33 secondary entries. The two
    subclasses differ only in where the secondary entries live:

    - exagrype.api.FOCCZ4Solver evolves all 58 as unknowns. The secondary
      entries follow the primary ones, i.e. the pointers are Q and Q+25.
    - exagrype.api.FOSOCCZ4Solver evolves 25 unknowns, and the other 33
      entries are auxiliary variables, i.e. they live in separate memory.

    The subclasses therefore override the hooks _ncp_implementation() and
    _source_implementation(), which call the kernel of their formulation.
    The base class appends the advection terms to the non-conservative
    product if the numerical scheme does not add them
    (@ref exagrype.api.CCZ4SolverBase.CCZ4SolverBase.ncp_adds_advection()).

    @ref exagrype.api.FOCCZ4Solver, @ref exagrype.api.FOSOCCZ4Solver

    """

    NUMBER_OF_CCZ4_ENTRIES = 58
    """! Number of entries which the kernels in exagrype::ccz4 work on """

    class EnforceCCZ4Constraint(enum.Enum):
        """!

        When to push the solution back onto the algebraic CCZ4 constraints

        - No: never. This is the default.
        - OnlyFinalSolution: after each task which writes a new solution
          (exahype2.api.Solver.SolutionUpdateTaskTag), i.e. once per time
          step.
        - IntermediateSteps: in addition after each task which writes an
          intermediate solution (exahype2.api.Solver.IntermediateSolutionUpdateTag),
          i.e. after each Runge-Kutta stage. For single-step schemes such as
          the Finite Volumes, this is the same as OnlyFinalSolution.

        @ref exagrype.api.CCZ4SolverBase.CCZ4SolverBase.enforce_ccz4_constraint()

        """

        No = 0
        OnlyFinalSolution = 1
        IntermediateSteps = 2

    DefaultParameters = {
        "CCZ4LapseType": 0,
        "CCZ4c": 1.0,
        "CCZ4e": 1.0,
        "CCZ4f": 0.75,
        "CCZ4b": 1,
        "CCZ4s": 0,
        "CCZ4itau": 1.0,
        "CCZ4eta": 1.0,
        "CCZ4k1": 0.1,
        "CCZ4k2": 0.0,
        "CCZ4k3": 0.5,
        "CCZ4GLMc": 1.2,
        "CCZ4GLMd": 2.0,
        "CCZ4mu": 0.2,
    }
    """!

    Default values of the CCZ4 parameters, taken from the legacy ExaGRyPE
    gauge wave setup: harmonic slicing (CCZ4LapseType=0) and a shift which
    does not evolve (CCZ4s=0). The legacy black hole runs use 1+log slicing
    (CCZ4LapseType=1) and the Gamma driver (CCZ4s=1) in its shifting shift
    variant (CCZ4b=1, which only matters once the shift evolves). Integer
    values become int constants of the generated solver class, all others
    double constants.

    The integer parameters are switches:

    - CCZ4LapseType: 0 for harmonic slicing, 1 for 1+log slicing.
    - CCZ4b: variant of the Gamma driver, if the shift evolves. 1 is the
      shifting shift, i.e. the equations of the shift, of b and of the
      derivatives of the shift hold their advection terms. 0 is the
      non-shifting shift without these terms
      (exagrype::ccz4::ncpFirstOrder()).
    - CCZ4s: 1 evolves the shift (Gamma driver), 0 freezes it.

    Whether the first-order or the second-order formulation is solved is
    not a parameter: Pick exagrype.api.FOCCZ4Solver or
    exagrype.api.FOSOCCZ4Solver.

    Why there are 58 entries, why the switches are integers and why there
    are no parameters CCZ4ds and CCZ4xi is written down in
    src/exagrype/ccz4/adrs/0001-ccz4-has-58-entries-and-integer-switches.md.

    """

    Switches = ("CCZ4LapseType", "CCZ4b", "CCZ4s")
    """!

    The integer parameters which accept 0 and 1 only.

    """

    ADVECTION_VELOCITY = "shift"
    """!

    The variable which advects the unknowns, i.e. the velocity of the
    advection terms, whoever adds them: the lopsided advection of the
    numerical scheme or the non-conservative product
    (@ref exagrype.api.CCZ4SolverBase.CCZ4SolverBase.advected_quantities()).

    """

    def __init__(
        self,
        name: str,
        unknowns: Dict[str, int],
        parameters: Dict[str, Union[int, float]] = None,
    ):
        """!

        Create a CCZ4 solver

        @param name: str
          Name of the solver (@ref exahype2.api.Solver).

        @param unknowns: Dict[str, int]
          The evolved unknowns of the subclass (@ref exahype2.api.Solver).
          The subclass has to add all entries of CCZ4 that it does not evolve
          as auxiliary variables, so the entries add up to 58.

        @param parameters: Dict[str, Union[int, float]]
          CCZ4 parameters that differ from DefaultParameters, e.g.
          {"CCZ4k1": 0.0}. Keys which are not in DefaultParameters are
          rejected, as they most likely are typos. The switches
          (@ref exagrype.api.CCZ4SolverBase.CCZ4SolverBase.Switches) have
          to be the integers 0 or 1.

        The solver has no initial condition. Add one as for any other solver,
        e.g. solver.add_pde_term(exagrype.api.initialvalues.gauge_wave()), or
        your own exahype2.api.InitPointWisely.

        """
        super().__init__(name, unknowns=unknowns)

        self._enforce_ccz4_constraint = CCZ4SolverBase.EnforceCCZ4Constraint.No

        self._parameters = dict(self.DefaultParameters)
        for key, value in (parameters or {}).items():
            if key not in self.DefaultParameters:
                raise KeyError(
                    f"Solver {name}: unknown CCZ4 parameter {key!r}. "
                    f"Available parameters are {list(self.DefaultParameters)}."
                )
            if key in self.Switches and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value not in (0, 1)
            ):
                raise TypeError(
                    f"Solver {name}: CCZ4 parameter {key!r} is an integer "
                    f"switch, i.e. has to be 0 or 1, but got {value!r}"
                )
            self._parameters[key] = value

        self.add_abstract_solver_attribute(
            'static_assert(Dimensions == 3, "CCZ4 (ExaGRyPE) is three-dimensional only");'
        )
        for key, value in self._parameters.items():
            cpp_type = (
                "int" if isinstance(self.DefaultParameters[key], int) else "double"
            )
            self.add_abstract_solver_attribute(
                f"static constexpr {cpp_type} {key} = {value};"
            )

        self.add_user_solver_includes(
            """
#include "exagrype/ccz4/Kernels.h"
#include "exahype2/fd/CentralAdvection.h"
#include "exagrype/initialvalues/InitialValues.h"
#include "exagrype/initialvalues/ApplyTwoPunctures.h"
"""
        )

        self._non_conservative_product = None
        self._register_pde_terms()

    def advected_quantities(self) -> List[str]:
        """!

        The unknowns whose advection terms the kernels of exagrype::ccz4 leave out

        The kernels hold no advection terms @f$ \\beta^k \\partial _k Q_q @f$
        which need the derivative of the advected unknown q itself. This list
        is the only place which defines these unknowns. Either the numerical
        scheme adds their advection with lopsided stencils, e.g. via

        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        exahype2.api.rkfd4.CentralDifferencesWithRungeKutta(
          solver, patch_size, rk_order=4,
          right_hand_side_kernel = exahype2.api.rkfd4.CentralDifferencesWithRungeKutta.RightHandSideKernel.CentralDifferencesWithLopsidedAdvection,
          advection_velocity     = exagrype.api.FOCCZ4Solver.ADVECTION_VELOCITY,
          advected_quantities    = solver.advected_quantities(),
        )
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        These are all unknowns besides the conformal metric G, the lapse,
        the shift and phi: The advection terms of the metric, the lapse,
        the shift and phi are part of the algebraic source
        (exagrype::ccz4::sourceFirstOrder()), which expresses them through the
        auxiliary unknowns, i.e. the first derivatives.

        or the solver appends it to the non-conservative product, discretised
        with the central differences of the scheme
        (exahype2::fd::addCentralAdvection()), if the scheme adds none. The
        solver rejects a scheme which advects other quantities
        (@ref exagrype.api.CCZ4SolverBase.CCZ4SolverBase.ncp_adds_advection()).

        The advection of b and of the derivatives of the shift (dxShift,
        dyShift, dzShift) belongs to the shifting shift only, i.e. the
        routine leaves these unknowns out unless CCZ4s=1 and CCZ4b=1.

        The equation of b holds the time derivative of Ghat (the unknown
        Z), which comprises the advection of Ghat. The shifting shift
        subtracts this advection again, i.e. only the advection of b itself
        is left, which is added as for any other unknown. The non-shifting
        shift (CCZ4b=0) keeps the advection of Ghat in the equation of b.
        Who advects each quantity on its own cannot add it there, so the
        kernel does so with the derivatives it is given
        (src/exagrype/ccz4/adrs/0002-ccz4b-selects-the-shifting-or-the-non-shifting-shift.md).

        @return List[str]
          Names of unknowns
          (@ref exagrype.api.FOCCZ4Solver.FOCCZ4Solver.Unknowns).

        """
        excluded = {"G", "lapse", "shift", "phi"}
        if self._parameters["CCZ4b"] == 0 or self._parameters["CCZ4s"] == 0:
            excluded |= {"b", "dxShift", "dyShift", "dzShift"}
        return [name for name in self.Unknowns if name not in excluded]

    def ncp_adds_advection(self) -> bool:
        """!

        Whether the non-conservative product has to add the advection terms

        The advection terms @f$ \\beta^k \\partial _k Q_q @f$ of the unknowns
        of advected_quantities() are added either by the numerical
        scheme or by the non-conservative product, never by both. You choose
        the formulation (the solver class) and whether the Finite Differences
        use lopsided advection (the kernel of the numerical scheme). The
        solver decides which fits this choice, and
        generate_solver_realisation() registers the non-conservative product
        accordingly:

        - exahype2.api.rkfd4.CentralDifferencesWithRungeKutta (and its
          subclass exagrype.api.FD4WithKODissipation) with the kernel
          CentralDifferences adds no advection terms. The non-conservative
          product adds them, discretised with the central differences of
          the scheme.
        - The same scheme with CentralDifferencesWithLopsidedAdvection adds
          the advection itself, and the non-conservative product adds none.
          The scheme has to advect exactly
          @ref exagrype.api.CCZ4SolverBase.CCZ4SolverBase.advected_quantities()
          with ADVECTION_VELOCITY. Otherwise, it would add some terms twice or
          miss some, and the routine raises a ValueError.
        - The Finite Volumes of exahype2.api.fv add no advection terms, and
          the non-conservative product adds none either, i.e. they solve CCZ4
          without these advection terms, as before
          (@ref exagrype.api.FOCCZ4Solver). Whether the non-conservative
          product should add them is open.

        The rationale is written down in
        src/exagrype/ccz4/adrs/0003-the-solver-picks-who-adds-the-advection.md.

        @return bool
          True if the non-conservative product has to add the advection terms.

        """
        scheme = self.numerical_scheme
        match scheme:
            case (
                exahype2.api.fv.RusanovWithExplicitEuler()
                | exahype2.api.fv.RusanovWithOptimisticExplicitEuler()
            ):
                return False
            case exahype2.api.rkfd4.CentralDifferencesWithRungeKutta():
                Kernel = (
                    exahype2.api.rkfd4.CentralDifferencesWithRungeKutta.RightHandSideKernel
                )
                if scheme.right_hand_side_kernel == Kernel.CentralDifferences:
                    return True
                velocity, advected = self._advection_indices()
                if scheme.advection_velocity != velocity or sorted(
                    scheme.advected_quantities
                ) != sorted(advected):
                    raise ValueError(
                        f"Solver {self.name}: the lopsided advection of the "
                        f"numerical scheme advects the quantities "
                        f"{scheme.advected_quantities} with the velocity "
                        f"{scheme.advection_velocity}, but the kernels leave the "
                        f"quantities {advected} to the scheme, to be advected "
                        f"with the velocity {velocity}. Pass "
                        f"advection_velocity={type(self).__name__}.ADVECTION_VELOCITY "
                        f"and advected_quantities=solver.advected_quantities() "
                        f"to the scheme, or use its kernel CentralDifferences "
                        f"instead, which lets the solver add the advection."
                    )
                return False
            case _:
                raise TypeError(
                    f"Solver {self.name}: cannot decide who adds the advection "
                    f"terms for the numerical scheme {type(scheme).__name__}. "
                    "Supported are exahype2.api.fv.RusanovWithExplicitEuler, "
                    "exahype2.api.fv.RusanovWithOptimisticExplicitEuler and "
                    "exahype2.api.rkfd4.CentralDifferencesWithRungeKutta."
                )

    def _advection_indices(self):
        """!
        Indices of the velocity components (ADVECTION_VELOCITY) and of the
        advected unknowns (advected_quantities()) within Q, in the
        form both the lopsided advection of the scheme and
        exahype2::fd::addCentralAdvection() take them
        """
        resolve = exahype2.api.rkfd4.resolve_quantities
        return (
            resolve(self, [self.ADVECTION_VELOCITY]),
            resolve(self, self.advected_quantities()),
        )

    def enforce_ccz4_constraint(self, value: "CCZ4SolverBase.EnforceCCZ4Constraint"):
        """!

        Set when the solver pushes the solution back onto the CCZ4 constraints

        By default, the solver does not enforce the constraints. If you
        switch the enforcement on, generate_solver_realisation() inserts a
        task which applies exagrype::ccz4::enforceCCZ4constraints() to each
        volume or grid point after each tagged solution update of the
        numerical scheme.

        @param value: CCZ4SolverBase.EnforceCCZ4Constraint
          No, OnlyFinalSolution or IntermediateSteps.

        """
        if not isinstance(value, CCZ4SolverBase.EnforceCCZ4Constraint):
            raise TypeError(
                f"Solver {self.name}: enforce_ccz4_constraint() expects a "
                f"CCZ4SolverBase.EnforceCCZ4Constraint, but got {value!r}"
            )
        self._enforce_ccz4_constraint = value

    def generate_solver_realisation(
        self,
        exahype2_project,
        peano_project,
        setup_simulation,
        time_step,
        plot,
        mesh_adaptation,
        add_global_tasks=True,
    ):
        """!

        Create the solver's task graphs and insert the constraint enforcement

        The routine first registers the non-conservative product, with the
        advection terms if the numerical scheme does not add them
        (@ref exagrype.api.CCZ4SolverBase.CCZ4SolverBase.ncp_adds_advection()).
        The solver cannot do this earlier, as the numerical scheme is
        assigned after the solver has been created. Afterwards, the routine
        calls the superclass, which lets the grid construction, the
        numerical scheme and the plotter add their tasks and generates the
        solver class with the PDE terms. If the constraint enforcement is
        switched on
        (enforce_ccz4_constraint()), it afterwards inserts the enforcement
        after each tagged solution update of the numerical scheme. Which
        patch a tagged task writes and which guard it uses depends on the
        scheme, so the routine distinguishes the schemes explicitly
        (@ref page_exahype2_solver_manipulate_solver_steps). It raises an
        exception for any other scheme.

        All insertions work on the patch which the tagged tasks write and
        use the cell guard:

        - exahype2.api.fv.RusanovWithExplicitEuler: one task after Update
          cell.
        - exahype2.api.fv.RusanovWithOptimisticExplicitEuler: one task after
          both Update cell and Insert patch interior. Each cell runs exactly
          one of the two, so the task constrains every cell once, after its
          update.
        - exahype2.api.rkfd4.CentralDifferencesWithRungeKutta and its
          subclass exagrype.api.FD4WithKODissipation: one task after
          Complete time step and, for IntermediateSteps, one task after each
          Prepare stage k on the stage input.

        The scheme's log of the time stepping graph does not show the
        enforcement. With TEXT or VISUALISATION logging, the routine hence
        logs the graph once more.

        """
        self._register_non_conservative_product(self.ncp_adds_advection())
        super().generate_solver_realisation(
            exahype2_project,
            peano_project,
            setup_simulation,
            time_step,
            plot,
            mesh_adaptation,
            add_global_tasks,
        )
        Enforce = CCZ4SolverBase.EnforceCCZ4Constraint
        if self._enforce_ccz4_constraint == Enforce.No:
            return

        ## The patches which the tagged tasks write: The solution and, for
        ## Runge-Kutta, the stage input
        scheme = self.numerical_scheme
        match scheme:
            case (
                exahype2.api.fv.RusanovWithExplicitEuler()
                | exahype2.api.fv.RusanovWithOptimisticExplicitEuler()
            ):
                stage_input = None
            case exahype2.api.rkfd4.CentralDifferencesWithRungeKutta():
                stage_input = self._stage_input
            case _:
                raise TypeError(
                    f"Solver {self.name}: cannot enforce the CCZ4 constraints for "
                    f"the numerical scheme {type(scheme).__name__}. Supported are "
                    "exahype2.api.fv.RusanovWithExplicitEuler, "
                    "exahype2.api.fv.RusanovWithOptimisticExplicitEuler and "
                    "exahype2.api.rkfd4.CentralDifferencesWithRungeKutta."
                )

        ## One task after all writers of the new solution: The optimistic
        ## Finite Volumes write it in one of two tasks, depending on the cell
        self._insert_constraint_enforcement(
            time_step,
            self._tasks_of_this_solver(
                time_step, exahype2.api.Solver.SolutionUpdateTaskTag
            ),
            self._patch,
        )
        ## One task after each stage, as the stages run one after the other
        if (
            self._enforce_ccz4_constraint == Enforce.IntermediateSteps
            and stage_input is not None
        ):
            for anchor in self._tasks_of_this_solver(
                time_step, exahype2.api.Solver.IntermediateSolutionUpdateTag
            ):
                self._insert_constraint_enforcement(time_step, [anchor], stage_input)

        title = (
            f"Time stepping task graph for {self.name} with CCZ4 constraint enforcement"
        )
        if peano4.api.Logger().is_visualisation:
            time_step.visualise(title=title)
        if peano4.api.Logger().is_text:
            time_step.log_as_text(title=title)

    def _tasks_of_this_solver(self, time_step, tag):
        """!
        Tasks of time_step which carry tag and belong to this solver, as the
        graph holds the tasks of all solvers
        """
        return [
            task
            for task in time_step.find_tasks(tag)
            if task.name.startswith(self.name + "_")
        ]

    def _insert_constraint_enforcement(self, time_step, anchors, patch):
        """!

        Insert the enforcement of the CCZ4 constraints after anchors

        The routine creates a task which applies
        exagrype::ccz4::enforceCCZ4constraints() to each volume or grid point
        of patch, and inserts it directly after all anchors via
        peano4.api.taskgraph.Graph.insert_after(). The anchors' successors,
        e.g. the projection onto the faces, hence see the constrained data
        only. The task uses the solver's cell guard: If several anchors
        write patch, each cell is updated by one of them, and the task runs
        once per cell after all of them.

        @param time_step: peano4.api.taskgraph.Graph
          The time stepping task graph which holds the anchors.

        @param anchors: List[peano4.api.taskgraph.Task]
          The tasks which write patch. If the list is empty, the routine
          does nothing.

        @param patch: peano4.api.datamodel.Patch
          The cell data which the anchors write, i.e. the solution or a
          stage input.

        """
        if not anchors:
            return
        anchor_names = " and ".join(
            anchor.name[len(self.name) + 1 :] for anchor in anchors
        )
        task = peano4.api.taskgraph.Task(
            name=f"{self.name}_Enforce CCZ4 constraints after {anchor_names}",
            entity=peano4.api.taskgraph.Task.GridEntity.CELL,
        )
        ## READ_AND_WRITE keeps the data alive for later tasks
        task.use_cell(patch, peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
        task.add_action_set(
            toolbox.blockstructured.api.ApplyPointWiselyToPatch(
                patch=patch,
                implementation=self._enforcement_implementation(),
                guard=self.grid_selector.createCellGuard(),
                additional_includes=self._get_default_includes()
                + self.user_solver_includes,
            )
        )
        time_step.insert_after(anchors, task)

    def _enforcement_implementation(self) -> str:
        """!
        C++ which pushes the unknowns Q of one volume back onto the constraints.
        Hook for the subclasses, as the unknowns they evolve differ.
        """
        return "::exagrype::ccz4::enforceCCZ4constraints(Q);"

    @property
    def parameters(self) -> Dict[str, Union[int, float]]:
        """!
        The CCZ4 parameters the solver is generated with (read-only copy).
        """
        return dict(self._parameters)

    def _register_pde_terms(self):
        """!

        Register the source and the maximum eigenvalue

        The terms are stateless, i.e. static functions: they only use the
        compile-time constants of the solver and the kernels of
        exagrype::ccz4. Both the Finite Volumes (exahype2.api.fv) and the
        Finite Differences (exahype2.api.rkfd4) call stateless terms. The
        non-conservative product is not registered here: which one fits
        depends on the numerical scheme, which is assigned after the solver
        has been created, so generate_solver_realisation() registers it
        (@ref exagrype.api.CCZ4SolverBase.CCZ4SolverBase._register_non_conservative_product()).

        """
        self.add_pde_term(
            exahype2.api.VolumetricSource(
                implementation=self._source_implementation(), stateless=True
            )
        )
        self.add_pde_term(
            exahype2.api.MaxEigenvalue(
                implementation=self._max_eigenvalue_implementation(), stateless=True
            )
        )

    def _register_non_conservative_product(self, advection: bool):
        """!

        Register the non-conservative product, with or without the advection terms

        generate_solver_realisation() calls this routine, as the
        non-conservative product depends on the numerical scheme
        (@ref exagrype.api.CCZ4SolverBase.CCZ4SolverBase.ncp_adds_advection()).
        The term calls the kernel of the formulation (_ncp_implementation())
        and, if advection is set, adds the advection terms of
        advected_quantities() with ADVECTION_VELOCITY
        (_advection_implementation()). It replaces the term of a former
        realisation, so a project can be generated several times.

        Any other non-conservative product, i.e. one added through
        add_pde_term(), is rejected: The solver would replace it silently, and
        it could add advection terms which the scheme adds as well, or miss
        them (src/exagrype/ccz4/adrs/0003-the-solver-picks-who-adds-the-advection.md).
        Whoever needs another non-conservative product derives a solver class
        and overrides _ncp_implementation().

        @param advection: bool
          Whether the term adds the advection terms.

        """
        name = exahype2.api.NonconservativeProduct.NAME
        foreign = [
            term
            for term in self.pde_terms
            if term.name == name and term is not self._non_conservative_product
        ]
        if foreign:
            raise ValueError(
                f"Solver {self.name}: a non-conservative product has been added "
                f"with add_pde_term(). {type(self).__name__} registers its "
                f"non-conservative product itself when it is realised, as the "
                f"advection terms depend on the numerical scheme "
                f"(ncp_adds_advection()), and would replace yours. Derive a "
                f"solver class from {type(self).__name__} and override "
                f"_ncp_implementation() instead."
            )
        implementation = self._ncp_implementation()
        if advection:
            implementation += self._advection_implementation()
        self._non_conservative_product = exahype2.api.NonconservativeProduct(
            implementation=implementation, stateless=True
        )
        self.add_pde_term(self._non_conservative_product)

    def _ncp_implementation(self) -> str:
        """!

        C++ which calls the non-conservative product kernel

        The Finite Volume kernels hand over the jump deltaQ along one normal,
        the fourth-order Finite Differences of exahype2.api.rkfd4 the
        derivative along one normal times the mesh width. This is what
        exagrype::ccz4::ncpFirstOrder() and ncpSecondOrder() expect.
        Hook for the subclasses, as they select the formulation.

        """
        raise NotImplementedError

    def _advection_implementation(self) -> str:
        """!

        C++ which adds the advection terms to the non-conservative product

        The term adds @f$ \\beta^n \\partial _n Q_q @f$ for the normal n of the
        call and each unknown q of advected_quantities(),
        discretised with the derivative the scheme hands over
        (exahype2::fd::addCentralAdvection()). The indices are those which
        the lopsided advection of a scheme gets, so either way the same
        terms are added. Both formulations hold the unknowns in Q, and Q
        holds the shift.

        """
        velocity, quantities = self._advection_indices()
        return f"""
  {{
    constexpr int velocity[] = {{{", ".join(str(i) for i in velocity)}}};
    constexpr int quantities[] = {{{", ".join(str(i) for i in quantities)}}};
    ::exahype2::fd::addCentralAdvection(Q, deltaQ, BTimesDeltaQ, normal, velocity, quantities, {len(quantities)});
  }}
"""

    def _source_implementation(self) -> str:
        """!
        C++ which calls the source term kernel. Hook for the subclasses.
        """
        raise NotImplementedError

    def _max_eigenvalue_implementation(self) -> str:
        return """
  ::exagrype::ccz4::maxEigenvalue(Q, normal, CCZ4e, CCZ4GLMc, CCZ4GLMd, lambdaMax);
"""
