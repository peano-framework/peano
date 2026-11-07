# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from __future__ import annotations

import re
from typing import Dict, List

import peano4.api.datamodel
import peano4.api.taskgraph
import toolbox.blockstructured.api


class AuxiliaryVariables:
    """!

    A set of auxiliary variables of a solver

    Auxiliary variables are quantities such as the sea floor, a mesh
    distortion or a density which live in every Finite Volume (or Finite
    Difference point) next to the unknowns, but which are not evolved by the
    PDE. A solver can have an arbitrary number of such sets, each of them
    vector-valued, e.g. ten bathymetry entries and another five entries
    describing a mesh distortion. Each set is held by its own cell data
    object, i.e. its own patch, rather than by the patch of the unknowns.

    Create a set with a name and a dictionary which maps the names of the
    parameters onto their multiplicities,

        bathymetry = exahype2.api.AuxiliaryVariables("Bathymetry", {"b": 1})

    and attach it to a solver via exahype2.api.Solver.add_auxiliary_variables().
    The name is also the name of a struct within the generated abstract
    solver, so you can write bathymetry[Bathymetry::b] in your C++ code.

    ## Impact on the C++ signatures

    Every PDE term of a solver with auxiliary variables receives, directly
    behind each Q (and each deltaQ), one pointer per attached set. The
    pointers come in the order in which the sets have been attached. The
    pointer is named after the set (first letter in lower case), the pointers
    behind deltaQ carry the prefix delta. For the set above and the flux, the
    signature therefore reads

        flux(const double* Q, const double* bathymetry, const Vector& x, int normal, double* F)

    and the signature of the non-conservative product is

        nonconservativeProduct(Q, bathymetry, deltaQ, deltaBathymetry, x, ...)

    Solvers without auxiliary variables keep their original signatures.

    ## Ownership and sharing

    The first solver a set is attached to owns it. Only the owner's
    initPointWisely() receives the set as writable argument and initialises
    it. Further solvers can attach the same set object. They share the data,
    i.e. they do not hold an additional copy, and initPointWisely() sees the
    set as const.

    @remark Naming
      These quantities used to be called material parameters. "Material"
      suggests that they are invariant, but they do not have to be: an
      auxiliary variable is any quantity that lives next to the unknowns and
      is not evolved by the PDE terms of the solver. It may well change over
      time if some other mechanism updates it. The current implementation
      writes the variables once during the setup, i.e. they are constant in
      time, but this is a restriction of the implementation, not part of the
      concept.

    Besides periodic boundaries, the halo layer of the volumes at the
    domain boundary is not available for auxiliary variables.

    """

    def __init__(self, name: str, variables: Dict[str, int]):
        """!

        @param name: str
          Name of the set. Has to be a valid C++ identifier starting with a
          capital letter. It feeds into the names of the data structures, the
          name of the struct which hosts the positions of the variables
          within the set, and - with a lower-case first letter - into the
          name of the arguments in the PDE term signatures.

        @param variables: dict
          Maps the name of each parameter onto its multiplicity.

        """
        if not re.fullmatch(r"[A-Z][A-Za-z0-9_]*", name):
            raise ValueError(
                f"The name of a set of auxiliary variables has to be a valid C++ identifier starting with a capital letter, but got {name!r}"
            )
        if not isinstance(variables, dict) or not variables:
            raise TypeError(
                "variables has to be a non-empty dict which maps variable names onto multiplicities"
            )
        self._name = name
        self._variables: Dict[str, int] = dict(variables)
        self._positions: Dict[str, int] = {}
        offset = 0
        for variable_name, multiplicity in self._variables.items():
            if multiplicity < 1:
                raise ValueError(
                    f"Auxiliary variable {variable_name} of {name} has multiplicity {multiplicity}"
                )
            self._positions[variable_name] = offset
            offset += multiplicity
        self._size = offset
        self._owner = None

        # Created by create_data_structures()
        self.patch: peano4.api.datamodel.Patch = None
        self.patch_overlap_current_time_step: peano4.api.datamodel.Patch = None
        self.patch_overlap_update: peano4.api.datamodel.Patch = None
        self._patch_size = None
        self._overlap = None
        self._added_to = set()

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> int:
        """! Total number of entries of the set, i.e. the sum of all multiplicities """
        return self._size

    @property
    def variables(self) -> Dict[str, int]:
        return dict(self._variables)

    @property
    def variable_positions(self) -> Dict[str, int]:
        """! Position of each named variable within the set """
        return dict(self._positions)

    @property
    def owner(self):
        """! The solver which initialises the set, i.e. the first one it was attached to """
        return self._owner

    @property
    def argument_name(self) -> str:
        """! Name of the argument of the PDE terms """
        return self._name[0].lower() + self._name[1:]

    @property
    def delta_argument_name(self) -> str:
        """! Name of the argument behind deltaQ """
        return "delta" + self._name

    def variable_names_per_entry(self) -> List[str]:
        """!
        One name per entry of the set, qualified by the set's name. A
        variable of multiplicity k yields k names with an index.
        """
        names = []
        for variable_name, multiplicity in self._variables.items():
            if multiplicity == 1:
                names.append(f"{self._name}.{variable_name}")
            else:
                names += [f"{self._name}.{variable_name}[{j}]" for j in range(multiplicity)]
        return names

    def variables_declaration(self) -> str:
        """! C++ struct which hosts the positions of the variables within the set """
        constants = "\n".join(
            f"      static constexpr int {variable_name} = {position};"
            for variable_name, position in self._positions.items()
        )
        return (
            f"    /**\n     * Positions of the variables of the auxiliary variables {self._name}\n     */\n"
            f"    struct {self._name} {{\n{constants}\n    }};"
        )

    def create_data_structures(self, patch_size: int, overlap: int, grid_selector):
        """!

        Create the cell and face data of the set

        The routine is idempotent, as a set may be shared by several solvers:
        the first call creates the data, any further call checks that the
        numerical schemes of the solvers agree on the patch size and halo.

        @param patch_size: int
          Number of volumes per axis within a patch.

        @param overlap: int
          Width of the halo layer.

        @param grid_selector: peano4.api.toolbox.GridSelector
          Selects the mesh entities which hold data.

        """
        if self.patch is not None:
            if self._patch_size != patch_size or self._overlap != overlap:
                raise ValueError(
                    f"The auxiliary variables {self._name} are shared by solvers with different patch sizes or halos: "
                    f"{self._patch_size}/{self._overlap} vs {patch_size}/{overlap}"
                )
            return
        self._patch_size = patch_size
        self._overlap = overlap
        identifier = self._name + "AuxiliaryVariables"
        self.patch = peano4.api.datamodel.Patch(
            (patch_size, patch_size, patch_size), self._size, identifier
        )
        self.patch_overlap_current_time_step = peano4.api.datamodel.Patch(
            (2 * overlap, patch_size, patch_size), self._size, identifier + "CurrentTimeStep"
        )
        self.patch_overlap_update = peano4.api.datamodel.Patch(
            (2 * overlap, patch_size, patch_size), self._size, identifier + "Update"
        )
        for patch in (
            self.patch,
            self.patch_overlap_current_time_step,
            self.patch_overlap_update,
        ):
            patch.grid_selector = grid_selector
            patch.generator = peano4.api.datamodel.PatchToDoubleArrayWithSmartPointer(
                patch, "double"
            )
        self.patch_overlap_current_time_step.generator.merge_method_definition = (
            toolbox.blockstructured.api.get_face_merge_implementation(
                self.patch_overlap_current_time_step
            )
        )
        self.patch_overlap_current_time_step.generator.includes += """
#include "peano4/utils/Loop.h"
"""
        self.patch_overlap_current_time_step.generator.send_condition = grid_selector.createFaceGuard()
        self.patch_overlap_current_time_step.generator.receive_and_merge_condition = grid_selector.createFaceGuard()
        self.patch_overlap_update.generator.send_condition = "false"
        self.patch_overlap_update.generator.receive_and_merge_condition = "false"

    def add_to_datamodel(self, peano_project):
        """! Register the data with the project (once, even if the set is shared) """
        if id(peano_project) in self._added_to:
            return
        self._added_to.add(id(peano_project))
        peano_project.datamodel.add_cell(self.patch)
        peano_project.datamodel.add_face(self.patch_overlap_current_time_step)
        peano_project.datamodel.add_face(self.patch_overlap_update)

    def __str__(self) -> str:
        return f"{self._name}: {self._variables}"


def split_signature(signature: str) -> List[str]:
    """! Split a C++ parameter list at the commas that are not within template brackets """
    return [
        parameter.strip()
        for parameter in re.split(r",(?![^<]*>)", signature)
        if parameter.strip()
    ]


def extend_signature(signature: str, solver) -> str:
    """!

    Add the pointers to the auxiliary variables of a solver to a PDE term's
    signature

    Each argument named Q is followed by one pointer per set, each argument
    named deltaQ is followed by one pointer to the jump of each set. The
    pointers are non-const iff Q is non-const (the initial condition) and the
    solver owns the set.

    @param signature: str
      The signature without auxiliary variables

    @param solver: exahype2.api.Solver

    """
    sets = solver.auxiliary_variables
    if not sets:
        return signature
    result = []
    for parameter in split_signature(signature):
        result.append(parameter)
        name = parameter.split()[-1]
        if name == "Q":
            writable = not parameter.startswith("const")
            for auxiliary in sets:
                qualifier = "" if writable and auxiliary.owner is solver else "const "
                result.append(f"{qualifier}double* __restrict__ {auxiliary.argument_name}")
        elif name == "deltaQ":
            for auxiliary in sets:
                result.append(f"const double* __restrict__ {auxiliary.delta_argument_name}")
    return ", ".join(result)


def forwarding_arguments(signature: str, solver) -> List[str]:
    """!

    Arguments to call the version of a PDE term with auxiliary variables
    from a caller that holds the original signature

    The caller's Q (and deltaQ) points to a volume in a buffer which hosts
    the unknowns followed by the entries of all auxiliary variables (the
    layout the compute kernels work on). The auxiliary variables are slices
    of this buffer.

    """
    arguments = []
    for parameter in split_signature(signature):
        name = parameter.split()[-1]
        arguments.append(name)
        if name in ("Q", "deltaQ"):
            for auxiliary in solver.auxiliary_variables:
                arguments.append(f"{name} + {solver.auxiliary_variable_offset(auxiliary)}")
    return arguments


def register_auxiliary_variables(solver, peano_project, patch_size: int, overlap: int):
    """!

    Create and register the data of all auxiliary variables of a solver

    The persistent data of a set (its patch and the face overlaps) exists only
    once, even if several solvers share it. Each solver gets its own
    temporary patch with halo per set (solver._auxiliary_variable_halos, a
    dict from the name of the set to the patch), so concurrently running
    solvers do not overwrite each other's halos.

    """
    solver._auxiliary_variable_halos = {}
    for auxiliary in solver.auxiliary_variables:
        auxiliary.create_data_structures(patch_size, overlap, solver.grid_selector)
        auxiliary.add_to_datamodel(peano_project)
        halo = peano4.api.datamodel.Patch(
            (patch_size + 2 * overlap, patch_size + 2 * overlap, patch_size + 2 * overlap),
            auxiliary.size,
            auxiliary.name + "AuxiliaryVariablesWithHaloOf" + solver.name,
        )
        halo.grid_selector = solver.grid_selector
        halo.generator = peano4.api.datamodel.PatchToDoubleArrayWithSmartPointer(halo, "double")
        peano_project.datamodel.add_cell(halo)
        solver._auxiliary_variable_halos[auxiliary.name] = halo


def add_reconstruction_of_auxiliary_variables(task, solver):
    """!

    Make a reconstruction task of a solver fill the halos of the solver's
    auxiliary variables

    @param task: peano4.api.taskgraph.Task
      The task which also reconstructs the unknowns.

    """
    Access = peano4.api.taskgraph.Task.DataAccess
    for auxiliary in solver.auxiliary_variables:
        halo = solver._auxiliary_variable_halos[auxiliary.name]
        task.use_cell(auxiliary.patch, Access.READ_AND_WRITE)
        task.use_cell(halo, Access.WRITE)
        task.use_face(auxiliary.patch_overlap_current_time_step, Access.READ_AND_WRITE)
        task.add_action_set(
            toolbox.blockstructured.api.ReconstructPatchWithHalo(
                patch=auxiliary.patch,
                patch_overlap=auxiliary.patch_overlap_current_time_step,
                reconstructed_patch=halo,
                guard=solver.grid_selector.createCellGuard(),
                add_assertions_to_halo_exchange=False,
                additional_includes=solver._get_default_includes() + solver.user_solver_includes,
            )
        )


def use_halos_of_auxiliary_variables(task, solver, access=None):
    """!

    Let a task (typically the update of the cells) read the halos of the auxiliary variables

    @param access: peano4.api.taskgraph.Task.DataAccess
      How the task uses the halos. Defaults to READ_AND_DISCARD. Pass another
      qualifier if the halos have to survive the task, e.g. because a later
      task of the same chain reads them, too.

    """
    if access is None:
        access = peano4.api.taskgraph.Task.DataAccess.READ_AND_DISCARD
    for auxiliary in solver.auxiliary_variables:
        task.use_cell(solver._auxiliary_variable_halos[auxiliary.name], access)


def volumes_expression(patch_size: int, halo: int) -> str:
    """! C++ expression for the number of volumes of a patch with halo """
    edge = f"({patch_size} + 2 * {halo})"
    return f"({edge} * {edge} * (Dimensions == 3 ? {edge} : 1))"


def code_to_assemble_kernel_input(solver, buffer: str, unknowns: str, halo_size: int, parameters_of=None) -> str:
    """!

    C++ code which assembles the input buffer of the compute kernels

    The kernels work on one buffer that holds the unknowns of a volume
    followed by the entries of all sets of auxiliary variables. The code
    defines a std::vector<double> named buffer. It is empty if the solver has
    no auxiliary variables.

    @param unknowns: str
      C++ expression for the unknowns of the patch with halo.

    @param parameters_of: function
      Maps a set onto the C++ expression of its patch with halo. Defaults to
      the halo patches of the solver.

    """
    if not solver.auxiliary_variables:
        return ""
    if parameters_of is None:
        parameters_of = lambda auxiliary: f"fineGridCell{solver._auxiliary_variable_halos[auxiliary.name].name}.value"
    volumes = volumes_expression(solver.numerical_scheme.patch_size, halo_size)
    stride = solver.unknowns + solver.number_of_auxiliary_variables
    copies = "".join(
        f"""
    for (int k=0; k<{auxiliary.size}; k++) {{
      {buffer}[volume*{stride} + {solver.auxiliary_variable_offset(auxiliary)} + k] = {parameters_of(auxiliary)}[volume*{auxiliary.size} + k];
    }}"""
        for auxiliary in solver.auxiliary_variables
    )
    return f"""
  const int {buffer}Volumes = {volumes};
  std::vector<double> {buffer}( {buffer}Volumes * {stride}, 0.0 );
  for (int volume=0; volume<{buffer}Volumes; volume++) {{
    for (int k=0; k<{solver.unknowns}; k++) {{
      {buffer}[volume*{stride} + k] = {unknowns}[volume*{solver.unknowns} + k];
    }}{copies}
  }}
"""


def code_to_allocate_kernel_output(solver, buffer: str) -> str:
    """! C++ code which defines the output buffer (a std::vector<double>) of the compute kernels """
    if not solver.auxiliary_variables:
        return ""
    volumes = volumes_expression(solver.numerical_scheme.patch_size, 0)
    stride = solver.unknowns + solver.number_of_auxiliary_variables
    return f"""
  const int {buffer}Volumes = {volumes};
  std::vector<double> {buffer}( {buffer}Volumes * {stride}, 0.0 );
"""


def code_to_copy_back_kernel_output(solver, buffer: str, unknowns: str) -> str:
    """! C++ code which copies the unknowns of the output buffer back into the patch """
    if not solver.auxiliary_variables:
        return ""
    stride = solver.unknowns + solver.number_of_auxiliary_variables
    return f"""
  for (int volume=0; volume<{buffer}Volumes; volume++) {{
    for (int k=0; k<{solver.unknowns}; k++) {{
      {unknowns}[volume*{solver.unknowns} + k] = {buffer}[volume*{stride} + k];
    }}
  }}
"""


def add_auxiliary_variables_to_setup(
    solver, initial_condition_task, project_on_faces_task, project_patch_onto_faces, roll_over_updated_face
):
    """!

    Let the setup of a solver write its auxiliary variables

    Auxiliary variables are written once, by their owner: the owner
    initialises them in its initial condition task and projects them onto the
    faces, so the halos can be reconstructed later on. A solver that merely
    shares a set reads it after the owner has written it.

    @param project_patch_onto_faces: class
      The action set that projects a patch onto the faces
      (exahype2.api.fv.actionsets.ProjectPatchOntoFaces).

    @param roll_over_updated_face: class
      The action set that rolls the projection over
      (exahype2.api.fv.actionsets.RollOverUpdatedFace).

    """
    Access = peano4.api.taskgraph.Task.DataAccess
    includes = solver._get_default_includes() + solver.user_solver_includes
    for auxiliary in solver.auxiliary_variables:
        # The setup is created before the numerical scheme adds the data to the project
        auxiliary.create_data_structures(
            solver.numerical_scheme.patch_size,
            solver.numerical_scheme.halo_size,
            solver.grid_selector,
        )
        if auxiliary.owner is solver:
            initial_condition_task.use_cell(auxiliary.patch, Access.WRITE)
            project_on_faces_task.use_cell(auxiliary.patch, Access.READ_AND_WRITE)
            project_on_faces_task.use_face(auxiliary.patch_overlap_update, Access.TEMPORARY)
            project_on_faces_task.use_face(auxiliary.patch_overlap_current_time_step, Access.WRITE)
            project_on_faces_task.add_action_set(
                project_patch_onto_faces(
                    auxiliary.patch,
                    auxiliary.patch_overlap_update,
                    guard=solver.grid_selector.createCellGuard(),
                    includes=includes,
                )
            )
            project_on_faces_task.add_action_set(
                roll_over_updated_face(
                    auxiliary.patch_overlap_current_time_step,
                    auxiliary.patch_overlap_update,
                    guard=solver.grid_selector.createFaceGuard(),
                    includes=includes,
                )
            )
            auxiliary._setup_initial_condition_task = initial_condition_task
        else:
            owners_task = getattr(auxiliary, "_setup_initial_condition_task", None)
            if owners_task is None:
                raise ValueError(
                    f"Solver {solver.name} shares the auxiliary variables {auxiliary.name} with "
                    f"{auxiliary.owner.name}, but the setup of {auxiliary.owner.name} has not been created yet. "
                    f"Add {auxiliary.owner.name} to the project before {solver.name}."
                )
            initial_condition_task.add_dependency(owners_task)
            initial_condition_task.use_cell(auxiliary.patch, Access.READ_AND_DISCARD)
            project_on_faces_task.use_cell(auxiliary.patch, Access.READ_AND_WRITE)
