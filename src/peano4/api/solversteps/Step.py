# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import peano4.api.solversteps.UserActionSet

from peano4.api.solversteps.StepToActionSets import StepToActionSets
from peano4.api.solversteps.StepToObserver import StepToObserver
from peano4.api.solversteps.SynchroniseActionSets import SynchroniseActionSets, get_task_groups


class Step:
    """!

    One step

    Most generic version of a solver step. A solverstep describes what Peano 4
    should do when it runs over the grid. As users have to specify which pieces
    of data from the data model are used by the solver step, it can handle all
    the user data stack flow.

    This class also is responsible to generate all function signatures of all
    the steps, observers and action sets. In the Peano terminology, a step
    corresponds to an observer. For any given step, we run through the mesh
    and pass the traversal automaton an observer which accepts events (aka
    state transition observations).

    It is the observer's job to then distribute these observations, broken
    down into grid actions such as "touch vertex first time", among the action
    sets that are tied to this observer. In Python you can have one action set
    and associate it with several observers. In the C++ code, we will flatten
    such a relation, i.e. create the corresponding action set multiple times.

    @see @ref peano4.api.solversteps.ActionSet.ActionSet

    name: String
      This name is used to create the underlying action set later on
    """

    def __init__(self, 
                 name, 
                 add_user_defined_actions=True, 
                 ):
        """
        By default, the step generates at least one action set for the user.
        However, you can disable this behaviour by passing False to
        add_user_defined_actions. Alternatively, use remove_all_action_sets()
        on the step.
        """
        self.name = name
        self.cell_data = []
        self.face_data = []
        self.vertex_data = []
        # Optional side-channel, empty unless a caller populates it: maps a
        # submodel already in cell_data/face_data/vertex_data to the
        # peano4.api.taskgraph.Task.DataAccess qualifier it was declared
        # with on the originating Task. use_cell()/use_face()/use_vertex()
        # below deliberately do not touch these - they exist purely so
        # peano4.api.taskgraph.compiler.Sequential.translate() can carry a
        # Task's qualifiers over onto the Step it compiles into, for
        # reporting (@see peano4.api.taskgraph.compiler.ReportCompiledStepSequence)
        # to display; nothing about how this Step's own generated Observer
        # behaves depends on them.
        self.cell_data_access = {}
        self.face_data_access = {}
        self.vertex_data_access = {}
        if add_user_defined_actions:
            self.action_sets = [peano4.api.solversteps.UserActionSet(name)]
        else:
            self.action_sets = []
        self.action_set_generator = StepToActionSets(self)
        self.observer_generator = StepToObserver(self)

    def get_observer_class_name(self) -> str:
        """!

        Class name of the observer this step is translated into. This is
        the step's name, cut down to 64 characters plus a consecutive number
        if it is longer (@see peano4.api.output.Helper.shortened_class_name()),
        as the class name also is the observer's file name. Everybody
        referring to the observer class, such as the generated runners, has
        to use this getter rather than the step's name.

        """
        # Imported here, as peano4.api.output in turn imports this package
        from peano4.api.output.Helper import shortened_class_name

        return shortened_class_name(self.name, max_length=64)

    def remove_all_actions(self):
        """
        Each step holds a set of actions. They describe what the step actually
        should do whenever it loads a vertex, runs into a cell, and so forth. By
        default, the step holds one user-defined actions which means
        that a stub is generated where users can insert their functionality. You
        can remove this one (or any other one added so far) with this routine.
        """
        self.action_sets = []

    def __sort(self):
        """!

        Simple bubble sort the action sets, such that their descend_invocation_order
        is preserved.

        @see @ref peano4.api.solversteps.ActionSet.ActionSet.descend_invocation_order

        """
        for i in range(0, len(self.action_sets)):
            for j in range(i + 1, len(self.action_sets)):
                if (
                    self.action_sets[i].descend_invocation_order
                    > self.action_sets[j].descend_invocation_order
                ):
                    tmp = self.action_sets[i]
                    self.action_sets[i] = self.action_sets[j]
                    self.action_sets[j] = tmp

    def add_action_set(self, action_set):
        """!

        Add a new action set to this mesh traversal step

        Each step holds a set of action_sets. They describe what the step actually
        should do whenever it loads a vertex, runs into a cell, and so forth. By
        default, the step holds one user-defined action_set (UserActionSet) which means
        that a stub is generated where users can insert their functionality. It is
        important in which order you add your action_sets: The whole tree traversal
        is a top-down/depth-first tree traversal. So all the enter, create, ...
        operations of a action_set are invoked exactly in the order you add them to
        the step. All the delete, leave, ... operations are invokved in reversed
        order.

        I used to have a flag append_at_end with the default True, but this one is
        now replaced, as each action set knows its priority.
        """
        self.action_sets.append(action_set)
        self.__sort()

    def set_project(self, project):
        """!

        Tie step to project

        You don't have to call this. It is automatically called once you add a step
        to the project's steps.
        """
        self.project = project
        self.namespace = project.namespace
        self.subdirectory = project.subdirectory
        self.temporary_subdirectory = project.temporary_subdirectory

    def use_cell(self, submodel):
        self.cell_data.append(submodel)

    def use_face(self, submodel):
        self.face_data.append(submodel)

    def use_vertex(self, submodel):
        self.vertex_data.append(submodel)

    def highest_descend_invocation_order(self):
        """!

        Return max descend_invocation_order over all action sets associated with this step.

        @see @ref peano4.api.solversteps.ActionSet.ActionSet.descend_invocation_order

        """
        self.__sort()
        if len(self.action_sets) == 0:
            return 0
        return self.action_sets[-1].descend_invocation_order

    def lowest_descend_invocation_order(self):
        """!

        Return minimum descend_invocation_order over all action sets associated with this step.

        @see @ref peano4.api.solversteps.ActionSet.ActionSet.descend_invocation_order

        """
        self.__sort()
        if len(self.action_sets) == 0:
            return 0
        return self.action_sets[0].descend_invocation_order

    def get_vertex_operations_signature(self):
        result = ["marker", "const peano4::datamanagement::VertexMarker&"]
        for i in self.vertex_data:
            result += ["fineGridVertex" + i.name, i.get_full_qualified_type() + "&"]
        for i in self.vertex_data:
            result += ["coarseGridVertices" + i.name, i.get_enumeration_type() + ""]
        for i in self.face_data:
            result += ["coarseGridFaces" + i.name, i.get_enumeration_type() + ""]
        for i in self.cell_data:
            result += ["coarseGridCell" + i.name, i.get_full_qualified_type() + "&"]
        return result

    def get_face_operations_signature(self):
        result = ["marker", "const peano4::datamanagement::FaceMarker&"]
        for i in self.vertex_data:
            result += ["fineGridVertices" + i.name, i.get_enumeration_type() + ""]
        for i in self.face_data:
            result += ["fineGridFace" + i.name, i.get_full_qualified_type() + "&"]
        for i in self.vertex_data:
            result += ["coarseGridVertices" + i.name, i.get_enumeration_type() + ""]
        for i in self.face_data:
            result += ["coarseGridFaces" + i.name, i.get_enumeration_type() + ""]
        for i in self.cell_data:
            result += ["coarseGridCell" + i.name, i.get_full_qualified_type() + "&"]
        return result

    def get_cell_operations_signature(self):
        result = ["marker", "const peano4::datamanagement::CellMarker&"]
        for i in self.vertex_data:
            result += ["fineGridVertices" + i.name, i.get_enumeration_type() + ""]
        for i in self.face_data:
            result += ["fineGridFaces" + i.name, i.get_enumeration_type() + ""]
        for i in self.cell_data:
            result += ["fineGridCell" + i.name, i.get_full_qualified_type() + "&"]
        for i in self.vertex_data:
            result += ["coarseGridVertices" + i.name, i.get_enumeration_type() + ""]
        for i in self.face_data:
            result += ["coarseGridFaces" + i.name, i.get_enumeration_type() + ""]
        for i in self.cell_data:
            result += ["coarseGridCell" + i.name, i.get_full_qualified_type() + "&"]
        return result

    def get_touch_cell_signature(self):
        result = ["marker", "const peano4::datamanagement::CellMarker&"]
        for i in self.vertex_data:
            result += ["fineGridVertices" + i.name, i.get_enumeration_type() + ""]
        for i in self.face_data:
            result += ["fineGridFaces" + i.name, i.get_enumeration_type() + ""]
        for i in self.cell_data:
            result += ["fineGridCell" + i.name, i.get_full_qualified_type() + "&"]

        for i in self.vertex_data:
            result += ["coarseGridVertices" + i.name, i.get_enumeration_type() + ""]
        for i in self.face_data:
            result += ["coarseGridFaces" + i.name, i.get_enumeration_type() + ""]
        for i in self.cell_data:
            result += ["coarseGridCell" + i.name, i.get_full_qualified_type() + "&"]

        return result

    def construct_output(self, output):
        """
        Each solver step basically gives us two big files/classes: A class which
        handles the actual data movements, i.e. is an implementation of the tree
        visitor pattern over the Peano4 tree. The other type is an interface which
        provides plugin points into the created data transitions from a user's
        perspective.

        Markers of type
        @ref peano4.api.solversteps.SynchroniseActionSets.SynchroniseActionSets
        yield no C++ code. They only separate the task groups which we hand
        over to the observer
        (@ref peano4.api.solversteps.SynchroniseActionSets.get_task_groups()).
        """

        included_action_sets = []
        index_of_action_set = {}
        for action_set in self.action_sets:
            if isinstance(action_set, SynchroniseActionSets):
                continue
            full_qualified_action_set_name = self.action_set_generator.construct_output(
                output, action_set
            )
            index_of_action_set[id(action_set)] = len(included_action_sets)
            included_action_sets.append(full_qualified_action_set_name)

        task_groups = [
            [index_of_action_set[id(action_set)] for action_set in group]
            for group in get_task_groups(self.action_sets)
        ]
        self.observer_generator.construct_output(output, included_action_sets, task_groups)

    def copy_action_sets_from_other_step(self, other_step):
        self.action_sets += other_step.action_sets
