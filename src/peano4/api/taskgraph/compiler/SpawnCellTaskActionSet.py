# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from peano4.api.solversteps.ActionSet import ActionSet


class SpawnCellTaskActionSet(ActionSet):
    """!

    Producer-side wrapper around an action set, which spawns touchCellFirstTime() and touchCellLastTime() as tasks

    The wrapper delegates all events to the wrapped action set with the
    following exceptions:

    - touchCellFirstTime() empties the cell's array of children tasks. It
      then takes two task numbers from the task enumerator: one for the task
      of touchCellFirstTime(), one for the task of touchCellLastTime(), and
      stores them in the cell's task labels. It writes the second number into
      the entry of the cell's parent which belongs to the cell
      (children tasks). If the wrapped action set's touchCellFirstTime()
      has a body and its guard holds, the wrapper spawns the body as a task
      with the first number. This task depends on the first task of the
      parent cell, and on the task spawned by the previous wrapped action set
      of the same producer. If the guard does not hold, the first label is
      set to tarch::multicore::TaskEnumerator::NoNumber.
    - touchCellLastTime() spawns the body as a task with the second number
      if the wrapped action set's touchCellLastTime() has a body. The task
      depends on the cell's first task, on the last tasks of all (up to 3^d)
      children and on the last task of the previous wrapped action set. If
      the guard does not hold, the wrapper spawns an empty task with the
      number, as the parent might wait for this number.
    - createCell() furthermore initialises both task labels with NoNumber.

    Both task numbers are reserved in touchCellFirstTime(), as the children
    are visited before the parent's touchCellLastTime(), and they have to
    tell their parent which task the parent has to wait for. The task
    enumerator is a static attribute of the task labels, i.e. it is
    shared by all producers and consumers which use these labels.

    The spawned task captures its environment by value ([=]). Cell data
    managed via smart pointers (e.g.
    @ref peano4.api.datamodel.PatchToDoubleArrayWithSmartPointer) is copied
    shallowly, i.e. the task writes into the cell's data. Plain arrays are
    copied deeply, i.e. writes are lost.

    Enumerators over vertices and faces hold pointers into Peano's stacks,
    which are invalid once the event has terminated. The task
    therefore captures each enumerator which its body might use as
    peano4::datamanagement::FaceHeapDataEnumerator or
    peano4::datamanagement::VertexHeapDataEnumerator under the same name.
    These enumerators hold copies of the faces or vertices. The copies are
    shallow if the data is managed via smart pointers.

    @ref peano4.api.taskgraph.compiler.CellTaskProducerConsumer warns about
    data which is not managed via smart pointers.

    All other events remain within the producer. They are hence
    invoked before the spawned tasks have necessarily terminated.

    """

    def __init__(
        self,
        action_set: ActionSet,
        task_labels_name: str,
        first_task_attribute_name: str,
        last_task_attribute_name: str,
        children_tasks_attribute_name: str,
        previous_first_task_attribute_name: str,
        previous_last_task_attribute_name: str,
        face_enumerators: list = [],
        vertex_enumerators: list = [],
    ):
        """!

        @param action_set: peano4.api.solversteps.ActionSet
          The wrapped action set.

        @param task_labels_name: str
          Name of the DaStGen2 cell data type holding the task labels. The
          generated observer passes it into the event as fineGridCell<name>
          and coarseGridCell<name>.

        @param first_task_attribute_name: str
          Name of the attribute within the task labels which holds the number
          of the task spawned by touchCellFirstTime().

        @param last_task_attribute_name: str
          Name of the attribute which holds the number of the task spawned by
          touchCellLastTime().

        @param children_tasks_attribute_name: str
          Name of the integer array attribute of size 3^d. Each child holds
          its last task number in the entry which belongs to its position
          within its parent.

        @param previous_first_task_attribute_name: str or None
          Name of the attribute holding the first task number of the previous
          action set of the same producer. The first task depends on this
          task. If None, there is no such dependency.

        @param previous_last_task_attribute_name: str or None
          Counterpart of previous_first_task_attribute_name for the last tasks.

        @param face_enumerators: [str]
          Names of the face enumerators, e.g. fineGridFacesQ, which the task
          captures as peano4::datamanagement::FaceHeapDataEnumerator.

        @param vertex_enumerators: [str]
          Names of the vertex enumerators, e.g. fineGridVerticesQ, which the
          task captures as peano4::datamanagement::VertexHeapDataEnumerator.

        """
        super().__init__(descend_invocation_order=action_set.descend_invocation_order, parallel=action_set.parallel)
        self.action_set = action_set
        self._task_labels_name = task_labels_name
        self._first_task_attribute_name = first_task_attribute_name
        self._last_task_attribute_name = last_task_attribute_name
        self._children_tasks_attribute_name = children_tasks_attribute_name
        self._previous_first_task_attribute_name = previous_first_task_attribute_name
        self._previous_last_task_attribute_name = previous_last_task_attribute_name
        self._face_enumerators = list(face_enumerators)
        self._vertex_enumerators = list(vertex_enumerators)

    def has_body_of(self, operation_name) -> bool:
        return self.action_set.get_body_of_operation(operation_name).strip() != ""

    def _lambda_captures(self) -> str:
        """!

        Capture list of the spawned task's lambda: Everything by value, and
        the vertex and face enumerators as copies of their data under the
        same name.

        """
        captures = ["="]
        captures += [
            f"{name} = peano4::datamanagement::FaceHeapDataEnumerator({name})" for name in self._face_enumerators
        ]
        captures += [
            f"{name} = peano4::datamanagement::VertexHeapDataEnumerator({name})" for name in self._vertex_enumerators
        ]
        return ", ".join(captures)

    def _accessor(self, attribute_name: str) -> str:
        return attribute_name[0].capitalize() + attribute_name[1:]

    def get_constructor_body(self):
        return self.action_set.get_constructor_body()

    def get_destructor_body(self):
        return self.action_set.get_destructor_body()

    def get_static_initialisations(self, full_qualified_classname):
        return self.action_set.get_static_initialisations(full_qualified_classname)

    def get_body_of_createGridControlEventsForNextTraversal(self):
        return self.action_set.get_implementation_of_createGridControlEventsForNextTraversal()

    def get_body_of_prepareTraversal(self):
        return self.action_set.get_implementation_of_prepareTraversal()

    def get_body_of_unprepareTraversal(self):
        return self.action_set.get_implementation_of_unprepareTraversal()

    def get_body_of_operation(self, operation_name):
        # Not used, as get_implementation_of_operation() is overwritten
        return ""

    def _spawn(self, operation_name: str, number: str, in_dependencies: str) -> str:
        return f"""
    tarch::multicore::spawnTask(
      new tarch::multicore::TaskWithCopyOfFunctor(
        tarch::multicore::Task::DontFuse,
        tarch::multicore::Task::DefaultPriority,
        [{self._lambda_captures()}]() mutable {{
{self.action_set.get_body_of_operation(operation_name)}
        }}
      ),
      {in_dependencies},
      {number}
    );
"""

    def _touch_cell_first_time(self) -> str:
        fine = f"fineGridCell{self._task_labels_name}"
        coarse = f"coarseGridCell{self._task_labels_name}"
        first = self._accessor(self._first_task_attribute_name)
        last = self._accessor(self._last_task_attribute_name)
        children = self._accessor(self._children_tasks_attribute_name)
        result = f"""
  for (int i=0; i<ThreePowerD; i++) {{
    {fine}.set{children}( i, tarch::multicore::TaskEnumerator::NoNumber );
  }}
"""
        if self.has_body_of(ActionSet.OPERATION_TOUCH_CELL_LAST_TIME):
            result += f"""
  {fine}.set{last}( {fine}.getTaskEnumerator().getNumber() );
  {{
    const int childIndex = peano4::utils::dLinearised( marker.getRelativePositionWithinFatherCell(), 3 );
    if (childIndex>=0 and childIndex<ThreePowerD) {{
      {coarse}.set{children}( childIndex, {fine}.get{last}() );
    }}
  }}
"""
        else:
            result += f"  {fine}.set{last}( tarch::multicore::TaskEnumerator::NoNumber );\n"
        if self.has_body_of(ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME):
            dependencies = "  std::set<tarch::multicore::TaskNumber> inDependencies;\n"
            if self._previous_first_task_attribute_name is not None:
                previous = self._accessor(self._previous_first_task_attribute_name)
                dependencies += f"    if ( {fine}.get{previous}()>=0 ) inDependencies.insert( {fine}.get{previous}() );\n"
            dependencies += f"    if ( {coarse}.get{first}()>=0 ) inDependencies.insert( {coarse}.get{first}() );\n"
            result += f"""
  if ({self.action_set.get_guard_of_operation(ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME)}) {{
    const int taskNumber = {fine}.getTaskEnumerator().getNumber();
    {fine}.set{first}( taskNumber );
{dependencies}
{self._spawn(ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME, "taskNumber", "inDependencies")}
  }}
  else {{
    {fine}.set{first}( tarch::multicore::TaskEnumerator::NoNumber );
  }}
"""
        else:
            result += f"  {fine}.set{first}( tarch::multicore::TaskEnumerator::NoNumber );\n"
        return result

    def _touch_cell_last_time(self) -> str:
        fine = f"fineGridCell{self._task_labels_name}"
        first = self._accessor(self._first_task_attribute_name)
        last = self._accessor(self._last_task_attribute_name)
        children = self._accessor(self._children_tasks_attribute_name)
        dependencies = f"""
  std::set<tarch::multicore::TaskNumber> inDependencies;
  if ( {fine}.get{first}()>=0 ) inDependencies.insert( {fine}.get{first}() );
  for (int i=0; i<ThreePowerD; i++) {{
    if ( {fine}.get{children}(i)>=0 ) inDependencies.insert( {fine}.get{children}(i) );
  }}
"""
        if self._previous_last_task_attribute_name is not None:
            previous = self._accessor(self._previous_last_task_attribute_name)
            dependencies += f"  if ( {fine}.get{previous}()>=0 ) inDependencies.insert( {fine}.get{previous}() );\n"
        return f"""{dependencies}
  if ({self.action_set.get_guard_of_operation(ActionSet.OPERATION_TOUCH_CELL_LAST_TIME)}) {{
{self._spawn(ActionSet.OPERATION_TOUCH_CELL_LAST_TIME, f"{fine}.get{last}()", "inDependencies")}
  }}
  else {{
    tarch::multicore::spawnTask(
      new tarch::multicore::TaskWithCopyOfFunctor(
        tarch::multicore::Task::DontFuse,
        tarch::multicore::Task::DefaultPriority,
        []() mutable {{}}
      ),
      inDependencies,
      {fine}.get{last}()
    );
  }}
"""

    def get_implementation_of_operation(self, operation_name):
        """!

        Delegate to the wrapped action set besides the events listed in the
        class description

        """
        fine = f"fineGridCell{self._task_labels_name}"
        if operation_name == ActionSet.OPERATION_CREATE_CELL:
            return (
                self.action_set.get_implementation_of_operation(operation_name)
                + f"\n  {fine}.set{self._accessor(self._first_task_attribute_name)}( tarch::multicore::TaskEnumerator::NoNumber );\n"
                + f"  {fine}.set{self._accessor(self._last_task_attribute_name)}( tarch::multicore::TaskEnumerator::NoNumber );\n"
            )
        elif operation_name == ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME:
            return self._touch_cell_first_time()
        elif operation_name == ActionSet.OPERATION_TOUCH_CELL_LAST_TIME and self.has_body_of(operation_name):
            return self._touch_cell_last_time()
        else:
            return self.action_set.get_implementation_of_operation(operation_name)

    def get_action_set_name(self):
        return self.action_set.get_action_set_name()

    def user_should_modify_template(self):
        return False

    def get_attributes(self):
        return self.action_set.get_attributes()

    def get_includes(self):
        return (
            self.action_set.get_includes()
            + """
#include <set>
#include "peano4/utils/Loop.h"
#include "tarch/multicore/multicore.h"
#include "tarch/multicore/Task.h"
#include "tarch/multicore/TaskEnumerator.h"
#include "peano4/datamanagement/FaceHeapDataEnumerator.h"
#include "peano4/datamanagement/VertexHeapDataEnumerator.h"
"""
        )
