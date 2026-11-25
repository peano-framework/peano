# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from typing import Dict, List

from peano4.api.solversteps.ActionSet import ActionSet

from .SpawnTaskActionSet import ENTITIES, SPECS, SpawnTaskActionSet, accessor


class WaitForTasksActionSet(ActionSet):
    """!

    Consumer-side action set which waits for all tasks that the wrappers of one task have spawned

    Each entity holds, per wrapper, the numbers of the tasks of its first and
    its last event. The consumer visits each persistent entity once and waits
    for the two tasks. It returns the numbers to the task enumerator and
    resets the labels to tarch::multicore::TaskEnumerator::NoNumber. As the
    last task of a cell depends on the cell's whole subtree, and the last
    task of a vertex and a face on all cells around it, the consumer finds
    the whole DAG finished once the traversal has terminated.

    Hanging entities are not visited by the consumer. The producer waits
    for their tasks (@ref peano4.api.taskgraph.compiler.SpawnTaskActionSet.SpawnTaskActionSet).

    """

    def __init__(self, name: str, wrappers: List[SpawnTaskActionSet]):
        super().__init__(descend_invocation_order=0, parallel=False)
        self._name = name
        self._wrappers = list(wrappers)

    def kinds(self) -> List[str]:
        """! Entities for which at least one wrapper spawns a task"""
        return [
            kind
            for kind in ENTITIES
            if any(
                kind in wrapper.attributes
                and (
                    wrapper.spawns(kind, first=True)
                    or wrapper.spawns(kind, first=False)
                )
                for wrapper in self._wrappers
            )
        ]

    def get_constructor_body(self):
        return ""

    def get_destructor_body(self):
        return ""

    def get_body_of_createGridControlEventsForNextTraversal(self):
        return "  return std::vector< peano4::grid::GridControlEvent >();\n"

    def get_body_of_prepareTraversal(self):
        return "\n"

    def get_body_of_unprepareTraversal(self):
        return "\n"

    def get_body_of_operation(self, operation_name):
        # Not used, as get_implementation_of_operation() is overwritten
        return ""

    def get_implementation_of_operation(self, operation_name):
        for kind in self.kinds():
            if operation_name != SPECS[kind].first_event:
                continue
            result = "\n"
            for wrapper in self._wrappers:
                if kind not in wrapper.attributes:
                    continue
                me = wrapper._me(kind)
                for attribute in (
                    wrapper.attributes[kind].last,
                    wrapper.attributes[kind].first,
                ):
                    result += f"""  {{
    int number = {me}.get{accessor(attribute)}();
    peano4::datamanagement::taskassembly::waitForAndReleaseTask( peano4::datamanagement::taskassembly::getTaskEnumerator(), number );
    {me}.set{accessor(attribute)}( number );
  }}
"""
            return result
        return ""

    def get_action_set_name(self):
        return self._name

    def user_should_modify_template(self):
        return False

    def get_attributes(self):
        return ""

    def get_includes(self):
        return """
#include "tarch/multicore/multicore.h"
#include "tarch/multicore/TaskEnumerator.h"
#include "peano4/datamanagement/TaskAssemblyDependencies.h"
"""
