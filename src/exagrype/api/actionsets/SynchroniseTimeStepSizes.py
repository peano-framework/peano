""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

from peano4.api.solversteps.ActionSet import ActionSet


class SynchroniseTimeStepSizes(ActionSet):
    """!

    Make several solvers advance with the same time step size

    The action set acts in unprepareTraversal() only. After the solvers have
    computed their next time step size, it takes the minimum of them and
    enforces it on all solvers (::exahype2::Solver::enforceTimeStepSize()).
    The task that hosts the action set is a GLOBAL_END_OF_STEP task, which
    depends on the last tasks of all solvers.

    """

    def __init__(self, solvers):
        """!

        @param solvers: List[exahype2.api.Solver]
          The solvers whose time step sizes are synchronised.

        """
        super().__init__(descend_invocation_order=1, parallel=False)
        self._solvers = list(solvers)

    def get_action_set_name(self):
        return __name__.replace(".py", "").replace(".", "_")

    def user_should_modify_template(self):
        return False

    def get_includes(self):
        return '#include "repositories/SolverRepository.h"\n#include <algorithm>\n#include <limits>\n'

    def get_body_of_unprepareTraversal(self):
        instances = [
            f"repositories::{solver.get_name_of_global_instance()}"
            for solver in self._solvers
        ]
        minimum = "std::numeric_limits<double>::max()"
        for instance in instances:
            minimum = f"std::min( {instance}.getMaxTimeStepSize(), {minimum} )"
        enforce = "\n".join(
            f"  {instance}.enforceTimeStepSize(timeStepSize);" for instance in instances
        )
        return f"""
  const double timeStepSize = {minimum};
{enforce}
"""
