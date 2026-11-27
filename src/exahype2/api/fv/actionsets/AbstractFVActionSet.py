# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from peano4.api.solversteps.ActionSet import ActionSet


class AbstractFVActionSet(ActionSet):
    def __init__(self, solver):
        """

        solver: ADERDG
          Reference to creating class

        """
        super(AbstractFVActionSet, self).__init__(descend_invocation_order=1, parallel=False)
        self._solver = solver
        pass

    def get_action_set_name(self):
        """

         You should replicate this function in each subclass, so you get
         meaningful action set names (otherwise, it will be
         AbstractFVActionSet0,1,2,...).

        """
        return __name__.replace(".py", "").replace(".", "_")

    def user_should_modify_template(self):
        return False

    def get_includes(self):
        return f"""
#include <functional>
#include <vector>
#include "exahype2/CellData.h"
#include "exahype2/fv/PatchUtils.h"
#include "kernels/{self._solver.name}_rusanov.h"
#include "kernels/{self._solver.name}_initialConditions.h"
""" + self._solver._get_default_includes() + self._solver.user_solver_includes
