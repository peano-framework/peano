# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

Compilers are simple functions 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def translate(graph: Graph, identifier: str, peano_project: "peano4.api.Project") -> List["peano4.api.solversteps.Step"]:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

which take a task graph and yield a series of @ref peano4.api.solversteps.Step instances.
The peano_project is the Peano project the steps eventually are added to.
Compilers can use it to add further data to the data model (@ref
peano4.api.taskgraph.compiler.CellTaskProducerConsumer). It might be None
if a compiler is used outside of a project, e.g. in unit tests.

In the Peano terminology, a @see peano4.api.solversteps.Step corresponds to an Observer: for every
mesh traversal, the traversal automaton is handed an observer that
distributes the grid events among the action sets tied to it.

There is one notable exception compared to the plain steps in 
@ref peano4.api.solversteps: Each step is also added an attribute
has_to_repeat which is a copy from Task's has_to_repeat. 
Consult the description of @ref peano4.api.taskgraph.Task for details on how to use this attribute to make a task repeat until a certain condition is met.

"""



from . import Sequential
from . import FuseSteps
from . import CellTaskProducerConsumer
from . import ValidateTaskGraph
