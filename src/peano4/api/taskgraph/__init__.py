# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

The task graph abstraction allows users to formalise their solver steps over small task graphs which are then lowered into a sequence of Peano mesh traversals. 


@tableofcontents


## Example

An ExaHyPE solver's generator, i.e. the part that translates the pyysical model into a numerical workflow, describes each algorithmic phase (setup, time stepping, plotting)
as a @ref peano4.api.taskgraph.Graph of @ref peano4.api.taskgraph.Task instances.
Each task defines computations over a mesh entity (vertex, face, cell, or global), subject to dependencies between the tasks.
Global tasks do not touch the mesh. They are tied to a traversal, to the start of a time step or to its end
(@ref peano4.api.taskgraph.Task.Task.GridEntity). The latter two never cost a mesh traversal of their own.
Typical examples for a task at the end of the step are the global CFL condition or a synchronisation of several solvers.
A simple task graph reads as follows:

1. Run over all cells and compute A.
2. Run over all faces and compute B, meanwhile do a global calculation. Both steps require A to have finished.
3. Run over all vertices once B has completed.
4. Wait for the third step and the global calculation and run over all cells again.

The solvers use this formalism to set up graphs similar to this one:

@image html task-graph-example.png

This is obviously strongly simplified. 
A proper yet still simplistic example can be found in @ref exahype2.api.fv.SetupSimulationOverRegularGrid.
The core interesting stuff looks similar to

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
refine_task                 = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Refine mesh",
                                                                entity=peano4.api.taskgraph.Task.GridEntity.CELL,
                                                                task_has_to_repeat=...
                                                                )
consolidate_load_balancing_task       = peano4.api.taskgraph.Task(name=f"{self._solver.name}_Consolidate mesh",
                                                                entity=peano4.api.taskgraph.Task.GridEntity.CELL,
                                                                task_has_to_repeat=...
                                                                )
 
refine_task.add_action_set( ... )
consolidate_load_balancing_task.add_action_set(toolbox.loadbalancing.api.actionsets.LoadBalancing())
 
initial_condition_task        = peano4.api.taskgraph.Task(name=f"{self._solver.name} Initial condition", entity=peano4.api.taskgraph.Task.GridEntity.CELL)
project_on_faces_task         = peano4.api.taskgraph.Task(name=f"{self._solver.name} Project on faces", entity=peano4.api.taskgraph.Task.GridEntity.CELL)
 
consolidate_load_balancing_task.add_dependency(refine_task)
initial_condition_task.add_dependency(consolidate_load_balancing_task)
project_on_faces_task.add_dependency(initial_condition_task)
 
initial_condition_task.add_action_set(
            InitialCondition(self._solver, guard=self._solver.grid_selector.createCellGuard())
)
project_on_faces_task.add_action_set(
            ProjectPatchOntoFaces(
                self._solver._patch,
                self._solver._patch_overlap_update,
                guard=self._solver.grid_selector.createCellGuard(),
                includes=self._solver._get_default_includes() + self._solver.user_solver_includes,
            )
)
project_on_faces_task.add_action_set(
            RollOverUpdatedFace(
                self._solver._patch_overlap_current_time_step,
                self._solver._patch_overlap_update,
                guard=self._solver.grid_selector.createFaceGuard(),
                includes=self._solver._get_default_includes() + self._solver.user_solver_includes,
            )
        )
 
setup_simulation.add_task(refine_task)
setup_simulation.add_task(consolidate_load_balancing_task)
setup_simulation.add_task(initial_condition_task)
setup_simulation.add_task(project_on_faces_task)
 
initial_condition_task.use_cell(self._solver._patch, peano4.api.taskgraph.Task.DataAccess.WRITE)
project_on_faces_task.use_cell(self._solver._patch, peano4.api.taskgraph.Task.DataAccess.READ_AND_WRITE)
project_on_faces_task.use_face(self._solver._patch_overlap_update, peano4.api.taskgraph.Task.DataAccess.TEMPORARY)
project_on_faces_task.use_face(self._solver._patch_overlap_current_time_step, peano4.api.taskgraph.Task.DataAccess.WRITE)

setup_simulation.visualise(title="Setup task graph for {}".format(self._solver.name))
setup_simulation.log_as_text(title="Setup task graph for {}".format(self._solver.name))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


The typical steps are always the same:

1. Create the tasks.
2. Give them a name, specify if they can repeat (off by default) and specify over which mesh entity they operate.
3. Add action sets to the tasks. Action sets are the actual code that is triggered by a task (and executed during a mesh traversal corresponding to this task later).
4. Add dependencies between the tasks.
5. Add task to task graph.


Note the nice feature at the end of the script: The API comes along
with a routine to visualise the arising task graph. This is the same
feature used for the example image above.

## Mapping the task graph onto Peano

The idea behind the task graph feature of Peano is that there are different
compilers which take the task graph and lower it onto grid traversals.
The term "lower" is taken from the xDSL compiler project and highlights that 
we go from more abstract definitions close to the actual implementation.

There are different compilers available, each producing a different type of 
a task graph implementation. You find them all in @ref peano4.api.taskgraph.compiler.
The simples one if @ref @ref peano4.api.taskgraph.compiler.Sequential, which 
maps the task graph onto a total order preservering the dependencies. Each task
of the rearranged total order (each step) is then mapped onto one mesh traversal.
This is the simplest way to execute a task graph, through obviously might not
be the most efficient one.

Besides the lowering onto Peano mesh traversals, the compilers also yield 
brief functions which we can call from the main() function later to step 
through all the mesh traversals which make up one task graph realisation.


## Graph properties

<div style="background-color: #f0c ; padding: 10px; border: 1px solid green;"> 
The task graph implicitly models dependencies over touch-first events. This is, 
if a task A is formalised over faces and B depends on cells, then the task graph
compiler might later conclude that A and B can be done in one mesh sweep, as 
all touchFaceFirstTime() complete before any touchCellFirstTime() is called on 
adjacent faces. These dependencies are not validated. Task A might as well host
an action set which does something in touchFaceLastTime() which feeds into B.
It is the user's responsibility to ensure that the tasks are formalised correctly.
</div>


### Task dependencies

Each task can depend on an arbitrary number of predecessor tasks. There is 
no feature to model outgoing dependencies. They are implicitly given by other
tasks referencing predecessors.

The task graph does not have to have a unique root task. It can comprise
multiple ready tasks and can have multiple terminal tasks.

### Tasks and action sets

Each task can hold an arbitrary number of Peano 4 action sets (@ref peano4.api.solversteps.ActionSet).
Each action set can plug into all kinds of events. This seems to be a contradiction:
The task is tied to a particular mesh entity, while the action set can define actions for multiple mesh entities. 
The task association defines dependencies, the action set what is done.
If task A depends on B and B is tied to vertices, any operation over a cell in 
A will be executed after all operations over the cell's adjacent vertices in B 
have completed.
The action sets of A still might also do something on cells and faces as well, 
but this is not subject to the dependency handling.

### Task repetition

By default, tasks in the task graph do not repeat. This means that they are 
executed once. But users can attach a boolean expression to a task which is 
evaluated at the end of the task's execution. If it evaluates to true, then the task is repeated. This is useful for tasks that need to be executed multiple times until a certain condition is met.
Unfortunately, a task that can repeat can never be merged with any successor,
so these repeating tasks kind of veto any optimisation.

### Task tags

Each task can carry arbitrarily many tags. They are plain strings, stored
in the set tags of @ref peano4.api.taskgraph.Task.Task, so one task can be
tagged by several independent parties at the same time. Tags are global:
Nobody constructs a tag object.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
task = peano4.api.taskgraph.Task(name="Update cell",
                                 entity=peano4.api.taskgraph.Task.GridEntity.CELL,
                                 tags={"MYTAG"})
task.tags.add(exahype2.api.Solver.SolutionUpdateTaskTag)
task.tags.add("MYOTHERTAG")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tags allow you to pick out tasks from a graph which someone else has
built: If you know that there is a task tagged with MYTAG, then
@ref peano4.api.taskgraph.Graph.find_tasks() returns all tasks which carry
this tag, and you can add new tasks behind them through
@ref peano4.api.taskgraph.Graph.insert_after().

The tags carry no intrinsic semantics from the task graph's point of view,
i.e. the task graph compilers ignore them. It is up to the user to define
their meaning.



@see @ref peano4.api.taskgraph.Task
@see @ref peano4.api.taskgraph.Graph
@see @ref peano4.api.taskgraph.compiler
@see @ref peano4.api.taskgraph.create_runner

"""

from .Task import Task
from .Graph import Graph, wait_for_visualisations
from .Runner import create_runner
from . import compiler
