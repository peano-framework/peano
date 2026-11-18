# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import peano4.api.output.ActionSet
import os

class StepToObserver(object):
  """!
  
  Take an action set and construct an observer from it.

  An action set is, as the name suggests, a set of actions. Each action
  set will have to be mapped onto one C++ action set, which is a where the 
  action behaviour is implemented: a selection of functions which react to 
  events such as "I have hit a face for the first time".
  
  An observer administers these action sets:
  
  - It holds a series of action sets dispatches the events to them.
  - It manages the data movements induced by the grid traversal automaton
    running through the mesh.
    
  Consequently, each action set among the solver steps yields exactly one
  observer, but potentially many action sets in the C++ implementation.
    
    
  ## Important links

  - @ref page_peano_action_sets    
  - @ref peano4.api.taskgraph


  """
  
  
  
  def __init__(self,step):
    self.step = step
    
    
  def construct_output(self,output,included_action_sets,task_groups=None):
    """!
      Construct one mapping

      @param task_groups: [[int]] or None
        Groups of indices into included_action_sets, i.e. action sets
        which run in parallel - @ref peano4.api.output.Observer.Observer.
    """
    observer = peano4.api.output.Observer(
      self.step.get_observer_class_name(), self.step.project.namespace + [ "observers" ], os.path.join(self.step.project.temporary_subdirectory, "observers"),
      included_action_sets,
      self.step.vertex_data,
      self.step.face_data,
      self.step.cell_data,
      task_groups
    )
    output.artefacts.append( observer )
    # This is the temporary location
    temporary_observer_path = observer.get_cpp_file_name()
    true_observer_path = os.path.join(self.step.project.subdirectory, "observers", temporary_observer_path.split('/')[-1])
    output.makefile.add_cpp_file( true_observer_path, generated=True )
