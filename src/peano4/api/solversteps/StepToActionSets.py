# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import peano4.api.output.ActionSet
import os

class StepToActionSets(object):
  """!
  
  Maps a step to an action set
  
  This is a generator, i.e. it is 
  """
  
  
  def __init__(self,step):
    self.step = step

  def construct_output(self,output,action_set):
      """
        Construct one action_set
      """
      number_of_user_action_sets = 0
      for x in self.step.action_sets:
        if x.user_should_modify_template():
          number_of_user_action_sets +=1

      action_set_name = ""
      subnamespace = ""
      if action_set.user_should_modify_template():
        subnamespace = "actions" 
        action_set_name = action_set.get_action_set_name()
        print( "user has to modify class " + action_set.get_action_set_name() + " in actions directory manually ")
      else:
        subnamespace = "observers" 
        action_set_name = self.step.get_observer_class_name() + "2" + action_set.get_action_set_name() + str( self.step.action_sets.index(action_set))

      new_action_set = peano4.api.output.ActionSet(
        action_set_name, self.step.project.namespace + [ subnamespace ], os.path.join(self.step.project.temporary_subdirectory, subnamespace), action_set
      )
 
      for i in self.step.vertex_data:
        new_action_set.include_files.append( i.subdirectory + "vertexdata/" + i.name + ".h" )    
      for i in self.step.face_data:
        new_action_set.include_files.append( i.subdirectory + "facedata/" + i.name + ".h" )    
      for i in self.step.cell_data:
        new_action_set.include_files.append( i.subdirectory + "celldata/" + i.name + ".h" )    

      if len(self.step.vertex_data)>0:
        new_action_set.operations.append( [ peano4.api.solversteps.ActionSet.OPERATION_CREATE_PERSISTENT_VERTEX,  "void" ] + self.step.get_vertex_operations_signature() )
        new_action_set.operations.append( [ peano4.api.solversteps.ActionSet.OPERATION_DESTROY_PERSISTENT_VERTEX, "void" ] + self.step.get_vertex_operations_signature() )
        new_action_set.operations.append( [ peano4.api.solversteps.ActionSet.OPERATION_CREATE_HANGING_VERTEX,     "void" ] + self.step.get_vertex_operations_signature() )
        new_action_set.operations.append( [ peano4.api.solversteps.ActionSet.OPERATION_DESTROY_HANGING_VERTEX,    "void" ] + self.step.get_vertex_operations_signature() )
        new_action_set.operations.append( [ peano4.api.solversteps.ActionSet.OPERATION_TOUCH_VERTEX_FIRST_TIME,   "void" ] + self.step.get_vertex_operations_signature() )
        new_action_set.operations.append( [ peano4.api.solversteps.ActionSet.OPERATION_TOUCH_VERTEX_LAST_TIME,    "void" ] + self.step.get_vertex_operations_signature() )

      if len(self.step.face_data)>0:
        new_action_set.operations.append( [ peano4.api.solversteps.ActionSet.OPERATION_CREATE_PERSISTENT_FACE,  "void" ] + self.step.get_face_operations_signature() )
        new_action_set.operations.append( [ peano4.api.solversteps.ActionSet.OPERATION_DESTROY_PERSISTENT_FACE, "void" ] + self.step.get_face_operations_signature() )
        new_action_set.operations.append( [ peano4.api.solversteps.ActionSet.OPERATION_CREATE_HANGING_FACE,     "void" ] + self.step.get_face_operations_signature() )
        new_action_set.operations.append( [ peano4.api.solversteps.ActionSet.OPERATION_DESTROY_HANGING_FACE,    "void" ] + self.step.get_face_operations_signature() )
        new_action_set.operations.append( [ peano4.api.solversteps.ActionSet.OPERATION_TOUCH_FACE_FIRST_TIME,   "void" ] + self.step.get_face_operations_signature() )
        new_action_set.operations.append( [ peano4.api.solversteps.ActionSet.OPERATION_TOUCH_FACE_LAST_TIME,    "void" ] + self.step.get_face_operations_signature() )

      if len(self.step.cell_data)>0:
        new_action_set.operations.append( [ peano4.api.solversteps.ActionSet.OPERATION_CREATE_CELL,           "void" ] + self.step.get_cell_operations_signature() )
        new_action_set.operations.append( [ peano4.api.solversteps.ActionSet.OPERATION_DESTROY_CELL,          "void" ] + self.step.get_cell_operations_signature() )

      new_action_set.operations.append( 
        [ peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_FIRST_TIME, "void" ] + 
        self.step.get_touch_cell_signature()
      )
      new_action_set.operations.append( 
        [ peano4.api.solversteps.ActionSet.OPERATION_TOUCH_CELL_LAST_TIME, "void" ] + 
        self.step.get_touch_cell_signature()
      )

      output.artefacts.append( new_action_set )
      # This is the temporary location
      temporary_action_set_path = new_action_set.get_cpp_file_name()
      true_action_set_path = os.path.join(self.step.project.subdirectory, subnamespace, temporary_action_set_path.split('/')[-1])
      output.makefile.add_cpp_file( true_action_set_path, generated=True )

      return subnamespace + "::" + new_action_set.classname
