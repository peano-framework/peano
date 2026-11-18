# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from enum import IntEnum


class DoFAssociation(IntEnum):
  """
  
   Superclass has to be IntEnum, as I use this one within Jinja2 templates
   where I struggled to compare against enum variants. I however can always
   compare against integers.
  
  """
  Undef  = 0
  Vertex = 1
  Face   = 2
  Cell   = 3
  #
  # I use generic for standard MPI messages or DaStGen messages which are used
  # within the grid mangement and traversal, i.e. have nothing to do with particular
  # grid entities
  #
  Generic = 4
  Global  = 5
  
  
def get_logical_type_name_for_association(association, name):
  result = ""
  if association==DoFAssociation.Vertex:
    result += "VertexData"
  elif association==DoFAssociation.Cell:
    result += "CellData"
  elif association==DoFAssociation.Face:
    result += "FaceData"
  elif association==DoFAssociation.Global:
    result += "GlobalData"
  else:
    assert False
  result += name
  return result


def get_subnamespace_for_association(association):
  if association==DoFAssociation.Vertex:
    return "vertexdata"
  elif association==DoFAssociation.Cell:
    return "celldata" 
  elif association==DoFAssociation.Face:
    return "facedata" 
  elif association==DoFAssociation.Global:
    return "globaldata"
  else:
    assert False

  
class DoF(object):
  """!
  
  Base class for any degree of freedom object
  
  Any degree of freedom object in Peano is derived from DoF. There are 
  different subclasses of this type which are actually used in most 
  implementations. The most popular one is peano4.api.datamodel.DaStGen2 which
  encodes a DoF implementation which is based upon plain C++ objects created
  through DaStGen2. However, there are also other data models such as bespoke
  patch data represented through peano4.api.datamodel.Patch.
  
  @image html dof-class-diagram.png

  Every DoF in Peano has a name and a unique association with a mesh entity
  which is held in association. There are some further properties tied to 
  each DoF such as a position x and a mesh width h, but we leave it to 
  subclasses to provide those guys.
  
  Please consult the tutorials on @ref tutorials_peano4_assign_data_to_mesh "assigning data to a mesh" and
  @ref tutorials_peano4_mask_out_data_usage "masking out data storage"
  on some instructions where this class is actually used in code.

  """
  
  
  def __init__(self, name):
    """!
    
    Create a degree of freedom object
    
    Both the association and the namespace are not to be set directly,
    but through the operation configure().

    ## Additional store/load/send/receive arguments
    
    - The first entry of this triple is the expression that you want the 
      observer to pass in.
    - The second one is the C++ type of this expression.
    - The third one is the name that you wanna use in the dof signatures
      for this argument.
      
      It is the responsibility of the DoF subclass to ensure that the second 
      entry is properly used to initialise the store and load routines.
      Most define an additional setter for property.
    
        
    ## Attributes
    
    association: DoFAssociation
    
    name:        String
      Has to be a fit to the C++ naming conventions
      
    namespace:   [String]
      Sequence of namespaces.
      
    additional_load_and_store_arguments: [(String,String,String)]
      This flag is, by default, an empty list. If you add an entry to this 
      list, each store and load routine will get an additional parameter. 
      Consult documentation above for the semantics of the list entries.

    grid_selector: peano4.api.toolbox.GridSelector or None
      None by default. If set, the task graph compiler restricts the
      storage of this DoF to the mesh entities selected: The selector's
      guard is AND-combined with each of the use/load/store predicates the
      compiler derives (@see peano4.api.taskgraph.compiler.DeriveLoadStoreComputeFlag).
      That is, data are only used, loaded or stored if the selector holds
      and the compiler has concluded that the data are needed.


    """
    self.association = DoFAssociation.Undef
    self.name        = name
    self.namespace   = []
    self._additional_load_and_store_arguments = []
    self.grid_selector = None
  
  
  def configure(self,namespace,association, subdirectory="", temporary_subdirectory=""):
    """
    Typically called by model as soon as you add an object to it
    """
    self.namespace = [x for x in namespace]
    self.association  = association
    self.namespace   += [ get_subnamespace_for_association(association) ]   
    self.subdirectory = subdirectory
    self.temporary_subdirectory = temporary_subdirectory
    

  def get_full_qualified_type(self):
    result = ""
    for i in self.namespace:
      result += i
      result += "::"
    result += self.name
    return result
    

  def get_logical_type_name(self):
    """
      What should the data type be called within the data repository,
      or within action sets. We add a prefix name here.
    """
    return  get_logical_type_name_for_association(self.association, self.name)


  def get_enumeration_type(self):
    """
      What should the data type be called within the data repository.
    """
    result = ""
    if self.association==DoFAssociation.Vertex:
      result += "peano4::datamanagement::VertexEnumerator<"
    elif self.association==DoFAssociation.Face:
      result += "peano4::datamanagement::FaceEnumerator<"
    else:
      assert False, "association was {}".format( self.association )
    result += self.get_full_qualified_type()
    result += ">"
    return result


  @property
  def additional_load_and_store_arguments(self):
    return self._additional_load_and_store_arguments


  def additional_load_and_store_arguments_for_other_dof(
    self, 
    argument_name,
    use_dof_association = None
  ):
    """
    
    You can make Peano's store and load arguments of any DoF depend on other 
    DoFs that you have loaded before. So you can load X and then make Y later
    on depend on the state of X. For this, you have to add a tuple to 
    Y's additional_load_and_store_arguments. It is a very technical tuple.
    This routine helps you to construct it for X.
    
    use_dof_association: DoFAssociation
      Set this one to None and the routine will pick up self._association.
      However, there are cases where we want to construct the string and 
      the association is not yet set properly. In this case, you have to
      supplement the information manually.
      
    """
    if use_dof_association==None:
      use_dof_association = self.association
      
    stack_access_string = "repositories::DataRepository::_{}Stack.getForPush( repositories::DataRepository::DataKey(_spacetreeId,peano4::grid::PeanoCurve::CallStack))->top()".format(
      get_logical_type_name_for_association(use_dof_association, self.name)
      )

    data_type ="{}::{}".format(
      get_subnamespace_for_association(use_dof_association),
      self.name
    )
    
    return (stack_access_string,data_type,argument_name)

  @property
  def subnamespace(self):
    # Replace "/" with "::" in the directory string to make it a namespace
    return self.subdirectory.replace("/", "::")
