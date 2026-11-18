# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import os
import re

from .Helper import write_file, shortened_class_name

from .Overwrite import Overwrite

import peano4.api.solversteps.ActionSet

class ActionSet(object):
  """!

  Code generator for one action set

  Do not confuse this with @ref peano4.api.solversteps.ActionSet.ActionSet,
  which is the interface you actually implement (you subclass it and override
  get_guard_of_operation(), get_body_of_operation(), get_attributes(),
  get_includes(), ... to say
  *what* the action set does). Users typically never interact with this 
  class. They work against the ActionSet in solversteps.
  
  This class here is the corresponding writer: given one
  such solversteps.ActionSet instance (held as self.implementation), it
  emits the actual C++ .h/.cpp pair - the include guard, the class
  declaration inside the right namespace, a static tarch::logging::Log
  _log, the constructor/destructor, createGridControlEventsForNextTraversal(),
  prepareTraversal()/unprepareTraversal(), and one method per grid event
  (touchCellFirstTime, createPersistentVertex, ...) - calling back into
  the get_implementation_of_*() routines - which combine your
  get_guard_of_operation() and get_body_of_*() overrides - to fill in each
  method body.
  @ref peano4.api.solversteps.ActionSet.ActionSet is the data model, this class is the writer.

  # Where is is used

  The class is instantiated in exactly one place: construct_output() of @ref
  peano4.api.solversteps.StepToActionSets.StepToActionSets, which is called
  once per action set attached to a @ref peano4.api.solversteps.Step.Step,
  from Step.construct_output(), as it loops over self.action_sets. That
  call site decides two things this class here itself does not know about:

  - Which operations to generate.
  - What the total number of vertices, cells, ... is and what they are called.
  - Where the file goes and what it is called - if
    action_set.user_should_modify_template() is True, it is named after the
    action set itself and placed under actions/ (meant for you to hand-edit,
    regenerated only if missing); if False, it is placed under observers/
    and named "{step.name}2{action_set.get_action_set_name()}{index}" -
    regenerated unconditionally every run.

  # In short
  
  - This is the generator which takes one @ref peano4.api.solversteps.StepToActionSets.StepToActionSets and maps it onto a header/implementation file pair.
  - The outcome either ends up in actions/ (if user_should_modify_template() is True) or in observers/ (if False).
  - The outcome typically is named Observer2ActionSetName.
  
  ## Important links

  - @ref page_peano_action_sets    
  - @ref peano4.api.taskgraph

  """

  def __init__(self,classname,namespace,subdirectory,implementation = None):
    """!

     implementation Should be of type peano4.api.solversteps.ActionSet or
                    None. If it is None, then the generated stuff will be a
                    sole interface.

    """
    # The class name is also the file name, which might become too long
    self.classname    = shortened_class_name(classname)
    self.namespace    = namespace
    self.subdirectory = subdirectory
    self.operations   = [
      (peano4.api.solversteps.ActionSet.OPERATION_BEGIN_TRAVERSAL,"void"),
      (peano4.api.solversteps.ActionSet.OPERATION_END_TRAVERSAL,"void")
    ]
    self.include_files  = []
    self.typedefs       = []
    self.implementation = implementation

  def add_operation(self,name,signature):
    """
     signature is a long tuple. The first entry is the name of the routine,
     the second entry is the result type. From thereon, one entry gives the
     name of an attribute, the second one the type.
    """
    self.operations.append(signature)

  def __generate_includes(self, outputfile):
    for i in self.include_files: outputfile.write( '#include "{}"\n'.format(i) )

  def __get_operation_arguments(self,operation):
    result = "(\n      "
    i = 2
    while i<len(operation):
      result += operation[i+1]
      result += " "
      result += operation[i]
      i+=2
      if i<len(operation):
        result += ",\n      "

    result += ")"
    return result

  def __generate_operation(self,outputfile,operation):
    """
      outputfile points to file, operation to an operation object as added by the solver step
    """
    if self.implementation == None:
      outputfile.write( "    virtual " )
    else:
      outputfile.write( "    " )
      outputfile.write( operation[1] )
      outputfile.write( " ")
      outputfile.write( operation[0] )
      outputfile.write( self.__get_operation_arguments(operation) )
      if self.implementation == None:
        outputfile.write( " = 0")
      outputfile.write( ";\n\n")

  def __get_full_qualified_class_name(self):
    full_qualified_classname = ""
    for i in self.namespace:
      full_qualified_classname += i
      full_qualified_classname += "::"
    full_qualified_classname += self.classname
    return full_qualified_classname

  def __generate_header(self,overwrite,directory):
    import inspect, os
    filename = os.path.join(directory, self.subdirectory, self.classname + ".h")
    default_overwrite = True
    if self.implementation.user_should_modify_template():
      default_overwrite = False

    if write_file(overwrite,default_overwrite,filename):
      print( "{} written by {}".format(filename, os.path.basename(inspect.getfile(self.__class__))))

      with open( filename, "w" ) as outputfile:
          full_qualified_classname = self.__get_full_qualified_class_name()
          include_guard = "_{}_H_".format(full_qualified_classname.replace( "::", "_" ).upper())

          outputfile.write( "#ifndef {}\n#define {}\n\n\n".format(include_guard, include_guard))

          outputfile.write( """
#include "peano4/utils/Globals.h"
#include "peano4/datamanagement/VertexEnumerator.h"
#include "peano4/datamanagement/VertexMarker.h"
#include "peano4/datamanagement/FaceEnumerator.h"
#include "peano4/datamanagement/FaceMarker.h"
#include "peano4/datamanagement/CellMarker.h"
#include "peano4/grid/GridControlEvent.h"
#include "tarch/la/Vector.h"

#include <vector>


""" )

          outputfile.write( self.implementation.get_includes() )
          outputfile.write( "\n\n" )

          self.__generate_includes(outputfile)

          for i in self.namespace: outputfile.write( "namespace " + i + " {\n" )
          outputfile.write("  class {};\n".format(self.classname))
          for i in self.namespace: outputfile.write( "}\n" )

          outputfile.write( "class " + full_qualified_classname + "{\n" )
          outputfile.write( "  private:\n" )
          outputfile.write( "    static tarch::logging::Log  _log;\n" )
          outputfile.write( self.implementation.get_attributes() )

          outputfile.write( "  public:\n" )
          outputfile.write( """
    /**
     * Create action instance for one tree for one grid sweep
     *
     * <h2> Thread safety </h2>
     *
     * The creation of individual trees usually happens through peano4::parallel::SpacetreeSet::createObserverCloneIfRequired().
     * This routine is called lazily when we start to traverse a subtree.
     * Therefore, the creation of actions is not thread-safe.
     *
     *
     * @param treeNumber Number of the spacetree for which we create the tree instance. Is
     *                   smaller 0 if this is the prototype action used on a rank from which
     *                   the real actions are constructed from.
     */
""" )
          outputfile.write( "    " + self.classname + "(int treeNumber);\n\n" )
          outputfile.write( "    ~" + self.classname + "();\n\n" )
          outputfile.write( """
    std::vector< peano4::grid::GridControlEvent > createGridControlEventsForNextTraversal() const;
    static void prepareTraversal();
    static void unprepareTraversal();
""" )

          for i in self.operations: self.__generate_operation(outputfile,i)
          outputfile.write( "};\n\n\n" )
          outputfile.write( "#endif\n\n" )

  def get_cpp_file_name(self):
    return self.subdirectory + "/" + self.classname + ".cpp"

  def __generate_implementation(self,overwrite,directory):
    import inspect, os
    filename = os.path.join(directory, self.get_cpp_file_name())
    default_overwrite = True
    if self.implementation.user_should_modify_template():
      default_overwrite = False
    if write_file(overwrite,default_overwrite,filename):
      print( "{} written by {}".format(filename, os.path.basename(inspect.getfile(self.__class__))))
      outputfile = open( filename, "w" )

      #full_qualified_classname = self.__get_full_qualified_class_name()

      outputfile.write( "#include \"" + self.classname + ".h\"\n\n\n" )

      outputfile.write( "tarch::logging::Log " + self.__get_full_qualified_class_name() + "::_log( \""+ self.__get_full_qualified_class_name() + "\");\n\n\n" )

      outputfile.write( self.implementation.get_static_initialisations(self.__get_full_qualified_class_name()) )
      outputfile.write( "\n\n\n" )

      outputfile.write( self.__get_full_qualified_class_name() + "::" + self.classname + "(int treeNumber) {\n" )
      outputfile.write( self.implementation.get_constructor_body() )
      outputfile.write( "}\n\n\n" )

      outputfile.write( self.__get_full_qualified_class_name() + "::~" + self.classname + "() {\n" )
      outputfile.write( self.implementation.get_destructor_body() )
      outputfile.write( "}\n\n\n" )

      outputfile.write( "std::vector< peano4::grid::GridControlEvent > " + self.__get_full_qualified_class_name() + "::createGridControlEventsForNextTraversal() const {\n" )
      outputfile.write( self.implementation.get_implementation_of_createGridControlEventsForNextTraversal() )
      outputfile.write( "}\n\n\n" )

      outputfile.write( "void " + self.__get_full_qualified_class_name() + "::prepareTraversal() {\n" )
      outputfile.write( self.implementation.get_implementation_of_prepareTraversal() )
      outputfile.write( "}\n\n\n" )

      outputfile.write( "void " + self.__get_full_qualified_class_name() + "::unprepareTraversal() {\n" )
      outputfile.write( self.implementation.get_implementation_of_unprepareTraversal() )
      outputfile.write( "}\n\n\n" )

      for operation in self.operations:
        outputfile.write( operation[1] )
        outputfile.write( " " )
        outputfile.write( self.__get_full_qualified_class_name() + "::" + operation[0] )
        outputfile.write( self.__get_operation_arguments(operation) )
        outputfile.write( " {\n" )
        outputfile.write( self.implementation.get_implementation_of_operation( operation[0] ) )
        outputfile.write( "\n" )
        outputfile.write( "}\n\n\n" )

  def generate(self,overwrite,directory):
    if not os.path.exists( os.path.join(directory, self.subdirectory) ):
      os.mkdir(os.path.join(directory, self.subdirectory))
    self.__generate_header(overwrite,directory)
    self.__generate_implementation(overwrite,directory)
