# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import peano4.api.output.TemplatedHeaderImplementationFilePair
import os


class ModelToDataRepository(object):
  def __init__(self,model):
    self.model = model
    self.d     = {}


  def __build_up_dictionary_for_one_data_set(self,i):
    """
     
    To create a data repository, I run over all the faces, vertices, cells of the 
    project and befill the dictionary. There's multiple reasons for this:
    
    - I want to ensure that we include the headers in the repository such that 
      all data types are well-defined.
    - I create a stack map for each and every data entity.
    - I ensure all datatypes are created if this record requires a user-defined MPI
      datatype.
      
      
    i is of type peano4.api.datamodel.DoF.
    
    """
    self.d[ "DATA_CONTAINER_INCLUDES" ]      += i.generator.get_header_file_include()
    self.d[ "DATA_CONTAINER_INCLUDES" ]      += "\n"
    
    self.d[ "DATA_CONTAINER_DECLARATION" ]   += "static peano4::maps::HierarchicalStackMap< " + i.generator.get_stack_container() + ">  _" + i.get_logical_type_name() + "Stack;\n"
    self.d[ "DATA_CONTAINER_INSTANTIATION" ] += "peano4::maps::HierarchicalStackMap< " + i.generator.get_stack_container()  + ">   " + self.d[ "FULL_QUALIFIED_CLASS_NAME" ] + "::_" + i.get_logical_type_name() + "Stack;\n"
    #self.d[ "DATA_CONTAINER_DECLARATION" ]   += "static peano4::maps::STDStackMap< " + i.generator.get_stack_container() + ">  _" + i.get_logical_type_name() + "Stack;\n"
    #self.d[ "DATA_CONTAINER_INSTANTIATION" ] += "peano4::maps::STDStackMap< " + i.generator.get_stack_container()  + ">   " + self.d[ "FULL_QUALIFIED_CLASS_NAME" ] + "::_" + i.get_logical_type_name() + "Stack;\n"
    self.d[ "DATA_CONTAINER_CLEARS" ]        += "_" + i.get_logical_type_name() + "Stack.clear();\n"

    self.d[ "MPI_DATAYPE_INITIALISATION" ]   += i.get_full_qualified_type() + "::initDatatype();\n"
    self.d[ "MPI_DATAYPE_SHUTDOWN" ]         += i.get_full_qualified_type() + "::shutdownDatatype();\n"

      
  def __parse_data_declarations_in_model(self, n):
    subdirectory = self.model.subdirectories[n]
    self.d[ "DATA_CONTAINER_TYPE_DEFS" ]     = ""
    self.d[ "DATA_CONTAINER_INCLUDES" ]      = ""
    self.d[ "DATA_CONTAINER_DECLARATION" ]   = ""
    self.d[ "DATA_CONTAINER_INSTANTIATION" ] = ""
    self.d[ "MPI_DATAYPE_INITIALISATION" ]   = ""
    self.d[ "MPI_DATAYPE_SHUTDOWN" ]         = ""
    self.d[ "DATA_CONTAINER_CLEARS" ]        = ""

    self.d[ "FULL_QUALIFIED_CLASS_NAME" ] = ""
    for i in self.__get_full_namespace(n):
      self.d[ "FULL_QUALIFIED_CLASS_NAME" ] += i
      self.d[ "FULL_QUALIFIED_CLASS_NAME" ] += "::"
    self.d[ "FULL_QUALIFIED_CLASS_NAME" ] += self.__get_class_name()
    
    for i in self.model.cell_data:
      if i.subdirectory == subdirectory:
        self.__build_up_dictionary_for_one_data_set(i)

    for i in self.model.face_data:
      if i.subdirectory == subdirectory:
        self.__build_up_dictionary_for_one_data_set(i)

    for i in self.model.vertex_data:
      if i.subdirectory == subdirectory:
        self.__build_up_dictionary_for_one_data_set(i)

  def __get_full_namespace(self,i):
    return self.model.namespaces[i] + [ "repositories" ]
      
  def __get_class_name(self):
    return "DataRepository"


  def construct_output(self,output):
    for i in range(len(self.model.namespaces)):
      self.__parse_data_declarations_in_model(i)
      
      namespace  = self.__get_full_namespace(i)
      # print("model.subdirectory[", i, "]", " = ", self.model.subdirectories[i])
      # print("model.namespace[", i, "]", " = ", namespace)
      class_name = self.__get_class_name()
      output.makefile.add_cpp_file( self.model.subdirectories[i] + namespace[-1] + "/" + class_name + ".cpp", generated=True )
      templatefile_prefix = os.path.realpath(__file__).replace( ".pyc", "" ).replace( ".py", "" )
      local_d = self.d.copy()
      generated_files = peano4.api.output.TemplatedHeaderImplementationFilePair(
        templatefile_prefix+".template.h",
        templatefile_prefix+".template.cpp",
        class_name, 
        namespace,
        os.path.join(self.model.temporary_subdirectories[i], namespace[-1]),
        local_d,
        True)
      output.add(generated_files)
    pass
