# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import peano4.api.output.Jinja2TemplatedHeaderImplementationFilePair
import os

from .DoF import DoFAssociation
    
class FloatTypes():
  def __init__(self, float_type):
      if float_type=="float":
        self.cpp = "float"
        self.mpi = "MPI_FLOAT"
        self.number_of_mpi_types_per_element = 1
      elif float_type=="double":
        self.cpp = "double"
        self.mpi = "MPI_DOUBLE"
        self.number_of_mpi_types_per_element = 1
      elif float_type=="long double":
        self.cpp = "long double"
        self.mpi = "MPI_LONG_DOUBLE"
        self.number_of_mpi_types_per_element = 1
      elif float_type=="_Float16":
        self.cpp = "_Float16"
        self.mpi = "MPI_BYTE"
        self.number_of_mpi_types_per_element = 2
      elif float_type=="std::float16_t":
        self.cpp = "std::float16_t"
        self.mpi = "MPI_BYTE"
        self.number_of_mpi_types_per_element = 2
      elif float_type=="__bf16":
        self.cpp = "__bf16"
        self.mpi = "MPI_BYTE"
        self.number_of_mpi_types_per_element = 2
      elif float_type=="std::bfloat16_t":
        self.cpp = "std::bfloat16_t"
        self.mpi = "MPI_BYTE"
        self.number_of_mpi_types_per_element = 2
      else:
        raise Exception("Float type " + float_type + " not recognized")


class PatchToDoubleArray():
  """
  
  Very simple converter which maps the patch 1:1 onto a double array. 
  By default, I leave all the MPI merges empty though obviously the 
  merges often are plain copying over along faces (see the discussion 
  on Finite Volumes in the guidebook). If you need any proper MPI 
  handling, use merge_method_definition to inject it. Depending
  on how you use your patch (as cell or face or vertex data structure)
  the thing you pass has the respective semantics. The most popular
  default merge implementation is the one you find in the blockstructured
  toolbox.
  
  """
  def __init__(self,patch, float_type="double"):
    """
    
      includes  Includes to be added to the generated file. This is something
      totally different than get_header_file_include(). You will need to set
      something here if your send/receive conditions, e.g., call other stuff.
      
    """
    self.data = patch
    self.merge_method_definition     = ""
    self.send_condition              = "true"
    self.receive_and_merge_condition = "true"
    # @todo Das ist voellig falsch. Das sollte direkt im Patch sitzen und nicht im generator
    self.load_store_compute_flag     = "::peano4::grid::LoadStoreComputeFlag::LoadFromInputStream_ProvideToCalculations_StoreToOutputStream"
    self.includes                    = ""
    self.float_type                  = FloatTypes(float_type)


  def __str__(self):
    result =  """Map to double array
  - send condition:          """ + self.send_condition + """
  - receive/merge condition: """ + self.receive_and_merge_condition + """
  - load/store/compute flag: """ + self.load_store_compute_flag + """
  - merge criterion:         """
  
    if self.merge_method_definition!="":
      result += """user-defined
"""
    else:    
      result += """nop
"""
    return result    


  def get_stack_container(self):
    return "peano4::stacks::STDVectorStack< " + self.data.get_full_qualified_type() + " >";

    
  def get_header_file_include(self):
    """
    
     This is the include statement for the data container. It is not something
     directly generated but how to use the generated code. 
     
    """
    return """
#include "peano4/stacks/STDVectorStack.h"    
#include \"""" + self.data.namespace[-1] + "/" + self.data.name + """.h"
"""


  def _get_dictionary_for_output(self):
    d = { 
        "CARDINALITY_2D": str( int( self.data.no_of_unknowns*self.data.dim[0]*self.data.dim[1] )), 
        "CARDINALITY_3D": str( int( self.data.no_of_unknowns*self.data.dim[0]*self.data.dim[1]*self.data.dim[2] )),
        "MERGE_METHOD_DECLARATIONS": "",
        "MERGE_METHOD_DEFINITIONS": "",
        "SEND_CONDITION":              self.send_condition,
        "RECEIVE_AND_MERGE_CONDITION": self.receive_and_merge_condition,
        "INCLUDES":                    self.includes,
        "DATA_ASSOCIATION":            int(self.data.association),
        "MERGE_METHOD_DEFINITION":     self.merge_method_definition,
        "LOAD_STORE_COMPUTE_FLAG":     self.load_store_compute_flag,
        "FLOAT_TYPE":                  self.float_type.cpp,
        "MPI_FLOAT_TYPE":              self.float_type.mpi,
        "MPI_ELEMENTS_PER_FLOAT":      self.float_type.number_of_mpi_types_per_element,
        "ADDITIONAL_LOAD_STORE_ARGUMENTS": self.data.additional_load_and_store_arguments
      }
    return d


    
  def construct_output(self,output):
    d = self._get_dictionary_for_output()
    """
      Pass in a version of output
    """
    output.makefile.add_cpp_file( self.data.subdirectory + self.data.namespace[-1] + "/" + self.data.name + ".cpp", generated=True )
    templatefile_prefix = os.path.realpath(__file__).replace( ".pyc", "" ).replace( ".py", "" )
    generated_files = peano4.api.output.Jinja2TemplatedHeaderImplementationFilePair(
      templatefile_prefix+".template.h",
      templatefile_prefix+".template.cpp",
      self.data.name, 
      self.data.namespace,
      os.path.join(self.data.temporary_subdirectory, self.data.namespace[-1]),
      d, 
      True)
    output.add(generated_files)
