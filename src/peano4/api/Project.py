# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org

import filecmp
import gc
import os
import re
import shutil
import sys
import subprocess
import multiprocessing
import tempfile

import peano4.api.output
import peano4.api.datamodel
import peano4.api.solversteps
import peano4.api.runner

# Bound directly (not via the peano4.api.output.* attribute chain) because this
# name is used in a default parameter value below, which is evaluated while
# peano4.api itself is still being imported - the peano4.api attribute chain
# is not yet fully wired up at that point.
from .output import Overwrite
from .Logger import Logger

class Project (object):
  """!

  Represents a Peano 4 project.

  namespace Sequence of strings representing the (nested) namespace. Pass in
    [ "examples", "algebraicmg" ] for example if you wanna write a solver that
    is embedded into the namespace examples::algebraicmg.


  ## Global project properties

  If you want to alter some global constants, you should manipulate
  the constants attribute. It will eventually feed into the generated
  file Constants.h. See peano4.api.output.Constants for further info.

  """

  def __init__(self, namespace, project_name, directory = ".", subdirectory = "", executable = ""):
    """!
      project_name  Simple string.
    """
    if sys.version_info.major < 3:
      print( "Warning: should be invoked through python3, i.e. with newer Python version" )

    if subdirectory and not os.path.exists(subdirectory):
      os.mkdir(subdirectory)

    self.rootnamespace = namespace.copy()
    self.namespace = namespace
    self.namespaces = []
    self.subdirectories = []
    self.temporary_subdirectories = []
    self.project_name = project_name
    self.directory = directory
    self.temporary_directory = tempfile.mkdtemp()
    self._subdirectory = subdirectory
    self.temporary_subdirectory = tempfile.mkdtemp()
    if subdirectory:
      self.namespace += [ subdirectory ]
    if self.namespace:
      if not self.namespace in self.namespaces:
        self.namespaces.append(self.namespace)
      if not self.subdirectory in self.subdirectories:
        self.subdirectories.append(self.subdirectory)
        self.temporary_subdirectories.append(tempfile.mkdtemp())

    #
    # Create default output model, i.e. those parts that have to be there
    # always
    #
    self.output       = peano4.api.output.Output()

    #
    # Empty model by default
    #
    self.datamodel    = peano4.api.datamodel.Model(self.namespace, self, self.subdirectory)

    self.solversteps  = peano4.api.solversteps.Steps(self)

    self.main         = peano4.api.runner.DefaultSequence(self)

    self.is_generated         = False
    self.is_built             = False
    self.build_was_successful = False

    self.constants  = peano4.api.output.Constants(self)
    self.subconstants = []

    if executable:
      self.output.makefile.set_executable_name(executable)

    self.is_extended = False


  def __str__(self):
    return "(#steps=" + str(self.solversteps) + ",model=" + str(self.datamodel) + ")"

  def set_fenv_handler(self, args):
    self.main.d[ "FENV_ARGS" ] = args

  def cleanup(self):
    """!
      This routine has to be called after you've generated your code.
    """
    self.output.clear()
    self.datamodel.clear()
    self.solversteps.clear()

    self.main         = peano4.api.runner.DefaultSequence(self)

    self.is_generated = False
    self.is_built     = False

    self.constants  = peano4.api.output.Constants(self)
    self.subconstants = []

  def generate(self,
               overwrite=Overwrite.Default,
               throw_away_data_after_generation=False):
    """!
    Generate all code. If you add stuff to your project after a
    build, you have to (re-)generate the code. If you compile only
    once, you don't have to invoke this routine explicitly. It is
    lazily called by the other project routines - the latest before
    you run the code.

    It is important that I reset the output after each generate
    call before you change parameter settings and call generate
    again. To do so, invoke cleanup(). If you forget this, two
    subsequent generate calls enrich the output twice.

    throw_away_data_after_generation: Bool
      The Peano 4 memory footprint can become quite substantial
      effectively reducing the translation capabilities (as compilers
      tend to require a lot of memory, too). So pass in True if you
      want the script to throw away the data structures (and run a
      garbage collection) after all files have been generated. Please
      note that it invalidates both this object (and maybe another
      object that you've used to generate the present one - such as
      ExaHyPE). It really strips everything down to the stuff you
      absolutely need to translate and run the code.
    """
    Logger().new_section("Generate code")
    self.is_generated = True
    self.is_built = False

    # The constants and the output below write straight into self.directory,
    # which only exists already if a solver has created it.
    os.makedirs(self.directory, exist_ok=True)

    self.output.readme.add_package_description( self.constants.readme_entry() )

    self.datamodel.construct_output(self.output)
    self.solversteps.construct_output(self.output)

    self.main.construct_output(self.output,self.project_name + "-main")
    if self.subconstants:
      for i in range(len(self.subconstants)):
        self.subconstants[i].generate(overwrite, self.temporary_directory, self.temporary_subdirectories[i])
    else:
      self.constants.generate(overwrite, self.temporary_directory)

    # Generate the README last, so it is genuinely the final artefact: its
    # "is ready" message (see peano4.api.output.Readme) is meant to be the
    # last thing this routine reports before Logger's own atexit-triggered
    # close() takes over.
    self.output.generate(overwrite, self.temporary_directory, self.temporary_subdirectories)
    #self.output.generate(overwrite, self.directory, self.subdirectories)

    # Check whether any of the code generated into a temporary directory is 
    # different from the code that already exists in the corresponding subdirectory.
    # If so, copy the generated code into the subdirectory. This is intended 
    # to avoid unnecessary recompilation of unchanged code.
    self.temporary_subdirectories.append(self.temporary_subdirectory)
    self.subdirectories.append(self.subdirectory)
    for temporary_subdirectory, subdirectory in zip(self.temporary_subdirectories, self.subdirectories):
      for root, dirs, files in os.walk(temporary_subdirectory):
        for file in files:
          temporary_file_path = os.path.join(root, file)
          relative_path = os.path.relpath(temporary_file_path, temporary_subdirectory)
          target_file_path = os.path.join(self.directory, subdirectory, relative_path)
          # Check if the target file exists and is different from the temporary file
          if not os.path.exists(target_file_path) or not filecmp.cmp(temporary_file_path, target_file_path):
            # Create the target directory if it doesn't exist
            os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
            # Copy the temporary file to the target location
            shutil.copy(temporary_file_path, target_file_path)
            #last_point_location = target_file_path.find('.')
            #while target_file_path.find('.', last_point_location + 1) >= 0:
            #  last_point_location = target_file_path.find('.', last_point_location + 1)
            #file_prefix = target_file_path[0:target_file_path.find('.', last_point_location) - 1]
            #if os.path.isfile(file_prefix + ".o"):
            #  os.remove(file_prefix + ".o")
            #  Logger().log(f"Deleted {file_prefix}.o")
              
            Logger().log(f"Updated {target_file_path} from {temporary_file_path}")
    for root, dirs, files in os.walk(self.temporary_directory):
      for file in files:
        temporary_file_path = os.path.join(root, file)
        relative_path = os.path.relpath(temporary_file_path, self.temporary_directory)
        target_file_path = os.path.join(self.directory, self.directory, relative_path)
        # Check if the target file exists and is different from the temporary file
        if not os.path.exists(target_file_path) or not filecmp.cmp(temporary_file_path, target_file_path):
          # Create the target directory if it doesn't exist
          os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
          # Copy the temporary file to the target location
          shutil.copy(temporary_file_path, target_file_path)
          Logger().log(f"Updated {target_file_path} from {temporary_file_path}")

    if throw_away_data_after_generation:
      self.datamodel = None
      self.solversteps = None
      self.output      = None
      gc.collect()
      print("threw away all data and ran garbage collection")

  def build(self,
            make=True,
            make_clean_first=True,
            throw_away_data_after_build=False,
            number_of_parallel_builds=-1):
    """!
      Invokes the underlying make build mechanism on the project.

      number_of_parallel_builds: int
        This is mapped onto make -jnumber_of_parallel_builds, i.e., it
        determines how many parallel make instances the code spawns.
        Usually, a lot of the generated code is quite lengthy. Therefore
        compile time can be high. If you pass something smaller or equal
        to zero, it will use the core count as guideline how to compile.

      throw_away_data_after_build: Bool
        see generate()
    """
    if number_of_parallel_builds <= 0:
        number_of_parallel_builds = multiprocessing.cpu_count()

    if not self.is_generated:
      self.generate()

    if make:
      if make_clean_first:
        print("clean up project ... in {}".format(self.directory))
        try:
          if make_clean_first:
            subprocess.check_call(["make", "clean"], cwd=self.directory)
          self.is_built = False
          print("clean complete")
        except Exception as e:
          print("clean failed (" + str(e) + ") - continue anyway")

      if not self.is_built:
        print("start to compile with concurrency level of " + str(number_of_parallel_builds) + " ...")
        try:
          process = subprocess.run(["make", "-j"+str(number_of_parallel_builds)], cwd=self.directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
          # Extract warnings and errors from error message
          error_message = process.stderr.decode().strip()
          warnings = re.findall(r"(?i)warning:.*", error_message)
          errors = re.findall(r"(?i)error:.*", error_message)
          # Print warnings into the terminal
          if warnings:
            print("\nWarnings ({}):".format(len(warnings)))
            print("\n".join(warnings))
          if errors:
            print("\nErrors ({}):".format(len(errors)))
            error_message = "\n".join(errors)
          # Check the return code for linker errors
          # Raise exception with errors
          if process.returncode != 0:
            linker_error_message = process.stderr.decode().strip() + "\n" + error_message
            raise Exception(linker_error_message)
          print("compile completed successfully")
          self.is_built = True
          self.build_was_successful = True
        except Exception as e:
          self.is_built = True
          self.build_was_successful = False
          print(str(e))
          print("compile was not successful")
          sys.exit(1)
      else:
        print("cannot build as code generation has not been successful")

    if throw_away_data_after_build:
      self.cleanup()
      self.datamodel    = None
      self.solversteps  = None
      self.output       = None
      gc.collect()
      print("threw away all data and ran garbage collection")

  def run(self, executable, args=[], prefix=None, pipefile=None, rebuild_if_required=True):
    """!
    Runs the code. args should be a list of strings or the empty list.
    prefix is an array, too. A typical invocation looks alike
    project.run( ["16.0", "32.0"], ["/opt/mpi/mpirun", "-n", "1"] )
    The operation returns True if the run had been successful

    pipefile: string or None
    """
    success = False
    if rebuild_if_required and not self.is_built and not self.build_was_successful:
      self.build()

    if not rebuild_if_required or (self.is_built and self.build_was_successful):
      print( "run executable " + str(executable))

      invocation  = []
      if prefix!=None:
        invocation += prefix
      invocation += ["./" + executable]
      invocation += args

      try:
        result = None
        if pipefile==None:
          result = subprocess.run( invocation, stdout=subprocess.PIPE, stderr=subprocess.PIPE )
        else:
          result = subprocess.run( invocation, stdout=open( pipefile, "w" ), stderr=subprocess.PIPE )
        if result.returncode==0:
          if pipefile==None:
            print( result.stdout.decode("utf-8") )
          if result.stderr!=None:
            print( result.stderr.decode("utf-8") )
          print( "run completed without error code, but check for warnings and numerical assertions/errors" )
        else:
          print( result.stderr.decode("utf-8") )
          print( "run failed" )
        success = True
      except Exception as e:
        print( "run of application was not successful: " + str(e) )
        print( "invocation: " + str(invocation) )
    else:
      print( "cannot run as code compilation has not been successful" )
    return success


  def add_subproject(self,
                     subproject: 'Project',
                     master = False): #, subnamespace: str #def merge
    """!
    Adds a new Peano4 project into this Peano 4 project
    """
    """
      assert(
          lhs.namespace == rhs.namespace
      ), "namespaces of the projects being merged don't match"

      assert(
          lhs.directory == rhs.directory
      ), "directories of the projects being merged don't match"

      assert(
          lhs.output.makefile.d["CXX"] == lhs.output.makefile.d["CXX"]
      ), "CXX compilers of the projects being merged don't match"

      assert(
          lhs.output.makefile.d["FC"] == lhs.output.makefile.d["FC"]
      ), "FC compilers of the projects being merged don't match"

      assert(
          lhs.output.makefile.d["DIM"] == lhs.output.makefile.d["DIM"]
      ), "dimensions of the projects being merged don't match"

      assert(
          lhs.output.makefile.d["MODE"] == lhs.output.makefile.d["MODE"]
      ), "compile modes of the projects being merged don't match"
    """
      # assert(
      #   lhs._domain_offset == rhs._domain_offset and
      #   lhs._domain_size == rhs._domain_size and
      #   lhs._dimensions == rhs._dimensions and
      #   lhs._plotter_precision == rhs._plotter_precision
      # ), "global simulation parameters of the projects being merged don't match"

      # an internal utility function to merge two dictionaries
      # def _merge(lhs_dict: dict, rhs_dict: dict) -> dict:
      #   dict_merged = lhs_dict.copy()
      #   for key,value in rhs_dict.items():
      #     if key in dict_merged:
      #         dict_merged[key] += value
      #     else:
      #         dict_merged[key] = value
      #   return dict_merged

    assert(subproject._subdirectory != ""), "subdirectory of a subproject mustn't be empty"

    self.namespace = subproject.namespace
    self.namespaces.append(subproject.namespace)
    self.datamodel.namespace = subproject.datamodel.namespace
    self.datamodel._subdirectory = subproject.datamodel._subdirectory
    if not subproject.subdirectory in self.subdirectories:
      self.subdirectories.append(subproject.subdirectory)
      self.temporary_subdirectories.append(tempfile.mkdtemp())
    if not self.datamodel.namespace in self.datamodel.namespaces:
      self.datamodel.namespaces.append(self.datamodel.namespace)
    if not self.datamodel.subdirectory in self.datamodel.subdirectories:
      self.datamodel.subdirectories.append(self.datamodel.subdirectory)


    #
    # merge all DaStGen2 and ParticleSet attributes of datamodel
    #
    for data in subproject.datamodel.cell_data:
      self.datamodel.add_cell(data)
    for data in subproject.datamodel.face_data:
      self.datamodel.add_face(data)
    for particle in subproject.datamodel.vertex_data:
      self.datamodel.add_vertex(particle)
    for particle in subproject.datamodel.global_data:
      self.datamodel.add_global(particle)

    #
    # merge all steps of solversteps
    #

    # make the steps currently contained in the Peano4 project aware of the incoming subproject's data
    for step in self.solversteps._steps:
      added_step = subproject.solversteps._steps[0]
      step.cell_data.extend(added_step.cell_data)
      step.face_data.extend(added_step.face_data)
      step.vertex_data.extend(added_step.vertex_data)

    # make the incoming subproject's steps aware of the Peano4 project data
    if len(self.solversteps._steps):
      for added_step in subproject.solversteps._steps:
        step = self.solversteps._steps[0]
        added_step.cell_data.clear()
        added_step.cell_data.extend(step.cell_data)
        added_step.face_data.clear()
        added_step.face_data.extend(step.face_data)
        added_step.vertex_data.clear()
        added_step.vertex_data.extend(step.vertex_data)

    # finally append the incoming subproject's steps to the Peano4 project's steps
    for step in subproject.solversteps._steps:
      self.solversteps._steps.append(step)

    #
    # merge all keys and strings of output.makefile
    #

    # ... for compilers
    if (self.output.makefile.d["CXX"]):
      assert (
          self.output.makefile.d["CXX"] == subproject.output.makefile.d["CXX"]
      ), "the CXX compiler of the subproject being added doesn't match the one of the main project"
    else:
      self.output.makefile.d["CXX"] = subproject.output.makefile.d["CXX"]

    if (self.output.makefile.d["FC"]):
      assert (
          self.output.makefile.d["FC"] == subproject.output.makefile.d["FC"]
      ), "the FC compiler of the subproject being added doesn't match the one of the main project"
    else:
      self.output.makefile.d["FC"] = subproject.output.makefile.d["FC"]

    for flag in subproject.output.makefile.d["CXXFLAGS"].split():
      self.output.makefile.add_CXX_flag(flag)

    for flag in subproject.output.makefile.d["FCFLAGS"].split():
      self.output.makefile.add_Fortran_flag(flag)

    for flag in subproject.output.makefile.d["LDFLAGS"].split():
      self.output.makefile.add_linker_flag(flag)

    self.output.makefile.d["LIBS"] += subproject.output.makefile.d["LIBS"]
    self.output.makefile.d["GENERATED_INCLUDE_DIRS"] |= subproject.output.makefile.d["GENERATED_INCLUDE_DIRS"]
    self.output.makefile.d["MODE"] = subproject.output.makefile.d["MODE"]

    for module in subproject.output.makefile.d["FORTRAN_MODULES"]:
      self.output.makefile.add_Fortran_module(module)

    assert (
        self.output.makefile.d["EXECUTABLENAME"]
    ), "the name of the main project is empty"
    if not self.output.makefile.d["DIM"]:
      self.output.makefile.set_dimension(subproject.output.makefile.d["DIM"])
    else:
      assert (
          self.output.makefile.d["DIM"] == subproject.output.makefile.d["DIM"]
      ), "the dimensions of the added subproject doesn't match the current one of the main project"

    # ... for filepaths
    for filename in subproject.output.makefile.hfiles:
      self.output.makefile.add_h_file(filename, False)
    for filename in subproject.output.makefile.cppfiles:
      self.output.makefile.add_cpp_file(filename, False)
    for filename in subproject.output.makefile.fortranfiles:
      self.output.makefile.add_Fortran_file(filename, False)

    for filename in subproject.output.makefile.generated_hfiles:
      self.output.makefile.add_h_file(filename, True)
    for filename in subproject.output.makefile.generated_cppfiles:
      self.output.makefile.add_cpp_file(filename, True)
    for filename in subproject.output.makefile.generated_fortranfiles:
      self.output.makefile.add_Fortran_file(filename, True)

    #
    # merge all strings and artefacts of readme
    #
    for i in subproject.output.readme._entries:
      self.output.readme.add_entry(i)

    for i in subproject.output.readme._package_descriptions:
      self.output.readme.add_package_description(i)

    for i in subproject.output.readme._entries:
      self.output.readme.add_entry(i)

    for i in subproject.output.artefacts:
      self.output.add(i)

    #
    # the last added subproject becomes the master project (which will provide the template for the main file)
    #
    if not self.is_extended:
      self.main = subproject.main
      self.rootnamespace = subproject.rootnamespace
      self._subdirectory = subproject._subdirectory
      self.is_extended = True
    self.main.merge(subproject.main)

    # self.main.project = self
    # self.main.overwrite = subproject.main.overwrite
    # self.main.d.update(subproject.main.d)

    #
    # merge all constants
    #
    self.subconstants.append(subproject.constants)


  @property
  def subdirectory(self):
      subdirectory = self._subdirectory
      if subdirectory:
          subdirectory += "/"
      return subdirectory
