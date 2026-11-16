# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import os
import re
import sys
import subprocess
import jinja2

from .Helper import write_file

from .CompileMode import CompileMode


class Makefile(object):
    """Represents the created Makefile of a Peano 4 project"""

    default_overwrite = True

    def __init__(self):
        self.clear()

    @property
    def readme_entry(self):
        result_template = jinja2.Template(
            """
## Makefile settings

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CXX             = {{CXX}}
CXXFLAGS        = {{CXXFLAGS}}
FC              = {{FC}}
FCFLAGS         = {{FCFLAGS}}
LDFLAGS         = {{LDFLAGS}}
LIBS            = {{LIBS}}
DIM             = {{DIM}}
CONFIGUREPATH   = {{CONFIGUREPATH}}
EXECUTABLENAME  = {{EXECUTABLENAME}}
FORTRAN_MODULES = {{FORTRAN_MODULES}}
MODE            = {{MODE}}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Compiler version

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{{VERSION}}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
        )
        try:
            version_string = subprocess.run(
                [self.d["CXX"], "--version"], stdout=subprocess.PIPE
            )
        except Exception:
            version_string = subprocess.run(
                ["echo", "Unknown"], shell=True, stdout=subprocess.PIPE
            )
        self.d["VERSION"] = version_string.stdout.decode("utf-8")
        return result_template.render(**self.d) + self.configure_call

    @property
    def readme_package_descriptor(self):
        return jinja2.Template(
            """
{% if Parallel %}

### MPI parallelisation

Peano's distributed memory parallelisation is based upon plain space-filling
curves which are used first to split the cells among the ranks, and, after
that, to split the cells into chunks and deploy them onto cores (if shared
memory is used). There is no particularly novel contribution/paper on this
aspect of the code. The paper that is (algorithmically) most interesting is
an old EuroPar publication:

       @InProceedings{Bungartz:2006:Parallelisation,
         author={Bungartz, Hans-Joachim and Mehl, Miriam and Weinzierl, Tobias},
         editor={Nagel, Wolfgang E. and Walter, Wolfgang V. and Lehner, Wolfgang},
         title={A Parallel Adaptive Cartesian PDE Solver Using Space--Filling Curves},
         booktitle={Euro-Par 2006 Parallel Processing},
         year={2006},
         publisher={Springer Berlin Heidelberg},
         address={Berlin, Heidelberg},
         pages={1064--1074},
         abstract={In this paper, we present a parallel multigrid PDE solver working on adaptive hierarchical cartesian grids. The presentation is restricted to the linear elliptic operator of second order, but extensions are possible and have already been realised as prototypes. Within the solver the handling of the vertices and the degrees of freedom associated to them is implemented solely using stacks and iterates of a Peano space--filling curve. Thus, due to the structuredness of the grid, two administrative bits per vertex are sufficient to store both geometry and grid refinement information. The implementation and parallel extension, using a space--filling curve to obtain a load balanced domain decomposition, will be formalised. In view of the fact that we are using a multigrid solver of linear complexity {\\$}{\\backslash}mathcal{\\{}O{\\}}(n){\\$}, it has to be ensured that communication cost and, hence, the parallel algorithm's overall complexity do not exceed this linear behaviour.}, 
         isbn={978-3-540-37784-9}
       }

{% endif %}
{% if SharedOMP %}

### OpenMP parallelisation

Peano 4 run uses a wrapper around OpenMP to obtain a high task efficiency.
The wrapper can be read as user-level threading implemented on top of OpenMP's
tasking mechanism. It is described in

       @article{Schulz:2021:Tasking,
         title = {Task inefficiency patterns for a wave equation solver},
         journal = {IWOMP},"
         year = {2021},
         author = {Holger Schulz and Gonzalo Brito Gadeschi and Oleksandr Rudyy and Tobias Weinzierl},
      }

{% endif %}

{% if SharedOMP %}

Peano relies on a mixture of classic domain decomposition and task-based
parallelism. The domain decomposition provides the baseline performance,
and the tasking adds the big flexibility and scalability gain on top. The
key publication discussing the overall idea and algorithmic ingredients is
the SISC paper

       @article{Charrier:2020:Enclave,
         author = {Charrier, Dominic Etienne and Hazelwood, Benjamin and Weinzierl, Tobias},
         title = {Enclave Tasking for DG Methods on Dynamically Adaptive Meshes},
         journal = {SIAM Journal on Scientific Computing},
         volume = {42},
         number = {3},
         pages = {C69-C96},
         year = {2020},
         doi = {10.1137/19M1276194},
         URL = {https://doi.org/10.1137/19M1276194},
         eprint = {https://doi.org/10.1137/19M1276194}
       }

{% endif %}

"""
        ).render(**self.d)

    def executable_name(self):
        return self.d["EXECUTABLENAME"]

    def clear(self):
        self.configure_call = ""
        self.d = {}
        self.d["CXX"] = ""
        self.d["CXXFLAGS"] = ""
        self.d["FC"] = ""
        self.d["FCFLAGS"] = ""
        self.d["LDFLAGS"] = ""
        self.d["LIBS"] = "-lomp"
        self.d["DIM"] = "2"
        self.d["CONFIGUREPATH"] = "."
        self.d["EXECUTABLENAME"] = ""
        self.d["CMAKE_BUILD_DIR"] = ""
        self.d["FORTRAN_MODULES"] = []
        self.d["APP_SUBDIRECTORIES"] = []
        self.set_mode(CompileMode.DEBUG)
        self.clear_files()

    def set_executable_name(self, fname):
        self.d["EXECUTABLENAME"] = fname

    def clear_files(self):
        self.hfiles = []
        self.cppfiles = []
        self.mlir_dict = {}
        self.fortranfiles = []
        self.generated_hfiles = []
        self.generated_cppfiles = []
        self.generated_fortranfiles = []

    def set_dimension(self, dimension):
        self.d["DIM"] = str(dimension)

    def get_configure_path(self):
        """
        Returns the directory where the ./configure script is located
        """
        return self.d["CONFIGUREPATH"]

    def get_source_path(self):
        """
        Returns the directory where the ./configure script is located
        """
        return self.get_configure_path() + "/src"

    def add_header_search_path(self, path):
        """
        Add the header search path to both the C++ and the Fortran
        call command.
        """
        self.d["CXXFLAGS"] += " -I" + path
        self.d["FCFLAGS"] += " -I" + path

    def add_library(self, library_name, library_path=""):
        """
        If you want to link against a library from Peano, feel free to use
        get_Peano4_source_directory() and hand in a concatenation of this
        path plus a subpath. Otherwise, specify the absolute path where to
        search for. By default, Peano's src directory is in the search
        path.

        A popular invocation including one of Peano's toolboxes is

        project.output.makefile.add_library("ToolboxFiniteElements2d_trace", project.output.makefile.get_source_path() + "/toolbox/finiteelements")
        """
        if library_path != "":
            self.d["LIBS"] = "-L" + library_path + " " + self.d["LIBS"]
        self.d["LIBS"] += library_name + " "

    def set_mode(self, mode):
        """
        mode should be of type CompileMode. Pass in

        peano4.api.output.CompileMode.DEBUG

        for example. Debug is the default.
        """
        if mode == CompileMode.DEBUG:
            self.d["MODE"] = "DEBUG"
        elif mode == CompileMode.ASSERTS:
            self.d["MODE"] = "ASSERTS"
        elif mode == CompileMode.STATS:
            self.d["MODE"] = "STATS"
        elif mode == CompileMode.TRACE:
            self.d["MODE"] = "TRACE"
        elif mode == CompileMode.RELEASE:
            self.d["MODE"] = "RELEASE"
        else:
            assert False

    def set_CXX_compiler(self, value):
        self.d["CXX"] = value

    def set_CXX_flags(self, value):
        self.d["CXXFLAGS"] = value

    def add_CXX_flag(self, value, force=False):
        if value in self.d["CXXFLAGS"] and not force:
            print(
                "CXXFLAG "
                + value
                + " is already in list of flags. Ignored as force attribute is not set"
            )
        else:
            self.d["CXXFLAGS"] += " " + value

    def set_Fortran_compiler(self, value):
        self.d["FC"] = value

    def set_Fortran_flags(self, value):
        self.d["FCFLAGS"] = value

    def add_Fortran_flag(self, value, force=False):
        if value in self.d["FCFLAGS"] and not force:
            print(
                "FCFLAGS "
                + value
                + " is already in list of flags. Ignored as force attribute is not set"
            )
        else:
            self.d["FCFLAGS"] += " " + value

    def add_linker_flag(self, value):
        self.d["LDFLAGS"] += " " + value

    def set_linker_flags(self, value):
        self.d["LDFLAGS"] = value + " "

    def parse_configure_script_outcome(self, directory):
        """
        directory should point to the directory which holds the ./configure script.
        It furthermore has to be invoked after configure has passed successfully.
        This script does not accept relative paths. We then search for the subdirectory
        src and parse the Makefile there.
        """
        if 'PEANO_SRC_ROOT_DIR' in os.environ:
          directory = os.environ['PEANO_SRC_ROOT_DIR']

        self.d["CONFIGUREPATH"] = directory

        if 'PEANO_CMAKE_BUILD_DIR' in os.environ: # If this environment flag has been set, favour CMake
            self.d["CMAKE_BUILD_DIR"] = os.environ['PEANO_CMAKE_BUILD_DIR']
        else: # If not, and an Automake configuration is present, favour this instead
          try:
              input_file = directory + "/config.log"
              input = open(input_file, "r")
              self.configure_call = ""
              print("parse configure outcome " + input_file + " to extract configure settings")
              MakefileDefined = [
                  "Parallel",
                  "SharedOMP",
                  "SharedSYCL",
                  "SharedCPP",
              ]

              for line in input:
                  if "./configure" in line and self.configure_call == "":
                      print("found the configure call info " + line)
                      self.configure_call = line.strip()
                  for define in MakefileDefined:
                      if re.search("#define " + define, line):
                          self.d[define] = "1"

              input_file = directory + "/src/Makefile"
              input = open(input_file, "r")
              print("parse configure outcome " + input_file + " to extract compile settings")
              MakefileConstants = [
                  "CXXFLAGS_PEANO_2D_RELEASE",
                  "CXXFLAGS_PEANO_2D_STATS",
                  "CXXFLAGS_PEANO_2D_ASSERTS",
                  "CXXFLAGS_PEANO_2D_TRACE",
                  "CXXFLAGS_PEANO_2D_DEBUG",
                  "CXXFLAGS_PEANO_3D_RELEASE",
                  "CXXFLAGS_PEANO_3D_STATS",
                  "CXXFLAGS_PEANO_3D_ASSERTS",
                  "CXXFLAGS_PEANO_3D_TRACE",
                  "CXXFLAGS_PEANO_3D_DEBUG",
                  "LDFLAGS_PEANO_RELEASE",
                  "LDFLAGS_PEANO_STATS",
                  "LDFLAGS_PEANO_ASSERTS",
                  "LDFLAGS_PEANO_TRACE",
                  "LDFLAGS_PEANO_DEBUG",
                  "LDADD_PEANO_2D_RELEASE",
                  "LDADD_PEANO_2D_STATS",
                  "LDADD_PEANO_2D_ASSERTS",
                  "LDADD_PEANO_2D_TRACE",
                  "LDADD_PEANO_2D_DEBUG",
                  "LDADD_PEANO_3D_RELEASE",
                  "LDADD_PEANO_3D_STATS",
                  "LDADD_PEANO_3D_ASSERTS",
                  "LDADD_PEANO_3D_TRACE",
                  "LDADD_PEANO_3D_DEBUG",
                  "CXX",
                  "FC",
                  "CXXFLAGS",
                  "LDFLAGS",
                  "LIBS",
                  "GPU_VENDOR",
                  "GPU_ARCH"
              ]

              for line in input:
                  for constant in MakefileConstants:
                      if re.search(constant + " *=", line) and line.startswith(constant):
                          try:
                              flags = line.split("=", 1)[1].strip()
                              #print("add " + constant + "=" + flags)
                              self.d[constant] = flags
                          except:
                              print("Error in " + line + " for token " + constant)
              return
          except IOError:
              self.d["CMAKE_BUILD_DIR"] = next((path for path in (os.path.join(directory, subdir) for subdir in os.listdir(directory)) if os.path.isfile(os.path.join(path, "PeanoTargets.cmake"))), "")
              pass # If there is no Automake configuration we try CMake as last attempt

        if self.d["CMAKE_BUILD_DIR"] != "":
            cmake_cache_file = self.d["CMAKE_BUILD_DIR"] + "/CMakeCache.txt"
            if os.path.isfile(cmake_cache_file):
                with open(cmake_cache_file, 'r') as input:
                     for line in input:
                          key_value = line.split("=", 1)
                          if len(key_value) == 2:
                              key, value = key_value
                              key = key.split(":")[0].strip()
                              if key == "CMAKE_CXX_COMPILER":
                                  self.d["CXX"] = value.strip()
                              if key == "CMAKE_Fortran_COMPILER":
                                  self.d["FC"] = value.strip()
        else: # Nothing has been configured so far
              raise Exception("""
  Error: if you call parse_configure_script_outcome(), please hand over directory where
  ./configure had been called. You passed """ + directory
                              )

    def add_h_file(self, filename, generated=False):
        """
        Add a new header filename.
        This is actually not needed for compilation,
        but is useful to have for IDEs where the header files can be displayed
        in the project.

        filename: String
           Filename of a C/C++ header file. They usually should have the .h/.hpp extension.
        generated: Bool
           Use this flag for generated files which can be tracked
           by the cleaning routines 'distclean' or 'maintainer-clean' of the Makefile.
        """
        if generated:
            if (
                self.generated_hfiles.count(filename) == 0
                and self.hfiles.count(filename) == 0
            ):
                self.generated_hfiles.append(filename)
        else:
            if self.hfiles.count(filename) == 0:
                self.hfiles.append(filename)
                if self.generated_hfiles.count(filename) > 0:
                    self.generated_hfiles.remove(filename)

    def remove_h_file(self, filename):
        if filename in self.generated_hfiles:
            self.generated_hfiles.remove(filename)
        if filename in self.hfiles:
            self.hfiles.remove(filename)

    def add_cpp_file(self, filename, generated=False):
        """
        Add a new cpp filename. This is basically a set implementation, i.e., you can
        add files multiple times, but they are not inserted multiple times. This
        is important, as the steps add the cpp files. Multiple steps can hold the
        same action, so this action would be created multiple times.

        All the standard Peano 4 routines rely on this function to add their
        generated files to the build environment. Nothing stops you however to
        add more files yourself.

        filename: String
           Filename of a C++ file. They usually should have the .cpp/.cxx extension.
        generated: Bool
           Use this flag for generated files which can be tracked
           by the cleaning routines 'distclean' or 'maintainer-clean' of the Makefile.
        """
        if generated:
            # Non-generated file takes precedence over generated files.
            # We only add a generated file if has not been added yet as a non-generated file instead.
            if (
                self.generated_cppfiles.count(filename) == 0
                and self.cppfiles.count(filename) == 0
              ):
                self.generated_cppfiles.append(filename)
        else:
            # If a file is a non-generated file we definitely want to add it.
            if self.cppfiles.count(filename) == 0:
                self.cppfiles.append(filename)
                # We ensure that we only add a .cpp file once.
                # Otherwise we would get duplicated symbols errors.
                # But we never delete a non-generated file.
                if self.generated_cppfiles.count(filename) > 0:
                    self.generated_cppfiles.remove(filename)

    def remove_cpp_file(self, filename):
        if filename in self.generated_cppfiles:
            self.generated_cppfiles.remove(filename)
        if filename in self.cppfiles:
            self.cppfiles.remove(filename)
    
    def add_mlir_file(self, filename, pipeline):
        """
        Add a new mlir filename. It gets added to a dictionary where the associated value
        is the pipeline that should be applied to the file.

        filename: String
           Filename of an MLIR file. They have the .mlir extension.
        pipeline: String
            MLIR pipeline that should be applied to the file.
        generated: Bool
           Use this flag for generated files which can be tracked
           by the cleaning routines 'distclean' or 'maintainer-clean' of the Makefile.
        """
        mlir_prefix = filename.rsplit(".", 1)[0]
        self.mlir_dict[mlir_prefix] = pipeline

    def remove_mlir_file(self, filename):
        mlir_prefix = filename.rsplit(".", 1)[0]
        if mlir_prefix in self.mlir_dict:
            del self.mlir_dict[mlir_prefix]

    def add_Fortran_file(self, filename, generated=False):
        """
        Add a new Fortran file.

        All the standard Peano 4 routines rely on this function to add their
        generated files to the build environment. Nothing stops you however to
        add more files yourself. Don't add a file multiple times. This might
        break the compiler.

        Fortran is really picky about the translation order. So you have to add
        the stuff in the right order. Otherwise, Fortran might complain. This is
        your responsibility.

        If your file defines a module, please do not use this routine, but use
        add_Fortran_module() instead.

        filename: String
           Filename of a Fortran file. They usually should have the .f90 extension.
        generated: Bool
           Use this flag for generated files which can be tracked
           by the cleaning routines 'distclean' or 'maintainer-clean' of the Makefile.
        """
        if generated:
            if (
                self.generated_fortranfiles.count(filename) == 0
                and self.fortranfiles.count(filename) == 0
            ):
                self.generated_fortranfiles.append(filename)
        else:
            if self.fortranfiles.count(filename) == 0:
                self.fortranfiles.append(filename)
                if self.generated_fortranfiles.count(filename) > 0:
                    self.generated_fortranfiles.remove(filename)

    def remove_Fortran_file(self, filename):
        if filename in self.generated_fortranfiles:
            self.generated_fortranfiles.remove(filename)
        if filename in self.fortranfiles:
            self.fortranfiles.remove(filename)

    def add_Fortran_module(self, module_file, force=False):
        """
        Add a Fortran module

        module_file: String
          Filename of a Fortran source code file which hosts a module. It should
          have the extension .f90 or similar.

        force: Boolean
          Enforce that Fortran module is added even though it might already be in
          the list.
        """
        if not module_file.endswith(".f90"):
            print(
                "Warning: Fortran module file does not have extension .f90 ("
                + module_file
                + ") and translation thus might fail"
            )
        if module_file in self.d["FORTRAN_MODULES"] and not force:
            print(
                """Fortran module file
"""
                + module_file
                + """
is already in module file list. Did not add it once more. You can overwrite
this default behaviour via the force attribute in add_Fortran_module(). If
you create multiple Peano 4 makefiles in a row (as you change parameters, e.g.)
then this message can typically be ignored.
"""
            )
        elif module_file in self.d["FORTRAN_MODULES"] and force:
            print(
                "Fortran module file "
                + module_file
                + " is already in module file list but force flag is set. Add it"
            )
            self.d["FORTRAN_MODULES"].append(module_file)
        else:
            self.d["FORTRAN_MODULES"].append(module_file)

    def add_Fortran_modules(self, module_files):
        for i in module_files:
            self.add_Fortran_module(i)

    def generate(self, overwrite, directory, subdirectories=[]):
        """
        Generates build files and other project-related files based on templates.

        :param overwrite: Specifies whether existing files should be overwritten.
        :type overwrite: bool
        :param directory: The directory where the generated files will be placed.
        :type directory: str
        """
        self.d["H_HEADERS"] = self.hfiles
        self.d["CXX_SOURCES"] = self.cppfiles
        self.d["FORTRAN_SOURCES"] = self.fortranfiles
        self.d["GENERATED_H_HEADERS"] = self.generated_hfiles
        self.d["GENERATED_CXX_SOURCES"] = self.generated_cppfiles
        self.d["GENERATED_FORTRAN_SOURCES"] = self.generated_fortranfiles
        if subdirectories:
            for subdirectory in subdirectories:
                self.d["APP_SUBDIRECTORIES"].append(subdirectory)

        #self.d["MLIR_PREFIXES"] = mlir_prefixes
        self.d["MLIR_DICT"] = self.mlir_dict
        self.d["MLIR_OBJS"] = " ".join([f"{prefix}_mlir.o" for prefix in self.mlir_dict.keys()])

        # Encapsulate file generation as a function
        def generate_output_file(input_file_path, output_file_path):
            if write_file(overwrite, self.default_overwrite, output_file_path):
                print("write " + output_file_path)
                template_loader = jinja2.FileSystemLoader(searchpath=os.path.split(input_file_path)[0])
                templateEnv = jinja2.Environment(loader=template_loader)
                template = templateEnv.get_template(os.path.split(input_file_path)[1])
                with open(output_file_path, "w") as output:
                    output.write(template.render(self.d))

        file_dir = os.path.dirname(os.path.realpath(__file__))

        generate_output_file(os.path.join(file_dir, "Gitignore.template"), os.path.join(directory, ".gitignore"))

        if self.d["CMAKE_BUILD_DIR"] != "":
            generate_output_file(os.path.join(file_dir, "CMakeLists.txt.template"), os.path.join(directory, "CMakeLists.txt"))
            # Shortcut to the CMake build. Used to mirror Autotools.
            generate_output_file(os.path.join(file_dir, "Makefile.cmake.template"), os.path.join(directory, "Makefile"))
        else:
            generate_output_file(os.path.join(file_dir, "Makefile.template"), os.path.join(directory, "Makefile"))
