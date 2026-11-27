import exahype2.api
import exahype2.api.dslhype2 as dslhype
from exahype2.api import Target, OutputLanguage, MemoryLayout, IncludeMLIRFunctions
from ..Solver import Solver

import os
from collections import OrderedDict

def initialiseCentreOfPatch2D(data: "double*", x: "tarch::la::Vector<2, double>", h: "tarch::la::Vector<2, double>"):
    data_block = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, NV], [0, 1]], "data")
    initial_data = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, NV], [0, 1]], eval("initPointWisely", getVolumeCentre(x, h, NV)))
    data_block = initial_data

def initialiseFullPatch2D(data: "double*", x: "tarch::la::Vector<2, double>", h: "tarch::la::Vector<2, double>"):
    data_block = DataBlock([[0, NumberOfUnknowns], [-H, NV + H], [-H, NV + H], [0, 1]], "data")
    initial_data = DataBlock([[0, NumberOfUnknowns], [-H, NV + H], [-H, NV + H], [0, 1]], eval("initPointWisely", getVolumeCentre(x, h, NV)))
    data_block = initial_data

def initialiseCentreOfPatch3D(data: "double*", x: "tarch::la::Vector<2, double>", h: "tarch::la::Vector<2, double>"):
    data_block = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, NV], [0, NV], [0, 1]], "data")
    initial_data = DataBlock([[0, NumberOfUnknowns], [0, NV], [0, NV], [0, NV], [0, 1]], eval("initPointWisely", getVolumeCentre(x, h, NV)))
    data_block = initial_data

def initialiseFullPatch3D(data: "double*", x: "tarch::la::Vector<2, double>", h: "tarch::la::Vector<2, double>"):
    data_block = DataBlock([[0, NumberOfUnknowns], [-H, NV + H], [-H, NV + H], [-H, NV + H], [0, 1]], "data")
    initial_data = DataBlock([[0, NumberOfUnknowns], [-H, NV + H], [-H, NV + H], [-H, NV + H], [0, 1]], eval("initPointWisely", getVolumeCentre(x, h, NV)))
    data_block = initial_data

class InitialConditionsGenerator(Solver.NumericalScheme.NumericalSchemeGenerator):
    def construct_output(self, exahype2_project, peano_project, time_step, add_global_tasks=True):
        self.__create_initial_condition_functions(exahype2_project, peano_project)

    def __create_initial_condition_functions(self, exahype2_project, peano4_project):
        """!
        
        Calling this function is delayed until all PDE term information is provided.
        In this way we can avoid generating code that will not be used. The following
        information is required to translate a Python function using dslhype:
        - template_parameters: the variables that are used within the Python function
            to represent external information
        - functor_parameters: the PDE terms that can be used by the solver
        - function: the Python function itself

        We never generate an MLIR version of this kernel, since it is just not worth it
        """

        dimensions = exahype2_project._dimensions
        if dimensions == 2:
            initalise_centre_of_patch = initialiseCentreOfPatch2D
            initalise_full_patch = initialiseFullPatch2D
        else:
            initalise_centre_of_patch = initialiseCentreOfPatch3D
            initalise_full_patch = initialiseFullPatch3D
        
        output_directory = peano4_project.directory
        temporary_output_directory = peano4_project.temporary_directory

        required_includes = f"""
#include "Abstract{self._solver.name}.h"
#include "repositories/SolverRepository.h"
#include "exahype2/Solver.h"
#include "exahype2/CellData.h"
#include "exahype2/VolumeIndex.h"
#include "exahype2/fv/PatchUtils.h"
#include "exahype2/fv/rusanov/Functors.h"
"""
        stateless_kernel_generated=False
        non_stateless_kernel_generated=False
        for (_, stateless, _, _, _) in self._configurations:
            if stateless:
                stateless_kernel_generated = True
            else:
                non_stateless_kernel_generated = True
        
        temporary_kernels_directory = os.path.join(temporary_output_directory, "kernels")
        kernels_directory = "kernels"

        os.makedirs(temporary_kernels_directory, exist_ok=True)
        file = open(os.path.join(temporary_kernels_directory, f"{self._solver.name}_initialConditions{dimensions}d.h"), "w")
        file.write(f"""#pragma once
{required_includes}\n""")
        file.close()
        
        if stateless_kernel_generated:
            file = open(os.path.join(temporary_kernels_directory, f"{self._solver.name}_initialConditions{dimensions}d.cpph"), "w")
            file.write(f"""#include "{self._solver.name}_initialConditions{dimensions}d.h"\n""")
            file.close()
        
        if non_stateless_kernel_generated:
            file = open(os.path.join(temporary_kernels_directory, f"{self._solver.name}_initialConditions{dimensions}d.cpp"), "w")
            file.write(f"""#include "{self._solver.name}_initialConditions{dimensions}d.h"\n""")
            file.close()

        external_variables = {
            dslhype.Argument(dslhype.Name("NV"), dslhype.TInteger()): self._solver.numerical_scheme.patch_size,
            dslhype.Argument(dslhype.Name("H"), dslhype.TInteger()): self._solver.numerical_scheme.overlap,
            dslhype.Argument(dslhype.Name("NumberOfUnknowns"), dslhype.TInteger()): self._solver.unknowns
        }

        function_map = {
            dslhype.Name("initPointWisely"): dslhype.TFunction("std::function<void(const tarch::la::Vector<Dimensions, double>& x, double* __restrict__ Q)>&", OrderedDict({
                dslhype.Name("x"): dslhype.TVector(dslhype.TFloat(), dimensions), 
                dslhype.Name("h"): dslhype.TVector(dslhype.TFloat(), dimensions), 
                dslhype.Name("I"): dslhype.TArray(dslhype.TFloat(), 1)}))
        }

        for (target, stateless, output_language, memory_layout, _) in self._configurations:
            dslhype.translate_dslhype_function(
                solver=self._solver,
                function = initalise_centre_of_patch,
                function_arguments=[dslhype.Argument(dslhype.Name("data"), dslhype.TArray(dslhype.TFloat(), 1)),
                                    dslhype.Argument(dslhype.Name("x"), dslhype.TVector(dslhype.TFloat(), dimensions)),
                                    dslhype.Argument(dslhype.Name("h"), dslhype.TVector(dslhype.TFloat(), dimensions))],
                dimensions=dimensions,
                function_map=function_map,
                external_variables=external_variables,
                pde_terms=self._solver.pde_terms,
                output_file_name=f"{self._solver.name}_initialConditions{dimensions}d",
                target=target,
                stateless=stateless,
                memory_layout=memory_layout,
                function_name=f"{self._solver.name}_initialiseCentreOfPatch",
                output_language=output_language,
                output_directory=temporary_output_directory,
                generate_timed_version=False
            )
            dslhype.translate_dslhype_function(
                solver=self._solver,
                function = initalise_full_patch,
                function_arguments=[dslhype.Argument(dslhype.Name("data"), dslhype.TArray(dslhype.TFloat(), 1)),
                                    dslhype.Argument(dslhype.Name("x"), dslhype.TVector(dslhype.TFloat(), dimensions)),
                                    dslhype.Argument(dslhype.Name("h"), dslhype.TVector(dslhype.TFloat(), dimensions))],
                dimensions=dimensions,
                function_map=function_map,
                external_variables=external_variables,
                pde_terms=self._solver.pde_terms,
                output_file_name=f"{self._solver.name}_initialConditions{dimensions}d",
                target=target,
                stateless=stateless,
                memory_layout=memory_layout,
                function_name=f"{self._solver.name}_initialiseFullPatch",
                output_language=output_language,
                output_directory=temporary_output_directory,
                generate_timed_version=False
            )
        
        if stateless_kernel_generated:
            file = open(os.path.join(temporary_kernels_directory, f"{self._solver.name}_initialConditions{dimensions}d.h"), "a")
            file.write(f"""#include "{self._solver.name}_initialConditions{dimensions}d.cpph" """)
            file.close()

        peano4_project.output.makefile.add_h_file(os.path.join(kernels_directory, f"{self._solver.name}_initialConditions{dimensions}d.h"))
        if stateless_kernel_generated:
            peano4_project.output.makefile.add_h_file(os.path.join(kernels_directory, f"{self._solver.name}_initialConditions{dimensions}d.cpph"))
        if non_stateless_kernel_generated:
            peano4_project.output.makefile.add_cpp_file(os.path.join(kernels_directory, f"{self._solver.name}_initialConditions{dimensions}d.cpp"), generated=True)

        file = open(os.path.join(temporary_output_directory, "kernels", f"{self._solver.name}_initialConditions.h"), "w")
        file.write(f"""#include "{self._solver.name}_initialConditions{dimensions}d.h" """)
        file.close()