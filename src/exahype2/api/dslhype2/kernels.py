import exahype2.api.dslhype2 as dslhype
from exahype2.api import Solver, Target, OutputLanguage, MemoryLayout, PDETerm
import os
from enum import Enum

def generate_namespace(output_language: OutputLanguage, target: Target, memory_layout: MemoryLayout):
    if output_language == OutputLanguage.MLIR:
        function_namespace = ["mlir"]
    else:
        function_namespace = ["cpp"]

    if memory_layout == MemoryLayout.AoS_AllNested:
        function_namespace.append("nested")
    elif memory_layout == MemoryLayout.AoS_CellDataNested_TemporariesFlattened:
        function_namespace.append("cell_data_nested_temporaries_flattened")
    elif memory_layout == MemoryLayout.AoS_CellDataFlattened_TemporariesNested:
        function_namespace.append("cell_data_flattened_temporaries_nested")
    else:
        function_namespace.append("flattened")

    return function_namespace

def translate_dslhype_function(
    solver: Solver,
    function,
    dimensions: int,
    function_map: dict[dslhype.Name, dslhype.TFunction],
    external_variables, 
    pde_terms: list[PDETerm],
    output_file_name: str,
    target: Target,
    stateless: bool,
    output_language: OutputLanguage = OutputLanguage.CPP,
    memory_layout: MemoryLayout = MemoryLayout.AoS_AllNested,
    function_name: str | None = None,
    output_directory: str = ".",
):
    """!
    function_name: str
      Name of the generated C++ function. Defaults to the name of the
      Python DSL function passed in as function.

    output_language: OutputLanguage
      Language of the generated code. Defaults to C++, but can be MLIR.

    output_directory: str
      Directory the kernels/ subdirectory with the generated files is
      created in, usually the Peano project's directory.

    @remark Existing files are appended to, not overwritten
      Several functions of one project end up in the same files
      (kernels/{output_file_name}.h and .cpph/.cpp), so the translation appends.
      Running a generator script a second time in the same directory therefore
      duplicates the kernels, and the project does not compile
      ("redefinition of ..."). Remove the kernels/ directory before you
      regenerate.
      @todo The generators should write every function into a file of its
      own (and a thin header that includes them). Then each file can simply
      be overwritten, and regenerating in place becomes safe.
    """
    if target == Target.Accelerator and stateless == False:
        raise Exception("Accelerated kernels must be stateless")
    if target == Target.Accelerator and memory_layout != MemoryLayout.AoS_AllFlattened:
        raise Exception("Accelerated kernels must use flattened memory layout for all data")

    reduced_functor_parameters: list[dslhype.Argument] = []
    for functor_name, functor_type in function_map.items():
        function_in_pde_terms = False
        for pde_term in pde_terms:
            if pde_term.NAME == functor_name.id:
                function_in_pde_terms = True
        if function_in_pde_terms:
            reduced_functor_parameters.append(dslhype.Argument(functor_name, functor_type))

    if memory_layout == MemoryLayout.AoS_AllFlattened or memory_layout == MemoryLayout.AoS_CellDataNested_TemporariesFlattened:
        dslhype.TArray.flatten_temporaries = True
    else:
        dslhype.TArray.flatten_temporaries = False
    # Name the type according to the C++ type, since MLIR uses !llvm.ptr for structs
    # Add fields in the order specified by the MLIR type since only MLIR needs this information
    if output_language == OutputLanguage.MLIR:
        if memory_layout == MemoryLayout.AoS_CellDataFlattened_TemporariesNested or memory_layout == MemoryLayout.AoS_AllFlattened:
            cell_data_type = dslhype.TStruct("::exahype2::MLIRCellDataFlattened&")
        else:
            cell_data_type = dslhype.TStruct("::exahype2::MLIRCellDataNested&")
    elif target == Target.Accelerator:
        cell_data_type = dslhype.TStruct("::exahype2::CopyCellDataGPU<double, double>&")
    elif memory_layout == MemoryLayout.AoS_CellDataFlattened_TemporariesNested or memory_layout == MemoryLayout.AoS_AllFlattened:
        cell_data_type = dslhype.TStruct("::exahype2::CellDataFlattened<double, double>&")
    else:
        cell_data_type = dslhype.TStruct("::exahype2::CellData<double, double>&")

    if memory_layout == MemoryLayout.AoS_AllNested or memory_layout == MemoryLayout.AoS_CellDataNested_TemporariesFlattened:
        cell_data_type.add_field("QIn", dslhype.TDataBlock(2, dslhype.TFloat(), flatten_array=False))
        cell_data_type.add_field("QOut", dslhype.TDataBlock(2, dslhype.TFloat(), flatten_array=False))
    else:
        cell_data_type.add_field("QIn", dslhype.TDataBlock(2, dslhype.TFloat(), flatten_array=True))
        cell_data_type.add_field("QOut", dslhype.TDataBlock(2, dslhype.TFloat(), flatten_array=True))

    cell_data_type.add_field("cellCentre", dslhype.TDataBlock(1, dslhype.TVector(dslhype.TFloat(), dimensions), flatten_array=False))
    cell_data_type.add_field("cellSize", dslhype.TDataBlock(1, dslhype.TVector(dslhype.TFloat(), dimensions), flatten_array=False))
    cell_data_type.add_field("t", dslhype.TArray(dslhype.TFloat(), 1))
    cell_data_type.add_field("dt", dslhype.TArray(dslhype.TFloat(), 1))
    #cell_data_type.add_field("id", dslhype.TArray(dslhype.TInteger(), 1))
    cell_data_type.add_field("maxEigenvalue", dslhype.TArray(dslhype.TFloat(), 1))
    cell_data_type.add_field("numberOfCells", dslhype.TInteger())

    #nested_cpu_cell_data_type = dslhype.TStruct("::exahype2::CellData<double, double>&")
    #nested_cpu_cell_data_type.add_field("QIn", dslhype.TDataBlock(2, dslhype.TFloat(), flatten_array=False))
    #nested_cpu_cell_data_type.add_field("QOut", dslhype.TDataBlock(2, dslhype.TFloat(), flatten_array=False))
    #nested_cpu_cell_data_type.add_field("cellCentre", dslhype.TDataBlock(1, dslhype.TVector(dslhype.TFloat(), dimensions), flatten_array=False))
    #nested_cpu_cell_data_type.add_field("cellSize", dslhype.TDataBlock(1, dslhype.TVector(dslhype.TFloat(), dimensions), flatten_array=False))
    #nested_cpu_cell_data_type.add_field("t", dslhype.TArray(dslhype.TFloat(), 1))
    #nested_cpu_cell_data_type.add_field("dt", dslhype.TArray(dslhype.TFloat(), 1))
    #nested_cpu_cell_data_type.add_field("id", dslhype.TArray(dslhype.TInteger(), 1))
    #nested_cpu_cell_data_type.add_field("maxEigenvalue", dslhype.TArray(dslhype.TFloat(), 1))
    #nested_cpu_cell_data_type.add_field("numberOfCells", dslhype.TInteger())

    #gpu_cell_data_type = dslhype.TStruct("::exahype2::CopyCellDataGPU<double, double>&")
    #gpu_cell_data_type.add_field("QIn", dslhype.TDataBlock(2, dslhype.TFloat(), flatten_array=True))
    #gpu_cell_data_type.add_field("QOut", dslhype.TDataBlock(2, dslhype.TFloat(), flatten_array=True))
    #gpu_cell_data_type.add_field("cellCentre", dslhype.TDataBlock(1, dslhype.TVector(dslhype.TFloat(), dimensions), flatten_array=False))
    #gpu_cell_data_type.add_field("cellSize", dslhype.TDataBlock(1, dslhype.TVector(dslhype.TFloat(), dimensions), flatten_array=False))
    #gpu_cell_data_type.add_field("t", dslhype.TArray(dslhype.TFloat(), 1))
    #gpu_cell_data_type.add_field("dt", dslhype.TArray(dslhype.TFloat(), 1))
    #gpu_cell_data_type.add_field("id", dslhype.TArray(dslhype.TInteger(), 1))
    #gpu_cell_data_type.add_field("maxEigenvalue", dslhype.TArray(dslhype.TFloat(), 1))
    #gpu_cell_data_type.add_field("numberOfCells", dslhype.TInteger())

    #mlir_cell_data_type = dslhype.TStruct("::exahype2::MLIRCellDataFlattened&")
    #mlir_cell_data_type.add_field("QIn", dslhype.TDataBlock(2, dslhype.TFloat(), flatten_array=True))
    #mlir_cell_data_type.add_field("QOut", dslhype.TDataBlock(2, dslhype.TFloat(), flatten_array=True))
    #mlir_cell_data_type.add_field("cellCentre", dslhype.TDataBlock(1, dslhype.TVector(dslhype.TFloat(), dimensions), flatten_array=True))
    #mlir_cell_data_type.add_field("cellSize", dslhype.TDataBlock(1, dslhype.TVector(dslhype.TFloat(), dimensions), flatten_array=True))
    #mlir_cell_data_type.add_field("numberOfCells", dslhype.TInteger())

    cpp_printer = dslhype.CPPPrinter(4)


    if output_language == OutputLanguage.CPP:
        if target == Target.Accelerator:
            printer = dslhype.OMPTargetPrinter(4)
            #dslhype.TArray.flatten_temporaries = True
            arguments = [
                dslhype.Argument(dslhype.Name("deviceNumber"), dslhype.TInteger()),
                dslhype.Argument(dslhype.Name("patchData"), cell_data_type)
            ]
        else:
            printer = dslhype.CPPPrinter(4)
            #dslhype.TArray.flatten_temporaries = False
            arguments = [
                dslhype.Argument(dslhype.Name("patchData"), cell_data_type)
            ]
    elif output_language == OutputLanguage.MLIR:
        printer = dslhype.MLIRPrinter(4, internal_name=function_name + ("_stateless" if stateless else "") + "_" +memory_layout.name.lower())
        if target == Target.Accelerator:
            raise Exception("MLIR output is not supported for the accelerator target yet")
        else:
            arguments = [
                dslhype.Argument(dslhype.Name("patchData"), cell_data_type)
            ]

    
    if not stateless and output_language == OutputLanguage.CPP:
        arguments.extend(reduced_functor_parameters)

    function_namespace = solver.numerical_scheme.kernel_namespace + generate_namespace(output_language, target, memory_layout)
    
    parser = dslhype.Parser()
    function = parser.parse(function, function_id=function_name + ("Stateless" if stateless else ""), function_namespace=function_namespace, function_arguments=arguments, function_map=function_map, external_variables=external_variables, enabled_pde_terms=pde_terms, stateless=stateless)


    function_declaration = function.print_declaration(printer)
    function_definition = function.print_definition(printer)
    function_declaration_with_timer = function.print_declaration_with_timer(printer)
    function_definition_with_timer = function.print_definition_with_timer(printer)

    if output_language == OutputLanguage.CPP:
        kernels_directory = os.path.join(output_directory, "kernels")
        os.makedirs(kernels_directory, exist_ok=True)
        file = open(os.path.join(kernels_directory, f"{output_file_name}.h"), "a")
        file.write(function_declaration)
        file.write(function_declaration_with_timer)

        if stateless == True:
            file = open(os.path.join(kernels_directory, f"{output_file_name}.cpph"), "a")
        else:
            file = open(os.path.join(kernels_directory, f"{output_file_name}.cpp"), "a")

        file.write(function_definition)
        file.write(function_definition_with_timer)
        file.close()

    elif output_language == OutputLanguage.MLIR:
        function_bridge = printer.print_function_bridge(function)

        kernels_directory = os.path.join(output_directory, "kernels")
        os.makedirs(kernels_directory, exist_ok=True)

        file = open(os.path.join(kernels_directory, f"{output_file_name}.h"), "a")
        file.write(function_declaration)
        file.write(function_declaration_with_timer)
        file.close()

        file = open(os.path.join(kernels_directory, f"{output_file_name}_{memory_layout.name.lower()}.mlir"), "w")
        file.write(function_definition)
        file.close()

        file = open(os.path.join(kernels_directory, f"{output_file_name}.cpp"), "a")
        file.write(function_bridge)
        file.write(function_definition_with_timer)
        file.close()


def generate_mlir_function_call_bridges(solver: Solver, function_map: dict[dslhype.Name, dslhype.TFunction], pde_terms: list[PDETerm], stateless: bool, output_file_name: str, output_directory: str = "."):
        cpp_printer = dslhype.CPPPrinter(4)
        reduced_functor_parameters: list[dslhype.Argument] = []
        for functor_name, functor_type in function_map.items():
            function_in_pde_terms = False
            for pde_term in pde_terms:
                if pde_term.NAME == functor_name.id:
                    function_in_pde_terms = True
            if function_in_pde_terms:
                reduced_functor_parameters.append(dslhype.Argument(functor_name, functor_type))
        # Generate PDE function calls
        fully_qualified_solver_name = "::".join(solver.numerical_scheme.kernel_namespace + [f"Abstract{solver.name}"])

        term_definitions = """#include <cstdint>
extern "C" {
"""
        for pde_term_parameter in reduced_functor_parameters:
            pde_term_name = pde_term_parameter.name
            pde_term_type = pde_term_parameter.type
            assert(isinstance(pde_term_type, dslhype.TFunction))
            term_prints: list[str] = []
            type_casts = []
            for parameter_name, parameter_type in pde_term_type.argument_types.items():
                if isinstance(parameter_type, dslhype.TArray):
                    element_type = parameter_type.element_type.print(cpp_printer)
                    term_prints.extend([
                        f"{element_type}* {parameter_name.id}_allocated",
                        f"{element_type}* {parameter_name.id}_aligned",
                        f"std::int64_t {parameter_name.id}_offset",
                        f"std::int64_t {parameter_name.id}_size",
                        f"std::int64_t {parameter_name.id}_stride",
                    ])
                    type_casts.append(
                        f"{element_type}* {parameter_name.id} = "
                        f"{parameter_name.id}_aligned + {parameter_name.id}_offset;"
                    )
                elif isinstance(parameter_type, dslhype.TVector):
                    element_type = parameter_type.base_type.print(cpp_printer)
                    term_prints.extend([
                        f"{element_type}* {parameter_name.id}_allocated",
                        f"{element_type}* {parameter_name.id}_aligned",
                        f"std::int64_t {parameter_name.id}_offset",
                        f"std::int64_t {parameter_name.id}_size",
                        f"std::int64_t {parameter_name.id}_stride",
                    ])
                    type_casts.append(
                        f"tarch::la::Vector<{parameter_type.dimensions},{element_type}> "
                        f"{parameter_name.id} = "
                        f"{parameter_name.id}_aligned + {parameter_name.id}_offset;"
                    )
                else:
                    term_prints.append(f"{parameter_type.print(cpp_printer)} {parameter_name.id}")

            argument_prints = [argument.print(cpp_printer) for argument in pde_term_type.argument_types.keys()]

            if stateless:
                function_call = f"{fully_qualified_solver_name}::{pde_term_name.id}({", ".join(argument_prints)}, ::exahype2::Solver::Offloadable::Yes);"
            else:
                function_call = f"{"::".join(solver.numerical_scheme.kernel_namespace + ["repositories"])}::{solver.get_name_of_global_instance()}.{pde_term_name.id}({", ".join(argument_prints)});"
            term_definitions += f"""void {pde_term_name.id}{"Stateless" if stateless else ""}({", ".join(term_prints)}){{
{os.linesep.join(type_casts)}
{function_call}
}}
"""
        term_definitions += "}"

        kernels_directory = os.path.join(output_directory, "kernels")
        os.makedirs(kernels_directory, exist_ok=True)
        file = open(os.path.join(kernels_directory, f"{output_file_name}.cpp"), "a")
        file.write(term_definitions)
        file.close()