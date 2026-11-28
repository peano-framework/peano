import exahype2.api.dslhype2 as dslhype
from exahype2.api import Solver, Target, OutputLanguage, MemoryLayout, PDETerm, MLIRPass, \
    IncludeMLIRFunctions, InlineFunctions, MoveMemRefAllocationsToStart, LowerAffineToScf
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
    function_arguments: list[dslhype.Argument],
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
    generate_timed_version: bool = True
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
        
    cpp_printer = dslhype.CPPPrinter(4)
    function_arguments = function_arguments.copy()


    if output_language == OutputLanguage.CPP:
        if target == Target.Accelerator:
            printer = dslhype.OMPTargetPrinter(4)
            function_arguments.insert(0, dslhype.Argument(dslhype.Name("deviceNumber"), dslhype.TInteger()))
        else:
            printer = cpp_printer
    
    elif output_language == OutputLanguage.MLIR:
        printer = dslhype.MLIRPrinter(4, internal_name=function_name + ("_stateless" if stateless else "") + "_" +memory_layout.name.lower())

    
    if not stateless and output_language == OutputLanguage.CPP:
        function_arguments.extend(reduced_functor_parameters)

    function_namespace = solver.numerical_scheme.kernel_namespace + generate_namespace(output_language, target, memory_layout)
    
    parser = dslhype.Parser()
    function = parser.parse(function, dimensions, function_id=function_name + ("Stateless" if stateless else ""), function_namespace=function_namespace, function_arguments=function_arguments, function_map=function_map, external_variables=external_variables, enabled_pde_terms=pde_terms, stateless=stateless)


    function_declaration = function.print_declaration(printer)
    function_definition = function.print_definition(printer)
    if generate_timed_version:
        function_declaration_with_timer = function.print_declaration_with_timer(printer)
        function_definition_with_timer = function.print_definition_with_timer(printer)

    if output_language == OutputLanguage.CPP:
        kernels_directory = os.path.join(output_directory, "kernels")
        os.makedirs(kernels_directory, exist_ok=True)
        file = open(os.path.join(kernels_directory, f"{output_file_name}.h"), "a")
        file.write(function_declaration)
        if generate_timed_version:
            file.write(function_declaration_with_timer)

        if stateless == True:
            file = open(os.path.join(kernels_directory, f"{output_file_name}.cpph"), "a")
        else:
            file = open(os.path.join(kernels_directory, f"{output_file_name}.cpp"), "a")

        file.write(function_definition)
        if generate_timed_version:
            file.write(function_definition_with_timer)
        file.close()

    elif output_language == OutputLanguage.MLIR:
        function_bridge = printer.print_function_bridge(function)

        kernels_directory = os.path.join(output_directory, "kernels")
        os.makedirs(kernels_directory, exist_ok=True)

        file = open(os.path.join(kernels_directory, f"{output_file_name}.h"), "a")
        file.write(function_declaration)
        if generate_timed_version:
            file.write(function_declaration_with_timer)
        file.close()

        file = open(os.path.join(kernels_directory, f"{output_file_name}_{memory_layout.name.lower()}_{target.name.lower()}.mlir"), "w")
        file.write(function_definition)
        file.close()

        file = open(os.path.join(kernels_directory, f"{output_file_name}.cpp"), "a")
        file.write(function_bridge)
        if generate_timed_version:
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

def construct_mlir_pipeline(target: Target, passes: list[MLIRPass]) -> str:

    xdsl_pass_map: dict[type[MLIRPass], bool] = {
        IncludeMLIRFunctions: True,
        InlineFunctions: False,
        MoveMemRefAllocationsToStart: True,
        LowerAffineToScf: False
    }
    enabled_pass_map: dict[type[MLIRPass], MLIRPass] = {}
    for mlir_pass in passes:
        enabled_pass_map[type(mlir_pass)] = mlir_pass

    optional_passes_in_order: list[MLIRPass] = []
    if MoveMemRefAllocationsToStart in enabled_pass_map:
        optional_passes_in_order.append(enabled_pass_map[MoveMemRefAllocationsToStart])
    if IncludeMLIRFunctions in enabled_pass_map:
        optional_passes_in_order.append(enabled_pass_map[IncludeMLIRFunctions])
    if InlineFunctions in enabled_pass_map:
        if IncludeMLIRFunctions not in enabled_pass_map:
            raise Exception("IncludeMLIRFunctions must be enabled to enable InlineFunctions")
        optional_passes_in_order.append(LowerAffineToScf())
        optional_passes_in_order.append(enabled_pass_map[InlineFunctions])

    # The passes also have to be organised into groups based on whether they require xdsl-opt or mlir-opt
    grouped_passes: list[tuple[bool, list[MLIRPass]]] = []
    current_group: list[MLIRPass] = []

    if len(optional_passes_in_order) == 0:
        if target == Target.Serial:
            return "mlir-opt --pass-pipeline=\"builtin.module(inline,canonicalize,cse,loop-invariant-code-motion,canonicalize,cse,loop-invariant-code-motion,cse,canonicalize,expand-strided-metadata,fold-memref-alias-ops,lower-affine,finalize-memref-to-llvm,loop-invariant-code-motion,canonicalize,cse,finalize-memref-to-llvm,convert-scf-to-cf,convert-openmp-to-llvm,convert-math-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts,canonicalize,cse)\" $<"
        elif target == Target.Accelerator:
            return "xdsl-opt -p 'include-mlir-functions{mlir-file=\"euler_clean.mlir\" filename-being-modified=\"$<\"}' $< | mlir-opt --pass-pipeline=\"builtin.module(lower-affine,inline)\" | dslhype-opt --pass-pipeline=\"builtin.module(canonicalize,cse,loop-invariant-code-motion,inline,expand-strided-metadata,fold-memref-alias-ops,symbol-dce,canonicalize,func.func(gpu-map-parallel-loops),convert-parallel-loops-to-gpu,fold-memref-alias-ops,lower-affine,gpu-kernel-outlining,func.func(gpu-async-region),canonicalize,convert-arith-to-llvm{index-bitwidth=64},convert-scf-to-cf,convert-cf-to-llvm{index-bitwidth=64},nvvm-attach-target{chip=sm_90},gpu.module(inline,convert-gpu-to-nvvm,canonicalize,cse),gpu-to-llvm,gpu-module-to-binary,finalize-memref-to-llvm{index-bitwidth=64},reconcile-unrealized-casts)\""
    else:
        currently_using_xdsl = xdsl_pass_map[type(optional_passes_in_order[0])]
        for optional_pass in optional_passes_in_order:
            if currently_using_xdsl != xdsl_pass_map[type(optional_pass)]:
                grouped_passes.append((currently_using_xdsl, current_group.copy()))
                current_group.clear()
                currently_using_xdsl = not currently_using_xdsl
            current_group.append(optional_pass)
        grouped_passes.append((currently_using_xdsl, current_group.copy()))

        if grouped_passes[0][0] == True:
            pipeline_command = f"xdsl-opt -p '{",".join([str(optional_pass) for optional_pass in grouped_passes[0][1]])}' $<"
        else:
            pipeline_command = f"mlir-opt --pass-pipeline=\"builtin.module({",".join([str(optional_pass) for optional_pass in grouped_passes[0][1]])})\" $<"

        for pass_group in grouped_passes[1:]:
            if pass_group[0] == True:
                pipeline_command = f"{pipeline_command} | xdsl-opt -p '{",".join([str(optional_pass) for optional_pass in pass_group[1]])}'"
            else:
                pipeline_command = f"{pipeline_command} | mlir-opt --pass-pipeline=\"builtin.module({",".join([str(optional_pass) for optional_pass in pass_group[1]])})\""
        return f"{pipeline_command} | mlir-opt --pass-pipeline=\"builtin.module(inline,canonicalize,cse,loop-invariant-code-motion,canonicalize,cse,loop-invariant-code-motion,cse,canonicalize,expand-strided-metadata,fold-memref-alias-ops,lower-affine,finalize-memref-to-llvm,loop-invariant-code-motion,canonicalize,cse,finalize-memref-to-llvm,convert-scf-to-cf,convert-openmp-to-llvm,convert-math-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts,canonicalize,cse)\" "