import exahype2.api.dslhype as dslhype
from exahype2.api import Target
import os
from enum import Enum



def translate_dslhype_function(
    function,
    function_namespace,
    template_parameters, 
    functor_parameters,
    pde_terms,
    output_file_name,
    target: Target,
    stateless: bool,
    function_name: str = None,
    output_directory: str = ".",
):
    """!
    function_name: str
      Name of the generated C++ function. Defaults to the name of the
      Python DSL function passed in as function.

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

    reduced_functor_parameters = []
    for functor in functor_parameters:
        function_in_pde_terms = False
        for pde_term in pde_terms:
            if pde_term.NAME == functor.id:
                function_in_pde_terms = True
        if function_in_pde_terms:
            reduced_functor_parameters.append(functor)
    
    parser = dslhype.Parser()
    tree = parser.parse(
        function, 
        template_parameters, 
        reduced_functor_parameters, 
        pde_terms,
        namespace=function_namespace, 
        stateless=stateless,
        use_accelerator=(target==Target.Accelerator),
        function_name=function_name,
    )

    function_declaration = tree.print_declaration()
    if target == Target.Accelerator:
        function_definition = tree.print_omp()
    else:
        function_definition = tree.print_cpp()
    function_declaration_with_timer = tree.print_declaration_with_timer()
    function_definition_with_timer = tree.print_definition_with_timer()

    kernels_directory = os.path.join(output_directory, "kernels")
    os.makedirs(kernels_directory, exist_ok=True)
    file = open(os.path.join(kernels_directory, f"{output_file_name}.h"), "a")
    file.write(function_declaration)
    file.write(function_declaration_with_timer)
    file.close()

    file = open(os.path.join(kernels_directory, f"{output_file_name}.cpph"), "a")
    file.write(function_definition)
    file.write(function_definition_with_timer)
    file.close()

#    if target == Target.Serial:
#        file = open(f"{file_name}.h", "a")
#        file.write(function_declaration)
#        file.write(function_declaration_with_timer)
#        file.close()
#    elif target == Target.Multicore:
#        file = open(f"{file_name}.h", "a")
#        file.write("#if defined(SharedOMP)")
#        file.write(function_declaration)
#        file.write(function_declaration_with_timer)
#        file.write("#endif")
#        file.close()
#    else: # Target.Accelerator
#        file = open(f"{file_name}.h", "a")
#        file.write("#if defined(GPUOffloadingOMP)")
#        file.write(function_declaration)
#        file.write(function_declaration_with_timer)
#        file.write("#endif")
#        file.close()
#
#    if target == Target.Serial:
#        function_definition = tree.print_cpp()
#        file = open(f"{file_name}.cpph", "a")
#        file.write(function_definition)
#        file.write(function_definition_with_timer)
#        file.close()
#    elif target == Target.Multicore:
#        function_definition = tree.print_omp()
#        file = open(f"{file_name}.cpph", "a")
#        file.write("#if defined(SharedOMP)")
#        file.write(function_definition)
#        file.write(function_definition_with_timer)
#        file.write("#endif")
#        file.close()
#    else: # Target.Accelerator
#        function_definition = tree.print_omp()
#        file = open(f"{file_name}.cpph", "a")
#        file.write("#if defined(GPUOffloadingOMP)")
#        file.write(function_definition)
#        file.write(function_definition_with_timer)
#        file.write("#endif")
#        file.close()
