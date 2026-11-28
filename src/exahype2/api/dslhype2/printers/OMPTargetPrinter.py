from ..AbstractSyntaxTree import *
from .CPPPrinter import CPPPrinter
import os

class OMPTargetPrinter(CPPPrinter):
    def __init__(self, spaces_per_tab: int):
        CPPPrinter.__init__(self, spaces_per_tab)
        self.nested_level = 0

    def print_function_definition(self, function: Function, indent_level: int = 0) -> str:
        argument_prints = [f"{argument.type.print(self)} {argument.name.print(self)}" for argument in function.arguments]
        statement_prints = [statement.print(self, indent_level + 2) for statement in function.body]
        return f"""
#if defined(OMPUnifiedSharedMemory)
#pragma omp requires unified_shared_memory
#endif

{self.get_indent(indent_level)}namespace {"::".join(function.namespace)} {{
{self.get_indent(indent_level + 1)}{"template <class Solver>" if function.stateless else ""}
{self.get_indent(indent_level + 1)}void {function.name.print(self)}({", ".join(argument_prints)}) {{
{os.linesep.join(statement_prints)}
{self.get_indent(indent_level + 1)}}}
{self.get_indent(indent_level)}}}
"""

    def print_data_block_assignment(self, data_block_assignment: DataBlockAssignment, indent_level: int = 0) -> str:
        return f"""{self.get_indent(indent_level)}#pragma omp target teams loop collapse(Dimensions + 1) device(deviceNumber)
{data_block_assignment.assignment.print(self, indent_level)}"""

    def print_data_block_construction_from_function_call(self, data_block_construction_from_function_call: DataBlockConstructionFromFunctionCall, indent_level: int = 0):
        return f"""{data_block_construction_from_function_call.allocation.print(self, indent_level)}
{self.get_indent(indent_level)}#pragma omp target teams loop collapse(Dimensions) device(deviceNumber)
{data_block_construction_from_function_call.initialisation.print(self, indent_level)}"""

    # A unary max evaluation is a reduction, so it requires a little more care in parallelising it
    def print_data_block_unary_max(self, data_block_unary_max: DataBlockUnaryMax, indent_level: int = 0) -> str:
        outer_loop_index_print = data_block_unary_max.loops[0].iteration_variable.print(self)
        outer_loop_start_print = data_block_unary_max.loops[0].iteration_start.print(self)
        outer_loop_end_print = data_block_unary_max.loops[0].iteration_end.print(self)

        return f"""{self.get_indent(indent_level)}for (int {outer_loop_index_print} = {outer_loop_start_print}; {outer_loop_index_print} < {outer_loop_end_print}; {outer_loop_index_print}++) {{
#pragma omp target teams loop collapse(Dimensions + 1) reduction(max:{data_block_unary_max.lhs.index_at([data_block_unary_max.loops[0].get_iteration_variable()]).print(self)})
{data_block_unary_max.loops[1].print(self, indent_level + 1)}
{self.get_indent(indent_level)}}}"""

    def print_memory_allocation(self, memory_allocation: MemoryAllocation, indent_level: int = 0) -> str:
        type_print = memory_allocation.type.print(self)
        lhs_print = memory_allocation.lhs.print(self)
        size_print = memory_allocation.allocation_size.print(self)

        if isinstance(memory_allocation.lhs, Variable):
            return f"""{self.get_indent(indent_level)}{type_print}* {lhs_print} = tarch::accelerator::GPUMemoryManager::getInstance().allocate<{type_print}>({size_print}, deviceNumber);"""
        else:
            # lhs is either a variable or a subscript, only variables require a type specifier
            # In general a GPU code will always flatten the arrays, so this branch will not be reached
            return f"""{self.get_indent(indent_level)}{lhs_print} = tarch::accelerator::GPUMemoryManager::getInstance().allocate<{type_print}>({size_print}, deviceNumber);"""

    def print_memory_deallocation(self, memory_deallocation: MemoryDeallocation, indent_level: int = 0) -> str:
        lhs_print = memory_deallocation.lhs.print(self)
        return f"""tarch::accelerator::GPUMemoryManager::getInstance().free({lhs_print}, deviceNumber);"""
    