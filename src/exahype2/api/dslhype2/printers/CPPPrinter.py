from ..Printer import Printer
from ..AbstractSyntaxTree import *
import os

class CPPPrinter(Printer):
    def __init__(self, spaces_per_tab: int):
        Printer.__init__(self, spaces_per_tab)

    def print_name(self, name: Name, indent_level: int = 0) -> str:
        return f"""{self.get_indent(indent_level)}{name.id}"""
    
    def print_function_declaration(self, function: Function, indent_level: int = 0) -> str:
        argument_prints = [f"{argument.type.print(self)} {argument.name.print(self)}" for argument in function.arguments]
        return f"""
{self.get_indent(indent_level)}namespace {"::".join(function.namespace)} {{
{self.get_indent(indent_level + 1)}{"template <class Solver>" if function.stateless else ""}
{self.get_indent(indent_level + 1)}void {function.name.print(self)}({", ".join(argument_prints)});
{self.get_indent(indent_level)}}}
"""

    def print_function_definition(self, function: Function, indent_level: int = 0) -> str:
        argument_prints = [f"{argument.type.print(self)} {argument.name.print(self)}" for argument in function.arguments]
        statement_prints = [statement.print(self, indent_level + 2) for statement in function.body]
        return f"""
{self.get_indent(indent_level)}namespace {"::".join(function.namespace)} {{
{self.get_indent(indent_level + 1)}{"template <class Solver>" if function.stateless else ""}
{self.get_indent(indent_level + 1)}void {function.name.print(self)}({", ".join(argument_prints)}) {{
{os.linesep.join(statement_prints)}
{self.get_indent(indent_level + 1)}}}
{self.get_indent(indent_level)}}}
"""

    def print_function_declaration_with_timer(self, function: Function, indent_level: int = 0) -> str:
        argument_prints = [f"{argument.type.print(self)} {argument.name.print(self)}" for argument in function.arguments]
        return f"""
{self.get_indent(indent_level)}namespace {"::".join(function.namespace)} {{
{self.get_indent(indent_level + 1)}{"template <class Solver>" if function.stateless else ""}
{self.get_indent(indent_level + 1)}void {function.name.print(self)}({", ".join(argument_prints)}, tarch::timing::Measurement& measurement);
{self.get_indent(indent_level)}}}
"""

    def print_function_definition_with_timer(self, function: Function, indent_level: int = 0) -> str:
        parameter_prints = [f"{argument.type.print(self)} {argument.name.print(self)}" for argument in function.arguments]
        argument_prints = [argument.name.print(self) for argument in function.arguments]
        return f"""
{self.get_indent(indent_level)}namespace {"::".join(function.namespace)} {{
{self.get_indent(indent_level + 1)}{"template <class Solver>" if function.stateless else ""}
{self.get_indent(indent_level + 1)}void {function.name.print(self)}({", ".join(parameter_prints)}, tarch::timing::Measurement& measurement) {{
{self.get_indent(indent_level + 2)}tarch::timing::Watch watch("benchmarks::exahype2::euler::kernelbenchmarks::cpp", "EulerFVSmooth_timeStepWithRusanov", false, true);
{self.get_indent(indent_level + 2)}{function.name.print(self)}{"<Solver>" if function.stateless else ""}({",".join(argument_prints)});
{self.get_indent(indent_level + 2)}watch.stop();
{self.get_indent(indent_level + 2)}measurement.setValue(watch.getCalendarTime());
{self.get_indent(indent_level + 1)}}}
{self.get_indent(indent_level)}}}
"""

    def print_array_allocation(self, array_allocation: ArrayAllocation, indent_level: int = 0):
        if array_allocation.array.array_size == 1 or array_allocation.array.flatten:
            return array_allocation.allocation.print(self, indent_level)
        else:
            return f"""{array_allocation.outer_allocation.print(self, indent_level)}
{array_allocation.inner_allocation.print(self, indent_level)}"""

    def print_array_deallocation(self, array_deallocation: ArrayDeallocation, indent_level: int = 0):
        if array_deallocation.array.array_size == 1 or array_deallocation.array.flatten:
            return array_deallocation.deallocation.print(self, indent_level)
        else:
            return f"""{array_deallocation.inner_deallocation.print(self, indent_level)}
{array_deallocation.outer_deallocation.print(self, indent_level)}"""

    def print_memory_allocation(self, memory_allocation: MemoryAllocation, indent_level: int = 0) -> str:
        type_print = memory_allocation.type.print(self)
        lhs_print = memory_allocation.lhs.print(self)
        size_print = memory_allocation.allocation_size.print(self)

        if isinstance(memory_allocation.lhs, Variable):
            return f"""{self.get_indent(indent_level)}{type_print}* {lhs_print} = new {type_print}[{size_print}];"""
        else:
            # lhs is either a variable or a subscript, only variables require a type specifier
            return f"""{self.get_indent(indent_level)}{lhs_print} = new {type_print}[{size_print}];"""

    def print_memory_deallocation(self, memory_deallocation: MemoryDeallocation, indent_level: int = 0) -> str:
        lhs_print = memory_deallocation.lhs.print(self)
        return f"""{self.get_indent(indent_level)}delete[] {lhs_print};"""

    def print_function_call(self, function_call: FunctionCall, indent_level: int = 0):
        argument_prints = [argument.print(self) for argument in function_call.arguments]
        if function_call.stateless:
            return f"""{self.get_indent(indent_level)}Solver::{function_call.name.id}({", ".join(argument_prints)}, Solver::Offloadable::Yes);"""
        else:
            return f"""{self.get_indent(indent_level)}{function_call.name.id}({", ".join(argument_prints)});"""
         
    def print_for(self, for_loop: For, indent_level: int = 0) -> str:
        index_print = for_loop.iteration_variable.print(self)
        start_print = for_loop.iteration_start.print(self)
        end_print = for_loop.iteration_end.print(self)

        statement_prints = [statement.print(self, indent_level + 1) for statement in for_loop.statements]

        return f"""{self.get_indent(indent_level)}for (int {index_print} = {start_print}; {index_print} < {end_print}; {index_print}++) {{
{os.linesep.join(statement_prints)}
{self.get_indent(indent_level)}}}"""

    def print_construction(self, construction: Construction, indent_level: int = 0) -> str:
        type_print = construction.lhs.get_type().print(self)
        lhs_print = construction.lhs.print(self)
        rhs_print = construction.rhs.print(self)

        return f"""{self.get_indent(indent_level)}{type_print} {lhs_print} = {rhs_print};"""

    def print_assignment(self, assignment: Assignment, indent_level: int = 0) -> str:
        lhs_print = assignment.lhs.print(self)
        rhs_print = assignment.rhs.print(self)

        return f"""{self.get_indent(indent_level)}{lhs_print} = {rhs_print};"""

    def print_data_block_construction_from_element(self, data_block_construction_from_element: DataBlockConstructionFromElement, indent_level: int = 0) -> str:
        pass
    
    def print_data_block_construction_from_external(self, data_block_construction_from_external: DataBlockConstructionFromExternal, indent_level: int = 0) -> str:
        type_print = data_block_construction_from_external.lhs.get_type().print(self)
        lhs_print = data_block_construction_from_external.lhs.print(self)
        assert(isinstance(data_block_construction_from_external.rhs.value, External))
        rhs_print = data_block_construction_from_external.rhs.value.reference

        return f"""{self.get_indent(indent_level)}{type_print} {lhs_print} = {rhs_print};"""

    def print_data_block_construction_from_function_call(self, data_block_construction_from_function_call: DataBlockConstructionFromFunctionCall, indent_level: int = 0):
        return f"""{data_block_construction_from_function_call.allocation.print(self, indent_level)}
{data_block_construction_from_function_call.initialisation.print(self, indent_level)}"""

    def print_data_block_construction_from_operation(self, data_block_construction_from_operation: DataBlockConstructionFromOperation, indent_level: int = 0):
        return f"""{data_block_construction_from_operation.allocation.print(self, indent_level)}
{data_block_construction_from_operation.assignment.print(self, indent_level)}"""
    
    def print_data_block_assignment(self, data_block_assignment: DataBlockAssignment, indent_level: int = 0) -> str:
        return data_block_assignment.assignment.print(self, indent_level)
    
    def print_data_block_unary_max(self, data_block_unary_max: DataBlockUnaryMax, indent_level: int = 0) -> str:
        return data_block_unary_max.loops[0].print(self, indent_level)

    def print_comment(self, comment: Comment, indent_level: int = 0):
        return f"""{self.get_indent(indent_level)}//{comment.comment}"""
    
    #################################
    # Expression Prints
    #################################
    def print_variable(self, variable: Variable, indent_level: int = 0) -> str:
        return f"""{self.get_indent(indent_level)}{variable.name.print(self)}"""

    def print_integer(self, integer: Integer, indent_level: int = 0) -> str:
        if isinstance(integer.value, int):
            return f"""{self.get_indent(indent_level)}{integer.value}"""
        else:
            assert(isinstance(integer.value, External))
            return f"""{self.get_indent(indent_level)}{integer.value.reference}"""

    def print_index(self, index: Index, indent_level: int = 0) -> str:
        if isinstance(index.value, int):
            return f"""{self.get_indent(indent_level)}{index.value}"""
        else:
            assert(isinstance(index.value, External))
            return f"""{self.get_indent(indent_level)}{index.value.reference}"""

    def print_float(self, float_val: Float, indent_level: int = 0) -> str:
        if isinstance(float_val.value, float):
            return f"""{self.get_indent(indent_level)}{float_val.value}"""
        else:
            assert(isinstance(float_val.value, External))
            return f"""{self.get_indent(indent_level)}{float_val.value.reference}"""

    def print_subscript(self, subscript: Subscript, indent_level: int = 0) -> str:
        base_print = subscript.base.print(self)
        if subscript.flatten and len(subscript.indices) == 2:
            index_prints = [subscript.flattened_index.print(self)]
        else:
            index_prints = [index.print(self) for index in subscript.indices]
        
        if subscript.get_address:
            return f"""{self.get_indent(indent_level)}&{base_print}[{"][".join(index_prints)}]"""
        else:
            return f"""{self.get_indent(indent_level)}{base_print}[{"][".join(index_prints)}]"""

    def print_element_binary_operation(self, element_binary_operation: ElementBinaryOperation, indent_level: int = 0) -> str:
        match element_binary_operation.operator:
            case Operator.ADDITION:
                operator = "+"
            case Operator.SUBTRACTION:
                operator = "-"
            case Operator.MULTIPLICATION:
                operator = "*"
            case Operator.DIVISION:
                operator = "/"
        lhs_print = element_binary_operation.lhs.print(self)
        rhs_print = element_binary_operation.rhs.print(self)
        if element_binary_operation.operator == Operator.MAX:
            return f"""{self.get_indent(indent_level)}std::fmax({lhs_print}, {rhs_print})"""
        return f"""{self.get_indent(indent_level)}({lhs_print} {operator} {rhs_print})"""

    def print_element_unary_operation(self, element_unary_operation: ElementUnaryOperation, indent_level: int = 0):
        return f"""{self.get_indent(indent_level)}-{element_unary_operation.operand.print(self)}"""
    
    def print_element_index(self, element_index: ElementIndex, indent_level: int = 0):
        element_print = element_index.element.print(self)
        index_print = element_index.index.print(self)
        return f"""{self.get_indent(indent_level)}{element_print}({index_print})"""
    
    #################################
    # Type Prints
    #################################
    def print_integer_type(self, integer_type: TInteger, indent_level: int = 0) -> str:
        return f"""{self.get_indent(indent_level)}int"""
    
    def print_index_type(self, index_type: TIndex, indent_level: int = 0) -> str:
        return f"""{self.get_indent(indent_level)}int"""

    def print_float_type(self, float_type: TFloat, indent_level: int = 0) -> str:
        return f"""{self.get_indent(indent_level)}double"""

    def print_vector_type(self, vector_type: TVector, indent_level: int = 0):
        return f"""{self.get_indent(indent_level)}::tarch::la::Vector<{vector_type.dimensions}, {vector_type.base_type.print(self)}>"""
    
    def print_array_type(self, array_type: TArray, indent_level: int = 0) -> str:
        element_type_print = array_type.element_type.print(self)
        if array_type.number_of_dimensions == 1 or array_type.flatten:
            return f"""{self.get_indent(indent_level)}{element_type_print}*"""
        else:
            return f"""{self.get_indent(indent_level)}{element_type_print}**"""

    def print_struct_type(self, struct_type: TStruct, indent_level: int = 0) -> str:
        return f"""{self.get_indent(indent_level)}{struct_type.name.id}"""

    def print_function_type(self, function_type: TFunction, indent_level: int = 0) -> str:
        return f"""{self.get_indent(indent_level)}{function_type.name.id}"""