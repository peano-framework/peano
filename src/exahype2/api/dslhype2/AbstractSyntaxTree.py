from __future__ import annotations
from abc import ABC, abstractmethod
from .Types import *
from .Expressions import *
from string import ascii_lowercase

class Function:
    def __init__(self, name: Name, namespace: list[str] = [], stateless: bool = False):
        self.name = name
        self.namespace = namespace
        self.arguments: list[Argument] = []
        self.body: list[Statement] = []
        self.stateless = stateless
        self.function_map: dict[Name, TFunction] = None

    def add_statements(self, new_statements: list[Statement]):
        self.body.extend(new_statements)

    def add_arguments(self, new_arguments: list[Argument]):
        self.arguments.extend(new_arguments)

    def print_declaration(self, printer, indent_level: int = 0):
        return printer.print_function_declaration(self, indent_level)
    
    def print_definition(self, printer, indent_level: int = 0):
        return printer.print_function_definition(self, indent_level)

    def print_declaration_with_timer(self, printer, indent_level: int = 0):
        return printer.print_function_declaration_with_timer(self, indent_level)
    
    def print_definition_with_timer(self, printer, indent_level: int = 0):
        return printer.print_function_definition_with_timer(self, indent_level)


class Argument:
    def __init__(self, name: Name, type: Type):
        self.name = name
        self.type = type

    def print(self, printer, indent_level: int = 0):
        return printer.print_argument(self, indent_level)


class Statement(ABC):
    @abstractmethod
    def print(self, printer, indent_level: int = 0) -> str:
        pass

class For(Statement):
    iteration_variable_names = list(ascii_lowercase)

    def __init__(self, iteration_start: Element | Variable, iteration_end: Element | Variable):
        if not isinstance(iteration_start.get_type(), TIndex) or not isinstance(iteration_end.get_type(), TIndex):
            raise Exception("Iteration ranges for For loops must use index types")

        self.iteration_start = iteration_start
        self.iteration_end = iteration_end
        self.statements: list[Statement] = []

        for variable_name in For.iteration_variable_names:
            if Name(variable_name) not in Variable.symbol_map:
                self.iteration_variable = Variable(Name(variable_name))
                Variable.symbol_map.update({self.iteration_variable.name: Index(-1)})
                break

    def end_scope(self):
        del Variable.symbol_map[self.iteration_variable.name]

    def add_statement(self, statement: Statement):
        self.statements.append(statement)
    
    def get_iteration_variable(self) -> Variable:
        return self.iteration_variable

    def print(self, printer, indent_level: int = 0):
        return printer.print_for(self, indent_level)

class Construction(Statement):
    def __init__(self, lhs: Variable, rhs: Expression):
        self.lhs = lhs
        self.rhs = rhs
        Variable.symbol_map.update({self.lhs.get_name(): self.rhs})

    def print(self, printer, indent_level: int = 0):
        return printer.print_construction(self, indent_level)

class Assignment(Statement):
    def __init__(self, lhs: AssignableExpression, rhs: Expression):
        self.lhs = lhs
        self.rhs = rhs

    def print(self, printer, indent_level: int = 0):
        return printer.print_assignment(self, indent_level)

class DataBlockConstructionFromElement(Statement):
    def __init__(self, lhs: Variable, rhs: DataBlock):
        self.lhs = lhs
        self.rhs = rhs
        Variable.symbol_map.update({self.lhs.get_name(): self.rhs})

        #self.memory_allocation = ArrayAllocation(self.lhs, rhs)
        #self.assignment = DataBlockAssignment(lhs, rhs.value)

    def print(self, printer, indent_level: int = 0):
        return printer.print_data_block_construction_from_element(self, indent_level)

class DataBlockConstructionFromExternal(Statement):
    def __init__(self, lhs: Variable, rhs: DataBlock):
        self.lhs = lhs
        self.rhs = rhs
        Variable.symbol_map.update({self.lhs.get_name(): self.rhs})

    def print(self, printer, indent_level: int = 0):
        return printer.print_data_block_construction_from_external(self, indent_level)


class DataBlockConstructionFromFunctionCall(Statement):
    def __init__(self, lhs: Variable, rhs: DataBlock):
        self.lhs = lhs
        self.rhs = rhs
        Variable.symbol_map.update({self.lhs.get_name(): self.rhs})
        assert(isinstance(rhs.value, FunctionCallStatement))

        self.allocation = ArrayAllocation(self.lhs, rhs)
        
        fors = [For(Index(0), self.rhs.iteration_range[-1][1] - self.rhs.iteration_range[-1][0])]
        output_data_block_indices = [fors[0].get_iteration_variable()]
        for interval in self.rhs.iteration_range[-2:0:-1]:
            fors.append(For(Index(0), interval[1] - interval[0]))
            output_data_block_indices.insert(0, fors[-1].get_iteration_variable())
            fors[-2].add_statement(fors[-1])
        
        indexed_arguments = []
        for argument in rhs.value.arguments:
            if isinstance(argument, DataBlock) and len(argument.iteration_range) > len(output_data_block_indices) or \
               isinstance(argument, Variable) and isinstance(argument.get_value(), DataBlock) and len(argument.get_value().iteration_range) > len(output_data_block_indices):
                indexed_arguments.append(argument.index_at(output_data_block_indices, get_address=True))
            else:
                indexed_arguments.append(argument.index_at(output_data_block_indices, get_address=False))
        indexed_arguments.append(self.lhs.index_at(output_data_block_indices, get_address=True))
        fors[-1].add_statement(FunctionCallStatement(rhs.value.name, indexed_arguments, rhs.value.type, rhs.value.stateless))
        self.initialisation = fors[0]
        
        for for_loop in fors:
            for_loop.end_scope()

    def print(self, printer, indent_level: int = 0):
        return printer.print_data_block_construction_from_function_call(self, indent_level)

class DataBlockConstructionFromOperation(Statement):
    def __init__(self, lhs: Variable, rhs: DataBlock):
        self.lhs = lhs
        self.rhs = rhs
        Variable.symbol_map.update({self.lhs.get_name(): self.rhs})
        assert(isinstance(rhs, DataBlockBinaryOperation))

        self.allocation = ArrayAllocation(self.lhs, rhs)
        self.assignment = DataBlockAssignment(lhs, rhs)
    
    def print(self, printer, indent_level: int = 0):
        return printer.print_data_block_construction_from_operation(self, indent_level)

"""!
Since this requires a reduction over a DataBlock, this evaluation is treated as a statement that assigns a value,
rather than an expression representing the value itself. Currently we assume that this is only used to evaluate 
maxEigenvalues, so no memory allocation is performed.
"""
class DataBlockUnaryMax(Statement):
    def __init__(self, lhs: Variable, rhs: Variable):
        self.lhs = lhs
        self.rhs = rhs
        data_block = self.rhs.get_value()
        assert(isinstance(data_block, DataBlock))
        self.loops = [For(Index(0), data_block.iteration_range[-1][1] - data_block.iteration_range[-1][0])]
        index = [self.loops[0].get_iteration_variable()]
        self.loops[0].add_statement(Assignment(self.lhs.index_at(index), Float(0.0)))

        for interval in data_block.iteration_range[-2::-1]:
            self.loops.append(For(Index(0), interval[1] - interval[0]))
            index.insert(0, self.loops[-1].get_iteration_variable())
            self.loops[-2].add_statement(self.loops[-1])
        
        self.loops[-1].add_statement(Assignment(self.lhs.index_at(index), ElementBinaryOperation(Operator.MAX, self.lhs.index_at(index), self.rhs.index_at(index))))

    def print(self, printer, indent_level: int = 0):
        return printer.print_data_block_unary_max(self, indent_level)

class FunctionCallStatement(Statement):
    def __init__(self, name: Name, arguments: list[Expression], type: TFunction, stateless: bool = False):
        self.name = name
        self.type = type
        self.arguments = arguments
        self.stateless = stateless
    
    def print(self, printer, indent_level: int = 0):
        return printer.print_function_call_statement(self, indent_level)

class DataBlockAssignment(Statement):
    def __init__(self, lhs: AssignableExpression, rhs: Expression):
        self.lhs = lhs
        self.rhs = rhs

        if not isinstance(lhs.get_type(), TDataBlock):
            raise Exception("LHS of DataBlock assignment must be a DataBlock")

        # Since lhs is an AssignableExpression of type TDataBlock, it must
        # either be a DataBlock Variable or a DataBlockSlice
        if isinstance(lhs, Variable):
            self.data_block = lhs.get_value()
        else:
            self.data_block = lhs
        assert(isinstance(self.data_block, DataBlock))

        rhs_type = rhs.get_type()
        if isinstance(rhs_type, TDataBlock):
            if rhs_type.number_of_dimensions > self.data_block.get_type().number_of_dimensions:
                raise Exception("RHS of data block assignment must have the same number of dimensions or less than the LHS")

        self.assignment = For(Index(0), self.data_block.iteration_range[-1][1] - self.data_block.iteration_range[-1][0])
        loops = [self.assignment]
        index = [self.assignment.get_iteration_variable()]
        

        # Iterate over the iteration range backwards, ignoring the last element which we have already considered
        
        for interval in self.data_block.iteration_range[-2::-1]:
            loops.append(For(Index(0), interval[1] - interval[0]))
            index.insert(0, loops[-1].get_iteration_variable())
            loops[-2].add_statement(loops[-1])

        loops[-1].add_statement(Assignment(self.lhs.index_at(index), self.rhs.index_at(index)))

        for loop in loops:
            loop.end_scope()

    def print(self, printer, indent_level: int = 0):
        return printer.print_data_block_assignment(self, indent_level)


class MemoryAllocation(Statement):
    def __init__(self, lhs: AssignableExpression, type: Type, allocation_size: Element | Variable):
        self.lhs = lhs
        self.type = type
        self.allocation_size = allocation_size

    def print(self, printer, indent_level: int = 0):
        return printer.print_memory_allocation(self, indent_level)

class MemoryDeallocation(Statement):
    def __init__(self, lhs: AssignableExpression):
        self.lhs = lhs

    def print(self, printer, indent_level: int = 0):
        return printer.print_memory_deallocation(self, indent_level)

class ArrayAllocation(Statement):
    array_allocations: list[ArrayAllocation] = []

    def __init__(self, variable: Variable, array: Array):
        self.variable = variable
        self.array = array

        if len(self.array.array_size) == 1:
            self.allocation = MemoryAllocation(self.variable, self.array.element_type, self.array.array_size[0])
        elif self.array.flatten:
            self.allocation = MemoryAllocation(self.variable, self.array.element_type, self.array.array_size[0] * self.array.array_size[1])
        else:
            self.use_1d_allocation = False
            outer_allocation_size = self.array.array_size[0]
            inner_allocation_size = self.array.array_size[1]

            self.outer_allocation = MemoryAllocation(self.variable, TArray(self.array.element_type, 1), outer_allocation_size)
            self.inner_allocation = For(Index(0), outer_allocation_size)
            self.inner_allocation.add_statement(MemoryAllocation(Subscript(self.variable, [self.inner_allocation.get_iteration_variable()]), self.array.element_type, inner_allocation_size))
            self.inner_allocation.end_scope()
        
        ArrayAllocation.array_allocations.append(self)
    

    def print(self, printer, indent_level: int = 0):
        return printer.print_array_allocation(self, indent_level)

class ArrayDeallocation(Statement):
    def __init__(self, array_allocation: ArrayAllocation):
        self.array = array_allocation.array
        if len(array_allocation.array.array_size) == 1 or array_allocation.array.flatten:
            self.deallocation = MemoryDeallocation(array_allocation.variable)
        else:
            self.inner_deallocation = For(Index(0), self.array.array_size[0])
            self.inner_deallocation.add_statement(MemoryDeallocation(Subscript(array_allocation.variable, [self.inner_deallocation.get_iteration_variable()])))
            self.inner_deallocation.end_scope()
            self.outer_deallocation = MemoryDeallocation(array_allocation.variable)
    
    def print(self, printer, indent_level: int = 0):
        return printer.print_array_deallocation(self, indent_level)

class Comment(Statement):
    def __init__(self, comment: str):
        self.comment = comment

    def print(self, printer, indent_level: int = 0):
        return printer.print_comment(self, indent_level)