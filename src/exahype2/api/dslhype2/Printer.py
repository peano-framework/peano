from abc import ABC, abstractmethod
from .AbstractSyntaxTree import *

class Printer(ABC):
    def __init__(self, spaces_per_tab: int):
        self.spaces_per_tab = spaces_per_tab
    
    def get_indent(self, indent_level: int):
        return ' ' * indent_level * self.spaces_per_tab

    @abstractmethod
    def print_name(self, name: Name, indent_level: int = 0):
        pass

    @abstractmethod
    def print_function_declaration(self, function: Function, indent_level: int = 0) -> str:
        pass

    @abstractmethod
    def print_function_definition(self, function: Function, indent_level: int = 0) -> str:
        pass

    @abstractmethod
    def print_function_declaration_with_timer(self, function: Function, indent_level: int = 0) -> str:
        pass

    @abstractmethod
    def print_function_definition_with_timer(self, function: Function, indent_level: int = 0) -> str:
        pass
    
    @abstractmethod
    def print_array_allocation(self, array_allocation: ArrayAllocation, indent_level: int = 0):
        pass

    @abstractmethod
    def print_array_deallocation(self, array_allocation: ArrayDeallocation, indent_level: int = 0):
        pass

    @abstractmethod
    def print_memory_allocation(self, memory_allocation: MemoryAllocation, indent_level: int = 0):
        pass

    @abstractmethod
    def print_memory_deallocation(self, memory_deallocation: MemoryDeallocation, indent_level: int = 0):
        pass

    @abstractmethod
    def print_function_call_statement(self, function_call: FunctionCallStatement, indent_level: int = 0):
        pass

    @abstractmethod
    def print_function_call_expression(self, function_call: FunctionCallExpression, indent_level: int = 0):
        pass

    @abstractmethod
    def print_for(self, for_loop: For, indent_level: int = 0):
        pass

    @abstractmethod
    def print_construction(self, construction: Construction, indent_level: int = 0):
        pass

    @abstractmethod
    def print_assignment(self, assignment: Assignment, indent_level: int = 0):
        pass

    @abstractmethod
    def print_data_block_construction_from_element(self, data_block_construction_from_element: DataBlockConstructionFromElement, indent_level: int = 0):
        pass
    
    @abstractmethod
    def print_data_block_construction_from_external(self, data_block_construction_from_external: DataBlockConstructionFromExternal, indent_level: int = 0):
        pass
    
    @abstractmethod
    def print_data_block_construction_from_function_call(self, data_block_construction_from_function_call: DataBlockConstructionFromFunctionCall, indent_level: int = 0):
        pass

    @abstractmethod
    def print_data_block_construction_from_operation(self, data_block_construction_from_operation: DataBlockConstructionFromOperation, indent_level: int = 0):
        pass
    
    @abstractmethod
    def print_data_block_assignment(self, data_block_assignment: DataBlockAssignment, indent_level: int = 0):
        pass
    
    @abstractmethod
    def print_data_block_unary_max(self, data_block_unary_max: DataBlockUnaryMax, indent_level: int = 0):
        pass

    @abstractmethod
    def print_comment(self, comment: Comment, indent_level: int = 0):
        pass

    #################################
    # Expression Prints
    #################################
    @abstractmethod
    def print_variable(self, variable: Variable, indent_level: int = 0):
        pass

    @abstractmethod
    def print_integer(self, integer: Integer, indent_level: int = 0):
        pass

    @abstractmethod
    def print_index(self, index: Index, indent_level: int = 0):
        pass

    @abstractmethod
    def print_float(self, float: Float, indent_level: int = 0):
        pass

    @abstractmethod
    def print_subscript(self, subscript: Subscript, indent_level: int = 0):
        pass

    @abstractmethod
    def print_element_binary_operation(self, element_binary_operation: ElementBinaryOperation, indent_level: int = 0):
        pass

    @abstractmethod
    def print_element_unary_operation(self, element_unary_operation: ElementUnaryOperation, indent_level: int = 0):
        pass

    @abstractmethod
    def print_element_index(self, element_index: ElementIndex, indent_level: int = 0):
        pass

    @abstractmethod
    def print_vector(self, vector: Vector, indent_level: int = 0):
        pass

    #################################
    # Type Prints
    #################################
    @abstractmethod
    def print_integer_type(self, integer_type: TInteger, indent_level: int = 0):
        pass

    @abstractmethod
    def print_index_type(self, index_type: TIndex, indent_level: int = 0):
        pass

    @abstractmethod
    def print_float_type(self, float_type: TFloat, indent_level: int = 0):
        pass

    @abstractmethod
    def print_vector_type(self, vector_type: TVector, indent_level: int = 0):
        pass
    
    @abstractmethod
    def print_array_type(self, array_type: TArray, indent_level: int = 0):
        pass

    @abstractmethod
    def print_struct_type(self, struct_type: TStruct, indent_level: int = 0):
        pass

    @abstractmethod
    def print_function_type(self, function_type: TFunction, indent_level: int = 0):
        pass
