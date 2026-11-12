from __future__ import annotations
from exahype2.api.dslhype2.Types import *
from collections.abc import Sequence
from enum import Enum
from copy import copy, deepcopy

"""!
An expression is an object with a value and a type. Since DSLHyPE
operates predominantly over loops all expressions must provide an index_at
function. Types that do not change between loop iterations (such as Elements)
would return themselves whereas types that are being iterated over would
return Subscripts.
"""
class Expression(ABC):
    @abstractmethod
    def print(self, printer, indent_level: int = 0):
        pass
    
    @abstractmethod
    def get_type(self) -> Type:
        pass

    @abstractmethod
    def index_at(self, indices: Sequence[Element | Variable], get_address = False) -> Expression:
        pass

"""!
An Element is an object that can be used to build an Array.
It provides an index_at function that returns itselfsince it will commonly appear
in DataBlock operations.
It is also used in index calculations, so methods to switch between Index and
Integer types are provided. These exist to be used at parsing time since any parsed
integer will be assumed to be Integers over Indices.
"""
class Element(Expression):
    @abstractmethod
    def get_type(self) -> TElement:
        pass
    
    def index_at(self, indices: Sequence[Element | Variable], get_address = False) -> Expression:
        return self

    @abstractmethod
    def convert_integers_to_indices(self) -> Element:
        pass

    @abstractmethod
    def convert_indices_to_integers(self) -> Element:
        pass

    def __add__(self, rhs: Element | Variable) -> Element | Variable:
        if isinstance(self, Index) and isinstance(self.value, int) and isinstance(rhs, Index) and isinstance(rhs.value, int):
            return Index(self.value + rhs.value)
        if isinstance(self, Integer) and isinstance(self.value, int) and isinstance(rhs, Integer) and isinstance(rhs.value, int):
            return Integer(self.value + rhs.value)
        if isinstance(rhs, Integer) and rhs.value == 0:
            return self
        if isinstance(self, Integer) and self.value == 0:
            return rhs
        return ElementBinaryOperation(Operator.ADDITION, self, rhs)
    
    def __sub__(self, rhs: Element | Variable) -> Element | Variable:
        if isinstance(self, Index) and isinstance(self.value, int) and isinstance(rhs, Index) and isinstance(rhs.value, int):
            return Index(self.value - rhs.value)
        if isinstance(self, Integer) and isinstance(self.value, int) and isinstance(rhs, Integer) and isinstance(rhs.value, int):
            return Integer(self.value - rhs.value)
        if isinstance(rhs, Integer) and rhs.value == 0:
            return self
        if isinstance(self, Integer) and self.value == 0:
            return -rhs
        if isinstance(rhs, ElementUnaryOperation):
            return self + rhs.operand
        return ElementBinaryOperation(Operator.SUBTRACTION, self, rhs)

    def __neg__(self) -> Element | Variable:
        if isinstance(self, Index) and isinstance(self.value, int):
            return Index(-self.value)
        if isinstance(self, Integer) and isinstance(self.value, int):
            return Integer(-self.value)
        return ElementUnaryOperation(Operator.SUBTRACTION, self)
    
    def __mul__(self, rhs: Element | Variable) -> Element | Variable:
        if isinstance(self, Index) and isinstance(self.value, int) and isinstance(rhs, Index) and isinstance(rhs.value, int):
            return Index(self.value * rhs.value)
        if isinstance(self, Integer) and isinstance(self.value, int) and isinstance(rhs, Integer) and isinstance(rhs.value, int):
            return Integer(self.value * rhs.value)
        if isinstance(rhs, Integer) and rhs.value == 1:
            return self
        if isinstance(self, Integer) and self.value == 1:
            return rhs
        return ElementBinaryOperation(Operator.MULTIPLICATION, self, rhs)

"""!
An Array is an abstraction over DataBlocks and Matrices.
"""
class Array(Expression):
    def __init__(self, element_type: TElement, size: Sequence[Element | Variable], flatten: bool):
        self.array_size = size
        assert(len(self.array_size) == 1 or len(self.array_size) == 2)
        self.element_type = element_type
        self.flatten = flatten | TArray.always_flatten

    @abstractmethod
    def get_type(self) -> TArray:
        pass

    @abstractmethod
    def calculate_indices(self, indices: Sequence[Element | Variable]) -> list[Element|Variable]:
        pass

    @staticmethod
    def linearise_index(indices: Sequence[Element | Variable], dimensions: Sequence[Element | Variable]) -> Element | Variable:
        scaling = Index(1)
        if len(indices) < len(dimensions):
            for length in dimensions[0:-len(indices)]:
                scaling = scaling * length
        
        # Extra indices should be ignored
        #indices_subset = indices[0:len(dimensions)]
        indices_subset = indices[-len(dimensions):]
        dimensions_subset = dimensions[-len(indices):]
        linearised_index = Index(0)
        factor = Index(1)

        for i, index in enumerate(indices_subset):
            interval = dimensions_subset[i]
            if isinstance(interval, Index) and interval.value == 1:
                # Subtract the iteration variable from the index so we only have an offset
                # The index will either be a variable, or binary operations where the leftmost node is a variable
                if isinstance(index, Variable):
                    variable = index
                else:
                    node = index
                    assert(isinstance(node, ElementBinaryOperation))
                    while isinstance(node, ElementBinaryOperation):
                        node = node.lhs
                    assert(isinstance(node, Variable))
                    variable = node
                linearised_index = linearised_index + factor * (index - variable)
            else:
                linearised_index = linearised_index + factor * index
            factor = factor * dimensions_subset[i]
        return linearised_index * scaling

"""!
Expressions are by default not assignable, as they represent temporary values. For
example, you cannot assign a value to (6 + 2). AssignableExpressions are those
which can be assigned to, such as variables, subscripts and slices. They must refer
to some variable to modify, either directly or indirectly, so AssignableExpressions
must be able to retrieve the name of this variable.
"""
class AssignableExpression(Expression):
    @abstractmethod
    def get_name(self) -> Name:
        pass

    @abstractmethod
    def index_at(self, indices: Sequence[Element | Variable], get_address = False) -> AssignableExpression:
        pass

class Variable(AssignableExpression):
    symbol_map: dict[Name, Expression] = {}

    def __init__(self, name: Name):
        self.name = name

    def get_name(self) -> Name:
        return self.name
    
    def get_value(self) -> Expression:
        return Variable.symbol_map[self.name]

    def get_type(self) -> Type:
        if self.name in Variable.symbol_map:
            return Variable.symbol_map[self.name].get_type()
        raise Exception(f"Variable name {self.name.id} is not stored in Variable.symbol_map")

    def print(self, printer, indent_level: int = 0):
        return printer.print_variable(self, indent_level)
    
    def convert_integers_to_indices(self) -> Variable:
        value = self.get_value()
        assert(isinstance(value, Element))
        value = value.convert_integers_to_indices()
        Variable.symbol_map[self.name] = value
        return self

    def convert_indices_to_integers(self) -> Variable:
        value = self.get_value()
        assert(isinstance(value, Element))
        value = value.convert_indices_to_integers()
        Variable.symbol_map[self.name] = value
        return self
    
    def index_at(self, indices: Sequence[Element | Variable], get_address = False) -> AssignableExpression:
        if isinstance(self.get_type(), Element):
            return self

        array = self.get_value()
        assert (isinstance(array, Array))
        return Subscript(self, array.calculate_indices(indices), get_address)

    def __add__(self, rhs: Element | Variable) -> Variable | Element:
        if isinstance(self.get_type(), TElement) and isinstance(rhs.get_type(), TElement):
            if isinstance(rhs, Integer) and rhs.value == 0:
                return self
            return ElementBinaryOperation(Operator.ADDITION, self, rhs)
        raise Exception("DataBlockBinaryOperation is not supported yet")
    
    def __sub__(self, rhs: Element | Variable) -> Variable | Element:
        if isinstance(self.get_type(), TElement) and isinstance(rhs.get_type(), TElement):
            if isinstance(rhs, Integer) and rhs.value == 0:
                return self
            if isinstance(self, Integer) and self.value == 0:
                return -rhs
            if isinstance(rhs, ElementUnaryOperation) and rhs.operator == Operator.SUBTRACTION:
                return ElementBinaryOperation(Operator.ADDITION, self, rhs.operand)
            return ElementBinaryOperation(Operator.SUBTRACTION, self, rhs)
        raise Exception("DataBlockBinaryOperation is not supported yet")

    def __neg__(self) -> Element:
        if isinstance(self.get_type(), TElement):
            return ElementUnaryOperation(Operator.SUBTRACTION, self)
        raise Exception("DataBlockUnaryOperation is not supported yet")

    def __mul__(self, rhs: Element | Variable) -> Variable | Element:
        if isinstance(self.get_type(), TElement) and isinstance(rhs.get_type(), TElement):
            if isinstance(rhs, Integer) and rhs.value == 1:
                return self
            return ElementBinaryOperation(Operator.MULTIPLICATION, self, rhs)
        raise Exception("DataBlockBinaryOperation is not supported yet")
    
    def __div__(self, rhs: Element | Variable) -> Element:
        if isinstance(self.get_type(), TElement) and isinstance(rhs.get_type(), TElement):
            if not (isinstance(self.get_type(), TFloat) and isinstance(rhs.get_type(), TFloat)):
                raise Exception("Division is only supported between expressions with type TFloat")
            return ElementBinaryOperation(Operator.DIVISION, self, rhs)
        raise Exception("DataBlockBinaryOperation is not supported yet")
    

# Elements
class Integer(Element):
    def get_type(self) -> TInteger:
        return TInteger()

    def __init__(self, value: int | External):
        self.value = value

    def convert_integers_to_indices(self) -> Index:
        return Index(self.value)

    def convert_indices_to_integers(self) -> Integer:
        return self
    
    def print(self, printer, indent_level: int = 0):
        return printer.print_integer(self, indent_level)

class Index(Integer):
    def get_type(self) -> TIndex:
        return TIndex()

    def convert_integers_to_indices(self) -> Index:
        return self

    def convert_indices_to_integers(self) -> Integer:
        return Integer(self.value)
    
    def print(self, printer, indent_level: int = 0):
        return printer.print_index(self, indent_level)
    
class Float(Element):
    def __init__(self, value: float | External):
        self.value = value

    def convert_integers_to_indices(self) -> Float:
        return self

    def convert_indices_to_integers(self) -> Float:
        return self
    
    def get_type(self) -> TFloat:
        return TFloat()

    def print(self, printer, indent_level: int = 0):
        return printer.print_float(self, indent_level)

class Operator(Enum):
    ADDITION = 1
    SUBTRACTION = 2
    MULTIPLICATION = 3
    DIVISION = 4
    MAX = 5

class ElementBinaryOperation(Element):
    def __init__(self, operator: Operator, lhs: Element | Variable | Subscript, rhs: Element | Variable | Subscript):
        if isinstance(lhs, Variable) and not isinstance(lhs.get_value(), Element) or \
           isinstance(rhs, Variable) and not isinstance(rhs.get_value(), Element):
            raise Exception("Variable operands of ElementBinaryOperation must be Elements")

        self.lhs = lhs
        self.rhs = rhs
        self.operator = operator

    def get_type(self) -> TElement:
        if isinstance(self.lhs.get_type(), TIndex) and isinstance(self.rhs.get_type(), TIndex):
            return TIndex()
        if isinstance(self.lhs.get_type(), TInteger) and isinstance(self.rhs.get_type(), TInteger):
            return TInteger()
        return TFloat()

    def convert_integers_to_indices(self) -> ElementBinaryOperation:
        return ElementBinaryOperation(self.operator, self.lhs.convert_integers_to_indices(), self.rhs.convert_integers_to_indices())

    def convert_indices_to_integers(self) -> ElementBinaryOperation:
        return ElementBinaryOperation(self.operator, self.lhs.convert_indices_to_integers(), self.rhs.convert_indices_to_integers())
    
    def print(self, printer, indent_level: int = 0):
        return printer.print_element_binary_operation(self, indent_level)


class ElementUnaryOperation(Element):
    def __init__(self, operator: Operator, operand: Element | Variable):
        self.operator = operator
        self.operand = operand

    def convert_integers_to_indices(self) -> ElementUnaryOperation:
        return ElementUnaryOperation(self.operator, self.operand.convert_integers_to_indices())

    def convert_indices_to_integers(self) -> ElementUnaryOperation:
        return ElementUnaryOperation(self.operator, self.operand.convert_indices_to_integers())
    
    def get_type(self) -> TElement:
        type = self.operand.get_type()
        assert(isinstance(type, TElement))
        return type

    def print(self, printer, indent_level: int = 0):
        return printer.print_element_unary_operation(self, indent_level) 

class ElementIndex(Element):
    def __init__(self, element: Element, index: Index):
        self.element = element
        self.index = index

    def convert_integers_to_indices(self) -> ElementUnaryOperation:
        pass
    
    def convert_indices_to_integers(self) -> ElementUnaryOperation:
        pass
        
    def get_type(self) -> TElement:
        pass

    def print(self, printer, indent_level: int = 0):
        return printer.print_element_index(self, indent_level)

class External(Expression):
    def __init__(self, reference: str):
        self.reference = reference

    def index_at(self, indices: Sequence[Element], get_address = False) -> None:
        raise Exception("External references should not be indexed")

    def get_type(self) -> None:
        raise Exception("External references have no type to obtain")

    def print(self, printer, indent_level: int = 0):
        raise Exception("External references should not have print called on them")

class ExternalStructField(External):
    def __init__(self, struct_name: Name, struct_type: TStruct, field_name: Name):
        self.struct_name = struct_name
        self.struct_type = struct_type
        self.field_name = field_name
        External.__init__(self, self.struct_name.id + "." + self.field_name.id)

    def get_type(self) -> Type:
        return self.struct_type.fields[self.field_name]

class DataBlock(Array):
    def __init__(self, iteration_range: Sequence[Sequence[Element | Variable]], value: Element | ExternalStructField | FunctionCall | None, flatten):
        self.value = value
        self.iteration_range = iteration_range

        size_in_memory: list[Element | Variable] = []
        for interval in self.iteration_range:
            size_in_memory.append(interval[1] - interval[0])
        self.size_in_memory = size_in_memory

        if isinstance(value, Element):
            element_type = value.get_type()
        elif isinstance(value, ExternalStructField):
            external_type = value.get_type()
            assert(isinstance(external_type, TArray))
            element_type = external_type.element_type
        else:
            element_type = TFloat()

        if len(self.size_in_memory) == 1:
            Array.__init__(self, element_type, size_in_memory, False)
        else:
            flattened_size = Index(1)
            for size in self.size_in_memory[0:-1]:
                flattened_size = flattened_size * size
            Array.__init__(self, element_type, [self.size_in_memory[-1], flattened_size], flatten=flatten)


    def calculate_indices(self, indices: Sequence[Element | Variable]) -> list[Element|Variable]:
        if len(self.iteration_range) == 1 or self.flatten:
            return [Array.linearise_index(indices, self.size_in_memory)]
        else:
            outer_index = indices[-1]
            inner_index = Array.linearise_index(indices[0:-1], self.size_in_memory[:-1])
            return [outer_index, inner_index]
    
    def get_type(self) -> TDataBlock:
        return TDataBlock(len(self.iteration_range), self.element_type, self.flatten)
    
    def print(self, printer, indent_level: int = 0):
        pass

    def index_at(self, indices: Sequence[Element | Variable], get_address = False) -> Expression:
        raise Exception("A DataBlock should not be indexed directly, only its slices or variables should be indexed.")

class DataBlockSlice(DataBlock, AssignableExpression):
    def __init__(self, original_data_block: Variable, iteration_range: list[list[Element | Variable]]):
        # A data block's value will not be used outside of an explicit constructor class, so any value can be set here
        DataBlock.__init__(self, iteration_range, None, original_data_block.get_value().flatten)
        self.original_data_block = original_data_block

    def index_at(self, indices: Sequence[Element | Variable], get_address = False) -> Expression:
        data_block = self.original_data_block.get_value()
        assert (isinstance(data_block, DataBlock) or isinstance(data_block, DataBlockSlice))
        
        offset_indices: list[Element | Variable] = []
        for i, index in enumerate(indices):
            offset_indices.append(index + self.iteration_range[i + len(self.iteration_range) - len(indices)][0] - data_block.iteration_range[i + len(data_block.iteration_range) - len(indices)][0])
        return self.original_data_block.index_at(offset_indices, get_address)
    
    def get_name(self) -> Name:
        return self.original_data_block.get_name()

    def get_type(self) -> TDataBlock:
        data_block = self.original_data_block.get_value()
        assert(isinstance(data_block, DataBlock))
        return TDataBlock(len(self.iteration_range), data_block.element_type, data_block.flatten)

    def print(self, printer, indent_level: int = 0):
        pass

class DataBlockBinaryOperation(DataBlock):
    def __init__(self, operator: Operator, lhs: Expression, rhs: Expression):
        self.operator = operator
        self.lhs = lhs
        self.rhs = rhs

        if isinstance(self.lhs.get_type(), TDataBlock) and isinstance(self.rhs.get_type(), TDataBlock):
            flatten = self.lhs.get_type().flatten or self.rhs.get_type().flatten
            if isinstance(self.lhs, Variable): 
                lhs_value = self.lhs.get_value()
                assert(isinstance(lhs_value, DataBlock))
                lhs_iteration_range = lhs_value.iteration_range
            else:
                assert(isinstance(self.lhs, DataBlock))
                lhs_iteration_range = self.lhs.iteration_range

            if isinstance(self.rhs, Variable): 
                rhs_value = self.rhs.get_value()
                assert(isinstance(rhs_value, DataBlock))
                rhs_iteration_range = rhs_value.iteration_range
            else:
                assert(isinstance(self.rhs, DataBlock))
                rhs_iteration_range = self.rhs.iteration_range

            # We start with the lhs iteration range, then expand it if the rhs is larger
            # We only check the first interval, since all others have to match
            iteration_range = deepcopy(lhs_iteration_range)
            if len(iteration_range) == 1:
                # If the DataBlock is 1D then the rhs must be the exact same dimensions or larger
                iteration_range = deepcopy(rhs_iteration_range)
            else:
                lhs_range = iteration_range[0][1] - iteration_range[0][0]
                rhs_range = rhs_iteration_range[0][1] - rhs_iteration_range[0][0]
                if isinstance(lhs_range, Index) and isinstance(rhs_range, Index):
                    assert(isinstance(lhs_range.value, int))
                    assert(isinstance(rhs_range.value, int))
                    if rhs_range.value > lhs_range.value:
                        iteration_range[0][0] = rhs_iteration_range[0][0]
                        iteration_range[0][1] = rhs_iteration_range[0][1]
                elif isinstance(lhs_range, Index) and isinstance(lhs_range.value, int) and lhs_range.value == 1:
                    # If the lhs interval is 1, it must be the smaller interval, even if the rhs cannot be evaluated at compile time
                    iteration_range[0][0] = rhs_iteration_range[0][0]
                    iteration_range[0][1] = rhs_iteration_range[0][1]

        elif type(self.lhs.get_type()) is TDataBlock:
            flatten = self.lhs.get_type().flatten
            if isinstance(self.lhs, Variable): 
                lhs_value = self.lhs.get_value()
                assert(isinstance(lhs_value, DataBlock))
                lhs_iteration_range = lhs_value.iteration_range
            else:
                assert(isinstance(self.lhs, DataBlock))
                lhs_iteration_range = self.lhs.iteration_range
            iteration_range = deepcopy(lhs_iteration_range)
        else:
            flatten = self.rhs.get_type().flatten
            if isinstance(self.rhs, Variable): 
                rhs_value = self.rhs.get_value()
                assert(isinstance(rhs_value, DataBlock))
                rhs_iteration_range = rhs_value.iteration_range
            else:
                assert(isinstance(self.rhs, DataBlock))
                rhs_iteration_range = self.rhs.iteration_range
            iteration_range = deepcopy(rhs_iteration_range)

        super().__init__(iteration_range, None, flatten)

    def index_at(self, indices: Sequence[Element | Variable], get_address = False) -> Expression:
        lhs = self.lhs.index_at(indices)
        rhs = self.rhs.index_at(indices)
        assert(isinstance(lhs, Element) or isinstance(lhs, Variable) or isinstance(lhs, Subscript))
        assert(isinstance(rhs, Element) or isinstance(rhs, Variable) or isinstance(rhs, Subscript))
        return ElementBinaryOperation(self.operator, lhs, rhs)

class DataBlockUnaryOperation(DataBlock):
    def __init__(self, operator: Operator, operand: Expression):
        assert(isinstance(operand.get_type(), TDataBlock))
        if isinstance(operand, Variable):
            iteration_range = operand.get_value().iteration_range
            flatten = operand.get_value().flatten
        else:
            iteration_range = operand.iteration_range
            flatten = operand.flatten

        self.operator = operator
        self.operand = operand
        super().__init__(iteration_range, None, flatten)


    def index_at(self, indices: Sequence[Element | Variable], get_address = False) -> Expression:
            element = self.operand.index_at(indices)
            assert(isinstance(element, Element) or isinstance(element, Variable) or isinstance(element, Subscript))
            return ElementUnaryOperation(self.operator, element)

class DataBlockIndex(DataBlock):
    def __init__(self, vector_data_block: Variable, index: Index):
        self.vector_data_block = vector_data_block
        self.index = index
        data_block_type = vector_data_block.get_type()
        assert(isinstance(data_block_type, TDataBlock))
        assert(isinstance(data_block_type.element_type, TVector))
        super().__init__(vector_data_block.get_value().iteration_range, None, data_block_type.flatten)

    def index_at(self, indices: Sequence[Element | Variable], get_address = False) -> Expression:
        return ElementIndex(self.vector_data_block.index_at(indices), self.index)
    
class Subscript(AssignableExpression):
    def __init__(self, base: AssignableExpression, indices: list[Element | Variable], get_address = False):
        assert(len(indices) == 1 or len(indices) == 2)
        for index in indices:
            if not isinstance(index.get_type(), TIndex):
                raise Exception("Index of subscript must be of index type")
        self.base = base
        self.indices = indices
        self.get_address = get_address

    def get_type(self) -> Type:
        return self.base.get_type()

    def get_name(self) -> Name:
        return self.base.get_name()

    def index_at(self, indices: Sequence[Element], get_address = False) -> Subscript:
        return self
    
    def print(self, printer, indent_level: int = 0):
        return printer.print_subscript(self, indent_level)
