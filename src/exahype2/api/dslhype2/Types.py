from abc import ABC, abstractmethod
from collections import OrderedDict

"""!
A Name is just a wrapper around a string used for naming variables
and functions.
"""
class Name:
    def __init__(self, id: str):
        self.id = id

    def __eq__(self, rhs):
        return self.id == rhs.id

    def __lt__(self, rhs):
        return self.id < rhs.id

    def __hash__(self):
        return hash(self.id)
    
    def print(self, printer, indent_level: int = 0):
        return printer.print_name(self, indent_level)

"""!
This is the type system used in DSLHyPE. Every expression has a get_type()
method which returns a Type. A Type at the highest level can be a TElement, 
a TArray, where TArrays are built from TElements, or a TStruct. A TStruct is
used to represent external C++ structs and classes.

TElements are the basic types, TInteger, TIndex and TFloat are supported.
TFloat represents the equivalent to C++ doubles.
TIndex is used for representing sizes and indices. The distinction from TInteger
is unnecessary for C++ generation but is important for MLIR.

TArrays can be TDataBlock or TMatrix.
"""

class Type(ABC):
    @abstractmethod
    def print(self, printer, indent_level: int = 0):
        pass

# Types are split into TElements and TArrays, which are groups of TElements
class TElement(Type):
    pass

class TArray(Type):
    flatten_temporaries: bool = False

    def __init__(self, element_type: TElement, number_of_dimensions: int, flatten = False):
        self.element_type = element_type
        # The number of internal dimensions, so should be 1 or 2
        self.number_of_dimensions = number_of_dimensions
        self.flatten = flatten

    def print(self, printer, indent_level: int = 0):
        return printer.print_array_type(self, indent_level)

# Elements
class TInteger(TElement):
    def print(self, printer, indent_level: int = 0) -> str:
        return printer.print_integer_type(self, indent_level)
    
class TIndex(TInteger):
    def print(self, printer, indent_level: int = 0) -> str:
        return printer.print_index_type(self, indent_level)

class TFloat(TElement):
    def print(self, printer, indent_level: int = 0) -> str:
        return printer.print_float_type(self, indent_level)

"""!
A TVector represents small vectors (typically 2D or 3D) which can be used as elements
of a DataBlock.
"""
class TVector(TElement):
    def __init__(self, base_type: TElement, dimensions: int):
        self.base_type = base_type
        self.dimensions = dimensions

    def print(self, printer, indent_level: int = 0) -> str:
        return printer.print_vector_type(self, indent_level)

# Arrays
class TDataBlock(TArray):
    def __init__(self, number_of_dimensions: int, element_type: TElement, flatten_array: bool):
        if number_of_dimensions == 1:
            TArray.__init__(self, element_type, number_of_dimensions=1, flatten=flatten_array)
        else:
            TArray.__init__(self, element_type, number_of_dimensions=2, flatten=flatten_array)

# Struct type used for representing CellData arguments
class TStruct(Type):
    def __init__(self, id: str):
        self.name = Name(id)
        self.fields: OrderedDict[Name, Type] = OrderedDict()

    def add_field(self, field_id: str, type: Type):
        self.fields[Name(field_id)] = type

    def print(self, printer, indent_level: int = 0) -> str:
        return printer.print_struct_type(self, indent_level)

class TFunction(Type):
    def __init__(self, id: str, argument_types: OrderedDict[Name, Type] = OrderedDict()):
        self.name = Name(id)
        self.argument_types = argument_types

    def add_argument(self, name: Name, type: Type):
        self.argument_types.update({name: type})

    def print(self, printer, indent_level: int = 0) -> str:
        return printer.print_function_type(self, indent_level)