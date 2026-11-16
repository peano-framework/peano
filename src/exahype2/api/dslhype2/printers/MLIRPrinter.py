from ..Printer import Printer
from ..AbstractSyntaxTree import *
from .CPPPrinter import CPPPrinter
import os
import io
import xdsl.printer
from xdsl.dialects import builtin, func, llvm, arith, affine, memref
from xdsl.ir import Block, Region, SSAValue, Attribute
from xdsl.ir.affine import AffineMap, AffineDimExpr, AffineConstantExpr, AffineExpr, AffineSymExpr
from xdsl.builder import Builder
from xdsl.rewriter import InsertPoint

def get_xdsl_type(type: Type) -> Attribute:
    if isinstance(type, TInteger):
        return builtin.IntegerType(64)
    if isinstance(type, TIndex):
        return builtin.IndexType()
    if isinstance(type, TFloat):
        return builtin.Float64Type()

    if isinstance(type, TArray):
        if isinstance(type.element_type, TFloat):
            if type.number_of_dimensions == 1:
                return MLIRPrinter.xdsl_memref_descriptor_1d
            elif type.flatten:
                return MLIRPrinter.xdsl_memref_descriptor_2d
            else:
                return MLIRPrinter.xdsl_memref_descriptor_nested
        if isinstance(type.element_type, TVector) and isinstance(type.element_type.base_type, TFloat):
            return MLIRPrinter.xdsl_memref_descriptor_2d

    raise Exception("Type was not handled for xdsl conversion")

class MLIRPrinter(Printer):
    xdsl_memref_descriptor_1d = llvm.LLVMStructType(
        builtin.StringAttr(""),
        builtin.ArrayAttr([
            llvm.LLVMPointerType(),
            llvm.LLVMPointerType(),
            builtin.IntegerType(64),
            llvm.LLVMArrayType(1, builtin.IntegerType(64)),
            llvm.LLVMArrayType(1, builtin.IntegerType(64)),
        ]),
    )
    xdsl_memref_descriptor_2d = llvm.LLVMStructType(
        builtin.StringAttr(""),
        builtin.ArrayAttr([
            llvm.LLVMPointerType(),
            llvm.LLVMPointerType(),
            builtin.IntegerType(64),
            llvm.LLVMArrayType(2, builtin.IntegerType(64)),
            llvm.LLVMArrayType(2, builtin.IntegerType(64)),
        ]),
    )
    xdsl_memref_descriptor_nested = llvm.LLVMStructType(
        builtin.StringAttr(""),
        builtin.ArrayAttr([
            llvm.LLVMPointerType(),
            llvm.LLVMPointerType(),
            builtin.IntegerType(64),
            llvm.LLVMArrayType(1, builtin.IntegerType(64)),
            llvm.LLVMArrayType(1, builtin.IntegerType(64)),
        ]),
    )
    
    def __init__(self, spaces_per_tab: int, internal_name: str):
        self.builders: list[Builder] = []
        self.output_string_stream = io.StringIO()
        self.xdsl_printer = xdsl.printer.Printer(stream=self.output_string_stream, indent_num_spaces=spaces_per_tab)
        self.symbol_map: dict[Name, SSAValue] = {}
        self.struct_map: dict[Name, TStruct] = {}
        self.nested_affine_level = 0
        self.affine_dimension_map: dict[Name, int] = {}
        self.affine_symbol_map: dict[Name, int] = {}
        self.cpp_printer = CPPPrinter(spaces_per_tab)
        self.internal_name = internal_name
        Printer.__init__(self, spaces_per_tab)

    def print_name(self, name: Name, indent_level: int = 0) -> str:
        return f"""{self.get_indent(indent_level)}{name.id}"""
    
    def print_function_declaration(self, function: Function, indent_level: int = 0) -> str:
        argument_prints = [f"{argument.type.print(self.cpp_printer)} {argument.name.print(self.cpp_printer)}" for argument in function.arguments]
        return f"""
namespace {"::".join(function.namespace)} {{
    void {function.name.print(self.cpp_printer)}({", ".join(argument_prints)});
}}
"""
    
    def print_function_definition(self, function: Function, indent_level: int = 0) -> str:
        parameter_xdsl_types = []
        for argument in function.arguments:
            if isinstance(argument.type, TStruct):
                parameter_xdsl_types.append(llvm.LLVMPointerType())
                self.struct_map[argument.type.name] = argument.type
            elif isinstance(argument.type, TFunction):
                parameter_xdsl_types.append(argument.type.print(self))
        xdsl_function = func.FuncOp(self.internal_name, (tuple(parameter_xdsl_types), ()))
        self.builders.append(Builder(insertion_point=InsertPoint.at_start(xdsl_function.body.block)))
        
        for i, argument in enumerate(function.arguments):
            if isinstance(argument.type, TStruct) or isinstance(argument.type, TFunction):
                self.symbol_map[argument.name] = xdsl_function.args[i]

        for statement in function.body:
            statement.print(self)
        
        return_op = func.ReturnOp()
        self.builders[-1].insert(return_op)

        external_functions = []
        assert(function.function_map is not None)
        for name, external_function in function.function_map.items():
            external_parameter_types = []
            for argument_type in external_function.argument_types.values():
                if isinstance(argument_type, TArray):
                    element_type = argument_type.element_type.print(self)
                    assert(isinstance(element_type, Attribute))
                    external_parameter_types.append(
                        builtin.MemRefType(
                            element_type,
                            [builtin.DYNAMIC_INDEX],
                            builtin.StridedLayoutAttr([1], None),
                        )
                    )
                elif isinstance(argument_type, TVector):
                    element_type = argument_type.base_type.print(self)
                    assert(isinstance(element_type, Attribute))
                    external_parameter_types.append(
                        builtin.MemRefType(
                            element_type,
                            [builtin.DYNAMIC_INDEX],
                            builtin.StridedLayoutAttr([1], None),
                        )
                    )
                else:
                    external_parameter_types.append(argument_type.print(self))

            external_functions.append(
                func.FuncOp(
                    name=name.id + "Stateless",
                    region=Region(),
                    visibility=builtin.StringAttr("private"),
                    function_type=(tuple(external_parameter_types), ()),
                )
            )
        
        module_op = builtin.ModuleOp(external_functions + [xdsl_function])
        self.xdsl_printer.print_op(module_op)
        return self.output_string_stream.getvalue().lstrip()


    def print_function_declaration_with_timer(self, function: Function, indent_level: int = 0) -> str:
        argument_prints = [f"{argument.type.print(self.cpp_printer)} {argument.name.print(self.cpp_printer)}" for argument in function.arguments]
        return f"""
{self.get_indent(indent_level)}namespace {"::".join(function.namespace)} {{
{self.get_indent(indent_level + 1)}void {function.name.print(self.cpp_printer)}({", ".join(argument_prints)}, tarch::timing::Measurement& measurement);
{self.get_indent(indent_level)}}}
"""

    def print_function_definition_with_timer(self, function: Function, indent_level: int = 0) -> str:
        parameter_prints = [f"{argument.type.print(self.cpp_printer)} {argument.name.print(self.cpp_printer)}" for argument in function.arguments]
        argument_prints = [argument.name.print(self.cpp_printer) for argument in function.arguments]
        return f"""
{self.get_indent(indent_level)}namespace {"::".join(function.namespace)} {{
{self.get_indent(indent_level + 1)}void {function.name.print(self.cpp_printer)}({", ".join(parameter_prints)}, tarch::timing::Measurement& measurement) {{
{self.get_indent(indent_level + 2)}tarch::timing::Watch watch("benchmarks::exahype2::euler::kernelbenchmarks", "EulerFVSmooth_timeStepWithRusanov", false, true);
{self.get_indent(indent_level + 2)}{function.name.print(self.cpp_printer)}({",".join(argument_prints)});
{self.get_indent(indent_level + 2)}watch.stop();
{self.get_indent(indent_level + 2)}measurement.setValue(watch.getCalendarTime());
{self.get_indent(indent_level + 1)}}}
{self.get_indent(indent_level)}}}
"""
    
    def print_function_bridge(self, function: Function, indent_level: int = 0) -> str:
        parameter_prints = [f"{argument.type.print(self.cpp_printer)} {argument.name.print(self.cpp_printer)}" for argument in function.arguments]
        argument_prints = []
        for argument in function.arguments:
            if isinstance(argument.type, TStruct):
                argument_prints.append(f"&{argument.name.id}")
        return f"""
extern "C" void {self.internal_name}(void* patchData);
namespace {"::".join(function.namespace)} {{
    void {function.name.id}({", ".join(parameter_prints)}) {{
        ::{self.internal_name}({", ".join(argument_prints)});
    }}
}}
"""

    def print_array_allocation(self, array_allocation: ArrayAllocation, indent_level: int = 0) -> SSAValue:
        if array_allocation.array.array_size == 1:
            return array_allocation.allocation.print(self, indent_level)
        elif array_allocation.array.flatten:
            outer_size_print = array_allocation.array.array_size[0].print(self)
            inner_size_print = array_allocation.array.array_size[1].print(self)
            if outer_size_print.type == builtin.IntegerType(64):
                cast_op = arith.IndexCastOp(outer_size_print, builtin.IndexType())
                self.builders[-1].insert(cast_op)
                outer_size_print = cast_op.result
            if inner_size_print.type == builtin.IntegerType(64):
                cast_op = arith.IndexCastOp(inner_size_print, builtin.IndexType())
                self.builders[-1].insert(cast_op)
                inner_size_print = cast_op.result

            memref_type = array_allocation.array.get_type().print(self)
            allocation_op = memref.AllocOp(dynamic_sizes=[outer_size_print, inner_size_print], symbol_operands=[], result_type=memref_type)
            self.builders[-1].insert(allocation_op)
            return allocation_op.memref
        else:
            outer_allocation = array_allocation.outer_allocation.print(self, indent_level)
            self.symbol_map[array_allocation.variable.name] = outer_allocation
            array_allocation.inner_allocation.print(self, indent_level)
            return outer_allocation

    def print_array_deallocation(self, array_deallocation: ArrayDeallocation, indent_level: int = 0):
        if array_deallocation.array.array_size == 1 or array_deallocation.array.flatten:
            return array_deallocation.deallocation.print(self, indent_level)
        else:
            return f"""{array_deallocation.inner_deallocation.print(self, indent_level)}
{array_deallocation.outer_deallocation.print(self, indent_level)}"""
        
    def print_memory_allocation(self, memory_allocation: MemoryAllocation, indent_level: int = 0) -> SSAValue:
        type_print = memory_allocation.type.print(self)
        size_print = memory_allocation.allocation_size.print(self)
        assert(isinstance(type_print, Attribute))
        assert(isinstance(size_print, SSAValue))

        if size_print.type == builtin.IntegerType(64):
            cast_op = arith.IndexCastOp(size_print, builtin.IndexType())
            self.builders[-1].insert(cast_op)
            size_print = cast_op.result
        memref_type = builtin.MemRefType(element_type=type_print, shape=[builtin.DYNAMIC_INDEX])
        allocation_op = memref.AllocOp(dynamic_sizes=[size_print], symbol_operands=[], result_type=memref_type)
        self.builders[-1].insert(allocation_op)

        if isinstance(memory_allocation.lhs, Subscript):
            subscript = memory_allocation.lhs
            destination = subscript.base.print(self)
            self.affine_symbol_map = {}
            index_map = self.generate_affine_expression(subscript.indices[0])
            dim_values = [self.symbol_map[name] for name in self.affine_dimension_map]
            symbol_values = [self.symbol_map[name] for name in self.affine_symbol_map]
            store_map = AffineMap(
                num_dims=len(self.affine_dimension_map),
                num_symbols=len(self.affine_symbol_map),
                results=(index_map,),
            )
            store_op = affine.StoreOp(
                allocation_op.memref,
                destination,
                indices=dim_values + symbol_values,
                map=builtin.AffineMapAttr(store_map),
            )
            self.builders[-1].insert(store_op)

        return allocation_op.memref
        #if isinstance(memory_allocation.lhs, Variable):
        #    return f"""{self.get_indent(indent_level)}{type_print}* {lhs_print} = new {type_print}[{size_print}];"""
        #else:
        #    # lhs is either a variable or a subscript, only variables require a type specifier
        #    return f"""{self.get_indent(indent_level)}{lhs_print} = new {type_print}[{size_print}];"""

    def print_memory_deallocation(self, memory_deallocation: MemoryDeallocation, indent_level: int = 0) -> str:
        lhs_print = memory_deallocation.lhs.print(self)
        return f"""{self.get_indent(indent_level)}delete[] {lhs_print};"""
    
    def print_function_call(self, function_call: FunctionCall, indent_level: int = 0):
        argument_ssa_values: list[SSAValue] = []
        for argument in function_call.arguments:
            argument_ssa_values.append(argument.print(self))
            assert(isinstance(argument_ssa_values[-1], SSAValue))

        xdsl_function_call = func.CallOp(function_call.name.id + ("Stateless" if function_call.stateless else ""), argument_ssa_values, [])
        self.builders[-1].insert(xdsl_function_call)
    
    def print_for(self, for_loop: For, indent_level: int = 0):
        lower_bound_ssa_value = for_loop.iteration_start.print(self)
        upper_bound_ssa_value = for_loop.iteration_end.print(self)
        assert(isinstance(lower_bound_ssa_value, SSAValue))
        assert(isinstance(upper_bound_ssa_value, SSAValue))
        assert(self.builders[-1] is not None)
        if lower_bound_ssa_value.type == builtin.IntegerType(64):
            cast_op = arith.IndexCastOp(lower_bound_ssa_value, builtin.IndexType())
            lower_bound_ssa_value = cast_op.result
            self.builders[-1].insert(cast_op)
        if upper_bound_ssa_value.type == builtin.IntegerType(64):
            cast_op = arith.IndexCastOp(upper_bound_ssa_value, builtin.IndexType())
            upper_bound_ssa_value = cast_op.result
            self.builders[-1].insert(cast_op)
        
        body_block = Block(arg_types=[builtin.IndexType()])
        body_block.add_op(affine.YieldOp.get())
        map_identity = AffineMap(1, 0, (AffineDimExpr(0),))
        lower_bounds_map = builtin.AffineMapAttr(map_identity)
        upper_bounds_map = builtin.AffineMapAttr(map_identity)
        lower_bounds_groups = builtin.DenseIntElementsAttr.from_list(builtin.VectorType(builtin.IntegerType(32), [1]), [1])
        upper_bounds_groups = builtin.DenseIntElementsAttr.from_list(builtin.VectorType(builtin.IntegerType(32), [1]), [1])
        steps = builtin.ArrayAttr([builtin.IntegerAttr(1, builtin.IntegerType(64))])

        parallel_op = affine.ParallelOp.build(
            operands=[[lower_bound_ssa_value, upper_bound_ssa_value]],
            result_types=[[]],
            regions=[Region(body_block)],
            properties={
                "reductions": builtin.ArrayAttr([]),
                "lowerBoundsMap": lower_bounds_map,
                "lowerBoundsGroups": lower_bounds_groups,
                "upperBoundsMap": upper_bounds_map,
                "upperBoundsGroups": upper_bounds_groups,
                "steps": steps
            }
        )
        self.builders.append(Builder(insertion_point=InsertPoint.at_start(body_block)))

        self.symbol_map[for_loop.iteration_variable.name] = body_block.args[0]
        self.affine_dimension_map[for_loop.iteration_variable.name] = self.nested_affine_level
        self.nested_affine_level = self.nested_affine_level + 1

        for statement in for_loop.statements:
            statement.print(self)
        del self.symbol_map[for_loop.iteration_variable.name]
        del self.affine_dimension_map[for_loop.iteration_variable.name]
        self.nested_affine_level = self.nested_affine_level - 1
        self.builders.pop()
        self.builders[-1].insert(parallel_op)

    def print_construction(self, construction: Construction, indent_level: int = 0):
        assert(isinstance(construction.rhs, Integer) or isinstance(construction.rhs, Float))
        if isinstance(construction.rhs.value, ExternalStructField):
            struct_variable_ssa_value = self.symbol_map[construction.rhs.value.struct_name]
            field_name = construction.rhs.value.field_name
            struct_type = construction.rhs.value.struct_type
            fields = [field for field in struct_type.fields.keys()]
            field_index = fields.index(field_name)

            xdsl_struct_field_types = [get_xdsl_type(field_type) for field_type in struct_type.fields.values()]
            xdsl_struct_type = llvm.LLVMStructType(builtin.StringAttr(construction.rhs.value.struct_name.id), builtin.ArrayAttr(xdsl_struct_field_types))

            gep_op = llvm.GEPOp(ptr=struct_variable_ssa_value,
                                result_type=llvm.LLVMPointerType(),
                                indices=[0, field_index],
                                pointee_type=xdsl_struct_type)
            load_op = llvm.LoadOp(gep_op.result, result_type=builtin.IntegerType(64))
            self.symbol_map[construction.lhs.name] = load_op.results[0]

            assert(self.builders[-1] is not None)
            self.builders[-1].insert(gep_op)
            self.builders[-1].insert(load_op)

    def print_assignment(self, assignment: Assignment, indent_level: int = 0):
        #lhs_ssa_value = assignment.lhs.print(self)
        rhs_ssa_value = assignment.rhs.print(self)
        assert(isinstance(rhs_ssa_value, SSAValue))

        # Assignment in MLIR only makes sense when assigning into a memref, since otherwise it would
        # conflict with its SSA nature
        assert(isinstance(assignment.lhs, Subscript))
        memref_ssa_value = assignment.lhs.base.print(self)
        assert(isinstance(memref_ssa_value, SSAValue))
        assert(isinstance(memref_ssa_value.type, builtin.MemRefType))

        self.affine_symbol_map = {}
        outer_index_map = self.generate_affine_expression(assignment.lhs.indices[0])
        inner_index_map = self.generate_affine_expression(assignment.lhs.indices[1])

        dim_values = [self.symbol_map[name] for name in self.affine_dimension_map.keys()]
        symbol_values = [self.symbol_map[name] for name in self.affine_symbol_map.keys()]
        if assignment.lhs.flatten:
            map = AffineMap(num_dims=len(self.affine_dimension_map), num_symbols=len(self.affine_symbol_map), results=(outer_index_map, inner_index_map))
            store_op = affine.StoreOp(rhs_ssa_value, memref_ssa_value, indices=dim_values + symbol_values, map=builtin.AffineMapAttr(map))
            self.builders[-1].insert(store_op)
        else:
            outer_map = AffineMap(num_dims=len(self.affine_dimension_map), num_symbols=len(self.affine_symbol_map), results=(outer_index_map,))
            inner_map = AffineMap(num_dims=len(self.affine_dimension_map), num_symbols=len(self.affine_symbol_map), results=(inner_index_map,))
            load_op = affine.LoadOp(
                memref_ssa_value,
                indices=dim_values + symbol_values,
                map=builtin.AffineMapAttr(outer_map),
            )
            self.builders[-1].insert(load_op)
            store_op = affine.StoreOp(
                rhs_ssa_value,
                load_op.result,
                indices=dim_values + symbol_values,
                map=builtin.AffineMapAttr(inner_map),
            )
            self.builders[-1].insert(store_op)

    def print_data_block_construction_from_element(self, data_block_construction_from_element: DataBlockConstructionFromElement, indent_level: int = 0):
        pass
    
    def print_data_block_construction_from_external(self, data_block_construction_from_external: DataBlockConstructionFromExternal, indent_level: int = 0):
        assert(isinstance(data_block_construction_from_external.rhs.value, External))
        if isinstance(data_block_construction_from_external.rhs.value, ExternalStructField):
            struct_variable_ssa_value = self.symbol_map[data_block_construction_from_external.rhs.value.struct_name]
            field_name = data_block_construction_from_external.rhs.value.field_name
            struct_type = data_block_construction_from_external.rhs.value.struct_type
            fields = [field for field in struct_type.fields.keys()]
            field_index = fields.index(field_name)

            field_type = struct_type.fields[field_name]
            assert(isinstance(field_type, TArray))
            # Get a memref descriptor that describes the input field (which is an array)
            xdsl_memref_descriptor = get_xdsl_type(field_type)

            if field_type.flatten or field_type.number_of_dimensions == 1:
                memref_shape = [builtin.DYNAMIC_INDEX for _ in range(field_type.number_of_dimensions)]
                if isinstance(field_type.element_type, TVector):
                    memref_shape.append(builtin.DYNAMIC_INDEX)

                xdsl_struct_field_types = [get_xdsl_type(field_type) for field_type in struct_type.fields.values()]
                xdsl_struct_type = llvm.LLVMStructType(builtin.StringAttr(data_block_construction_from_external.rhs.value.struct_name.id), builtin.ArrayAttr(xdsl_struct_field_types))

                gep_op = llvm.GEPOp(ptr=struct_variable_ssa_value,
                                    result_type=llvm.LLVMPointerType(),
                                    indices=[0, field_index],
                                    pointee_type=xdsl_struct_type)

                descriptor_load_op = llvm.LoadOp(ptr=gep_op.result, result_type=get_xdsl_type(field_type))
                memref_cast_op = builtin.UnrealizedConversionCastOp.get(inputs=descriptor_load_op.results, result_type=[builtin.MemRefType(builtin.Float64Type(), memref_shape)])
                self.symbol_map[data_block_construction_from_external.lhs.name] = memref_cast_op.results[0]

                assert(self.builders[-1] is not None)
                self.builders[-1].insert(gep_op)
                self.builders[-1].insert(descriptor_load_op)
                self.builders[-1].insert(memref_cast_op)
            else:
                nested_memref_type = field_type.print(self)

                xdsl_struct_field_types = [
                    get_xdsl_type(field_type)
                    for field_type in struct_type.fields.values()
                ]
                xdsl_struct_type = llvm.LLVMStructType(
                    builtin.StringAttr(data_block_construction_from_external.rhs.value.struct_name.id),
                    builtin.ArrayAttr(xdsl_struct_field_types),
                )

                gep_op = llvm.GEPOp(
                    ptr=struct_variable_ssa_value,
                    result_type=llvm.LLVMPointerType(),
                    indices=[0, field_index],
                    pointee_type=xdsl_struct_type,
                )
                descriptor_load_op = llvm.LoadOp(
                    ptr=gep_op.result, result_type=xdsl_memref_descriptor
                )
                memref_cast_op = builtin.UnrealizedConversionCastOp.get(
                    inputs=descriptor_load_op.results,
                    result_type=[nested_memref_type],
                )
                self.symbol_map[data_block_construction_from_external.lhs.name] = (
                    memref_cast_op.results[0]
                )

                assert(self.builders[-1] is not None)
                self.builders[-1].insert(gep_op)
                self.builders[-1].insert(descriptor_load_op)
                self.builders[-1].insert(memref_cast_op)

    def print_data_block_construction_from_function_call(self, data_block_construction_from_function_call: DataBlockConstructionFromFunctionCall, indent_level: int = 0):
        memref = data_block_construction_from_function_call.allocation.print(self)
        self.symbol_map[data_block_construction_from_function_call.lhs.name] = memref
        self.print_for(data_block_construction_from_function_call.initialisation)

    def print_data_block_construction_from_operation(self, data_block_construction_from_operation: DataBlockConstructionFromOperation, indent_level: int = 0):
        memref = data_block_construction_from_operation.allocation.print(self, indent_level)
        self.symbol_map[data_block_construction_from_operation.lhs.name] = memref
        data_block_construction_from_operation.assignment.print(self, indent_level)

    def print_data_block_assignment(self, data_block_assignment: DataBlockAssignment, indent_level: int = 0):
        return data_block_assignment.assignment.print(self, indent_level)

    def print_data_block_unary_max(self, data_block_unary_max: DataBlockUnaryMax, indent_level: int = 0) -> str:
        return data_block_unary_max.loops[0].print(self, indent_level)
    
    def print_comment(self, comment: Comment, indent_level: int = 0):
        pass

    #################################
    # Expression Prints
    #################################
    def print_variable(self, variable: Variable, indent_level: int = 0) -> SSAValue:
        return self.symbol_map[variable.name]

    def print_integer(self, integer: Integer, indent_level: int = 0) -> SSAValue:
        assert(isinstance(integer.value, int))
        assert(self.builders[-1] is not None)
        constant_op = arith.ConstantOp(builtin.IntegerAttr(integer.value, 64))
        self.builders[-1].insert(constant_op)
        return constant_op.result

    def print_index(self, index: Index, indent_level: int = 0) -> SSAValue:
        assert(isinstance(index.value, int))
        assert(self.builders[-1] is not None)
        constant_op = arith.ConstantOp(builtin.IntegerAttr(index.value, builtin.IndexType()))
        self.builders[-1].insert(constant_op)
        return constant_op.result
        

    def print_float(self, float_val: Float, indent_level: int = 0) -> SSAValue:
        assert(isinstance(float_val.value, float))
        assert(self.builders[-1] is not None)
        constant_op = arith.ConstantOp(builtin.FloatAttr(float_val.value, 64))
        self.builders[-1].insert(constant_op)
        return constant_op.result

    def generate_affine_expression(self, element_node: Element | Variable) -> AffineExpr:
        if isinstance(element_node, Variable):
            if element_node.name in self.affine_dimension_map:
                return AffineDimExpr(self.affine_dimension_map[element_node.name])
            if element_node.name in self.symbol_map:
                if element_node.name not in self.affine_symbol_map:
                    index = len(self.affine_symbol_map.keys())
                    self.affine_symbol_map[element_node.name] = index
                    return AffineSymExpr(self.affine_symbol_map[element_node.name])
                
        if isinstance(element_node, Index):
            assert(isinstance(element_node.value, int))
            return AffineConstantExpr(element_node.value)
        if isinstance(element_node, ElementBinaryOperation):
            if element_node.operator == Operator.ADDITION:
                return self.generate_affine_expression(element_node.lhs) + self.generate_affine_expression(element_node.rhs)
            if element_node.operator == Operator.SUBTRACTION:
                return self.generate_affine_expression(element_node.lhs) - self.generate_affine_expression(element_node.rhs)
            if element_node.operator == Operator.MULTIPLICATION:
                return self.generate_affine_expression(element_node.lhs) * self.generate_affine_expression(element_node.rhs)
        raise Exception(f"Affine expression generation is not supported for {element_node}")

    def print_subscript(self, subscript: Subscript, indent_level: int = 0) -> SSAValue:
        if len(subscript.indices) == 1:
            memref_ssa_value = subscript.base.print(self)
            assert(isinstance(memref_ssa_value, SSAValue))
            assert(isinstance(memref_ssa_value.type, builtin.MemRefType))
            self.affine_symbol_map = {}
            index_map = self.generate_affine_expression(subscript.indices[0])

            dim_values = [self.symbol_map[name] for name in self.affine_dimension_map.keys()]
            symbol_values = [self.symbol_map[name] for name in self.affine_symbol_map.keys()]
            map = AffineMap(num_dims=len(self.affine_dimension_map), num_symbols=len(self.affine_symbol_map), results=(index_map,))

            if len(memref_ssa_value.type.get_shape()) == 2:
                index_apply_op = affine.ApplyOp(
                    map_operands=dim_values + symbol_values,
                    affine_map=builtin.AffineMapAttr(map),
                )
                inner_dimension_op = arith.ConstantOp(
                    builtin.IntegerAttr(1, builtin.IndexType())
                )
                inner_size_op = memref.DimOp(
                    memref_ssa_value, inner_dimension_op.result
                )
                subview_op = memref.SubviewOp.get(
                    source=memref_ssa_value,
                    offsets=[index_apply_op.result, 0],
                    sizes=[1, inner_size_op.result],
                    strides=[1, 1],
                    result_type=builtin.MemRefType(
                        memref_ssa_value.type.element_type,
                        [builtin.DYNAMIC_INDEX],
                        builtin.StridedLayoutAttr([1], None),
                    ),
                )
                self.builders[-1].insert(index_apply_op)
                self.builders[-1].insert(inner_dimension_op)
                self.builders[-1].insert(inner_size_op)
                self.builders[-1].insert(subview_op)
                return subview_op.result

            if subscript.get_address:
                index_apply_op = affine.ApplyOp(
                    map_operands=dim_values + symbol_values,
                    affine_map=builtin.AffineMapAttr(map),
                )
                dimension_index_op = arith.ConstantOp(
                    builtin.IntegerAttr(0, builtin.IndexType())
                )
                dimension_size_op = memref.DimOp(
                    memref_ssa_value, dimension_index_op.result
                )
                remaining_size_op = arith.SubiOp(
                    dimension_size_op.result, index_apply_op.result
                )
                subview_op = memref.SubviewOp.get(
                    source=memref_ssa_value,
                    offsets=[index_apply_op.result],
                    sizes=[remaining_size_op.result],
                    strides=[1],
                    result_type=builtin.MemRefType(
                        memref_ssa_value.type.element_type,
                        [builtin.DYNAMIC_INDEX],
                        builtin.StridedLayoutAttr([1], None),
                    ),
                )
                self.builders[-1].insert(index_apply_op)
                self.builders[-1].insert(dimension_index_op)
                self.builders[-1].insert(dimension_size_op)
                self.builders[-1].insert(remaining_size_op)
                self.builders[-1].insert(subview_op)
                return subview_op.result

            load_op = affine.LoadOp(memref_ssa_value, indices=dim_values + symbol_values, map = builtin.AffineMapAttr(map))
            self.builders[-1].insert(load_op)
            return load_op.result

        if len(subscript.indices) == 2 and subscript.get_address:
            if isinstance(subscript.base, Variable):
                base = subscript.base.get_value()
            else:
                base = subscript.base
            assert(isinstance(base, Array))
            memref_ssa_value = subscript.base.print(self)
            assert(isinstance(memref_ssa_value, SSAValue))
            assert(isinstance(memref_ssa_value.type, builtin.MemRefType))

            self.affine_symbol_map = {}
            outer_index_map = self.generate_affine_expression(subscript.indices[0])
            inner_index_map = self.generate_affine_expression(subscript.indices[1])
            dim_values = [self.symbol_map[name] for name in self.affine_dimension_map.keys()]
            symbol_values = [self.symbol_map[name] for name in self.affine_symbol_map.keys()]
            map_operands = dim_values + symbol_values
            outer_map = AffineMap(
                num_dims=len(self.affine_dimension_map),
                num_symbols=len(self.affine_symbol_map),
                results=(outer_index_map,),
            )
            inner_map = AffineMap(
                num_dims=len(self.affine_dimension_map),
                num_symbols=len(self.affine_symbol_map),
                results=(inner_index_map,),
            )
            outer_apply_op = affine.ApplyOp(
                map_operands=map_operands,
                affine_map=builtin.AffineMapAttr(outer_map),
            )
            inner_apply_op = affine.ApplyOp(
                map_operands=map_operands,
                affine_map=builtin.AffineMapAttr(inner_map),
            )
            self.builders[-1].insert(outer_apply_op)
            self.builders[-1].insert(inner_apply_op)

            if base.flatten:
                assert(len(memref_ssa_value.type.get_shape()) == 2)
                dimension_index_op = arith.ConstantOp(
                    builtin.IntegerAttr(1, builtin.IndexType())
                )
                dimension_size_op = memref.DimOp(
                    memref_ssa_value, dimension_index_op.result
                )
                remaining_size_op = arith.SubiOp(
                    dimension_size_op.result, inner_apply_op.result
                )
                subview_op = memref.SubviewOp.get(
                    source=memref_ssa_value,
                    offsets=[outer_apply_op.result, inner_apply_op.result],
                    sizes=[1, remaining_size_op.result],
                    strides=[1, 1],
                    result_type=builtin.MemRefType(
                        memref_ssa_value.type.element_type,
                        [builtin.DYNAMIC_INDEX],
                        builtin.StridedLayoutAttr([1], None),
                    ),
                )
                self.builders[-1].insert(dimension_index_op)
                self.builders[-1].insert(dimension_size_op)
                self.builders[-1].insert(remaining_size_op)
                self.builders[-1].insert(subview_op)
                return subview_op.result

            else:
                load_op = affine.LoadOp(
                    memref_ssa_value,
                    indices=map_operands,
                    map=builtin.AffineMapAttr(outer_map),
                )
                self.builders[-1].insert(load_op)
                inner_memref_ssa_value = load_op.result
                assert(isinstance(inner_memref_ssa_value.type, builtin.MemRefType))
                dimension_index_op = arith.ConstantOp(
                    builtin.IntegerAttr(0, builtin.IndexType())
                )
                dimension_size_op = memref.DimOp(
                    inner_memref_ssa_value, dimension_index_op.result
                )
                remaining_size_op = arith.SubiOp(
                    dimension_size_op.result, inner_apply_op.result
                )
                subview_op = memref.SubviewOp.get(
                    source=inner_memref_ssa_value,
                    offsets=[inner_apply_op.result],
                    sizes=[remaining_size_op.result],
                    strides=[1],
                    result_type=builtin.MemRefType(
                        inner_memref_ssa_value.type.element_type,
                        [builtin.DYNAMIC_INDEX],
                        builtin.StridedLayoutAttr([1], None),
                    ),
                )
                self.builders[-1].insert(dimension_index_op)
                self.builders[-1].insert(dimension_size_op)
                self.builders[-1].insert(remaining_size_op)
                self.builders[-1].insert(subview_op)
                return subview_op.result

        if len(subscript.indices) == 2 and not subscript.get_address:
            memref_ssa_value = subscript.base.print(self)
            assert(isinstance(memref_ssa_value, SSAValue))
            assert(isinstance(memref_ssa_value.type, builtin.MemRefType))
            self.affine_symbol_map = {}
            outer_index_map = self.generate_affine_expression(subscript.indices[0])
            inner_index_map = self.generate_affine_expression(subscript.indices[1])

            dim_values = [self.symbol_map[name] for name in self.affine_dimension_map.keys()]
            symbol_values = [self.symbol_map[name] for name in self.affine_symbol_map.keys()]
            if subscript.flatten:
                map = AffineMap(num_dims=len(self.affine_dimension_map), num_symbols=len(self.affine_symbol_map), results=(outer_index_map, inner_index_map))
                load_op = affine.LoadOp(memref_ssa_value, indices=dim_values + symbol_values, map = builtin.AffineMapAttr(map))
                self.builders[-1].insert(load_op)
                return load_op.result

            outer_map = AffineMap(num_dims=len(self.affine_dimension_map), num_symbols=len(self.affine_symbol_map), results=(outer_index_map,))
            inner_map = AffineMap(num_dims=len(self.affine_dimension_map), num_symbols=len(self.affine_symbol_map), results=(inner_index_map,))
            outer_load_op = affine.LoadOp(
                memref_ssa_value,
                indices=dim_values + symbol_values,
                map=builtin.AffineMapAttr(outer_map),
            )
            self.builders[-1].insert(outer_load_op)
            inner_load_op = affine.LoadOp(
                outer_load_op.result,
                indices=dim_values + symbol_values,
                map=builtin.AffineMapAttr(inner_map),
            )
            self.builders[-1].insert(inner_load_op)
            return inner_load_op.result
        

    def print_element_binary_operation(self, element_binary_operation: ElementBinaryOperation, indent_level: int = 0) -> SSAValue:
        lhs = element_binary_operation.lhs.print(self)
        rhs = element_binary_operation.rhs.print(self)

        assert(isinstance(lhs, SSAValue))
        assert(isinstance(rhs, SSAValue))
        if lhs.type == builtin.IntegerType(64):
            cast_op = arith.SIToFPOp(lhs, builtin.Float64Type())
            lhs = cast_op.result
            self.builders[-1].insert(cast_op)
        if lhs.type == builtin.IndexType():
            cast_op = arith.UIToFPOp(lhs, builtin.Float64Type())
            lhs = cast_op.result
            self.builders[-1].insert(cast_op)
        if rhs.type == builtin.IntegerType(64):
            cast_op = arith.SIToFPOp(rhs, builtin.Float64Type())
            rhs = cast_op.result
            self.builders[-1].insert(cast_op)
        if rhs.type == builtin.IndexType():
            cast_op = arith.UIToFPOp(rhs, builtin.Float64Type())
            rhs = cast_op.result
            self.builders[-1].insert(cast_op)
        

        match element_binary_operation.operator:
            case Operator.ADDITION:
                arith_op = arith.AddfOp(lhs, rhs)
            case Operator.SUBTRACTION:
                arith_op = arith.SubfOp(lhs, rhs)
            case Operator.MULTIPLICATION:
                arith_op = arith.MulfOp(lhs, rhs)
            case Operator.DIVISION:
                arith_op = arith.DivfOp(lhs, rhs)
            case Operator.MAX:
                arith_op = arith.MaximumfOp(lhs, rhs)
        
        self.builders[-1].insert(arith_op)
        return arith_op.result

    def print_element_unary_operation(self, element_unary_operation: ElementUnaryOperation, indent_level: int = 0):
        assert(element_unary_operation.operator == Operator.SUBTRACTION)
        operand = element_unary_operation.operand.print(self)
        assert(isinstance(operand, SSAValue))

        if operand.type == builtin.Float64Type():
            arith_op = arith.NegfOp(operand)
        else:
            arith_op = arith.NegiOp(operand)

        self.builders[-1].insert(arith_op)
        return arith_op.result

    def print_element_index(self, element_index: ElementIndex, indent_level: int = 0):
        base = element_index.element.print(self)
        assert(isinstance(base, SSAValue))
        assert(isinstance(base.type, builtin.MemRefType))
        self.affine_symbol_map = {}
        index_map = self.generate_affine_expression(element_index.index)

        dim_values = [self.symbol_map[name] for name in self.affine_dimension_map.keys()]
        symbol_values = [self.symbol_map[name] for name in self.affine_symbol_map.keys()]
        map = AffineMap(num_dims=len(self.affine_dimension_map), num_symbols=len(self.affine_symbol_map), results=(index_map,))

        load_op = affine.LoadOp(base, indices=dim_values + symbol_values, map = builtin.AffineMapAttr(map))
        self.builders[-1].insert(load_op)
        return load_op.result

    #################################
    # Type Prints
    #################################
    def print_integer_type(self, integer_type: TInteger, indent_level: int = 0) -> builtin.IntegerType:
        return builtin.IntegerType(64)
    
    def print_index_type(self, index_type: TIndex, indent_level: int = 0) -> builtin.IndexType:
        return builtin.IndexType()

    def print_float_type(self, float_type: TFloat, indent_level: int = 0) -> builtin.Float64Type:
        return builtin.Float64Type()

    def print_vector_type(self, vector_type: TVector, indent_level: int = 0):
        return llvm.LLVMPointerType()
    
    def print_array_type(self, array_type: TArray, indent_level: int = 0) -> builtin.MemRefType:
        element_type_print = array_type.element_type.print(self)
        assert(isinstance(element_type_print, Attribute))

        if isinstance(array_type.element_type, TVector):
            if array_type.number_of_dimensions == 1:
                return builtin.MemRefType(element_type=element_type_print, shape=[builtin.DYNAMIC_INDEX, array_type.element_type.dimensions])
            elif array_type.flatten:
                return builtin.MemRefType(element_type=element_type_print, shape=[builtin.DYNAMIC_INDEX, builtin.DYNAMIC_INDEX, array_type.element_type.dimensions])
            else:
                return builtin.MemRefType(element_type=builtin.MemRefType(element_type=element_type_print, shape=[builtin.DYNAMIC_INDEX]), shape=[builtin.DYNAMIC_INDEX, array_type.element_type.dimensions])
        else:
            if array_type.number_of_dimensions == 1:
                return builtin.MemRefType(element_type=element_type_print, shape=[builtin.DYNAMIC_INDEX])
            elif array_type.flatten:
                return builtin.MemRefType(element_type=element_type_print, shape=[builtin.DYNAMIC_INDEX, builtin.DYNAMIC_INDEX])
            else:
                return builtin.MemRefType(element_type=builtin.MemRefType(element_type=element_type_print, shape=[builtin.DYNAMIC_INDEX]), shape=[builtin.DYNAMIC_INDEX])

    def print_struct_type(self, struct_type: TStruct, indent_level: int = 0) -> Attribute:
        return f"""{self.get_indent(indent_level)}{struct_type.name.id}"""

    def print_function_type(self, function_type: TFunction, indent_level: int = 0) -> builtin.FunctionType:
        xdsl_arguments: list[Attribute] = []
        for argument_type in function_type.argument_types:
            if isinstance(argument_type, TArray):
                xdsl_arguments.append(llvm.LLVMPointerType())
            else:
                xdsl_arguments.append(argument_type.print(self))
        return builtin.FunctionType.from_lists(inputs = xdsl_arguments, outputs = [])