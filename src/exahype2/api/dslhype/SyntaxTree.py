from abc import ABC, abstractmethod
import os
import copy
from enum import Enum
from operator import itemgetter

# We need to track the context of the code generation to support MLIR
class Context:

    def __init__(self):
        # A stack of blocks e.g. for loops where we can store hoisted MLIR operations
        self._blocks = [ Context.Block() ] 	
        
    def push_block(self):
        self._blocks.append(Context.Block())
        
    def pop_block(self):
        if self._blocks:
            return self._blocks.pop()
        else:
            return Context.Block()
        
    def get_block(self):
        if not self._blocks:
            self.push_block()
        return self._blocks[-1]

    def clear_block(self):
        if self._blocks:
            self._blocks[-1] = Context.Block()

    # We need to store block-level MLIR that has been hoisted and ensure we don't
    # repeat casting operators for indices within a block e.g. loops    
    class Block:
        
        def __init__(self):
            # We use this to hoist MLIR operations out of array indexing, function calls etc.
            self._mlir = []            
            # We use a set to store the ids that we have already processed within the block
            self._ids = {}           

        def push_mlir(self):
            self._mlir.append([])

        def pop_mlir(self):
            if self._mlir:
                return self._mlir.pop()
            else:
                return []
            
        def get_mlir(self):
            if not self._mlir:
                self.push_mlir()
            return self._mlir[-1]    

        def mlir_append(self, op):
            self.get_mlir().append(op)  
            
        # def add_id(self, id):
        #     self._ids.update(id)

        # def remove_id(self, id):
        #     self._ids.remove(id)

        # def get_id(self, id):
        #     return self._ids.get(id)
                        
        # def find_id(self, id):
        #     return id in self._ids


        
class Node(ABC):
    output_format = None
    spaces_per_tab = 4
    use_accelerator = False
    use_memory_manager = False
    mlir_id_count = 0	# We need a variable to keep track of the MLIR ids %0, %1 etc.
    type_map = {}
    struct_map = {}
    context = Context()
    target_variable_name = None  # For tracking when we're generating a specific variable name
    mlir_symbol_table = {}  # Maps DSL names (e.g., "patchData.QOut") to MLIR variable names (e.g., "%output")
    mlir_memref_types = {}  # Maps MLIR variable names (e.g., "%output") to their memref types (e.g., "memref<?x?xf64>")
   
    # Initialise the type, struct and function maps
    # NOTE: Populated lazily after Type classes are defined (see __init_type_map below)
    type_map.update({'void': 'void'})  # Will be replaced with TVoid() in Phase 2a
    type_map.update({'int': 'i64'})  # Match TInteger class and C++ kernel expectations
    type_map.update({'double': 'f64'})  # Will be replaced with TFloat() in Phase 2a
    type_map.update({'double*': 'memref<?xf64>'})
    type_map.update({'tarch::la::Vector<2, double>': 'memref<?xi64>'}) # Using memref instead of !llvm.ptr for standard indexing
    type_map.update({'tarch::la::Vector<3, double>': 'memref<?xi64>'}) # Using memref instead of !llvm.ptr for standard indexing
    type_map.update({'CellData<double, double>&': '!llvm.ptr'}) #'memref<?xstruct<f64, f64>>'})
    type_map.update({'const FluxFunctor&': '(!llvm.ptr, !llvm.ptr, !llvm.ptr, f64, f64, i32, !llvm.ptr) -> ()'})
    type_map.update({'const SourceFunctor&': '(!llvm.ptr, !llvm.ptr, !llvm.ptr, f64, f64, !llvm.ptr) -> ()'})
    type_map.update({'const NonconservativeProductFunctor&': '(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, f64, f64, i32, !llvm.ptr) -> ()'})   
    type_map.update({'const MaxEigenvalueFunctorInPlace&': '(!llvm.ptr, !llvm.ptr, !llvm.ptr, f64, f64, i32, !llvm.ptr) -> ()'})
    type_map.update({'const MaxEigenvalueFunctor&': '(!llvm.ptr, !llvm.ptr, !llvm.ptr, f64, f64, i32, !llvm.ptr) -> ()'})
    type_map.update({'peano4::utils::LoopPlacement': 'i32'})
    # Note: QIn, QOut, cellCentre, cellSize, t, dt, id, numberOfCells, etc. are populated from struct_map below
    
    # Functor function MLIR signatures for individual function calls
    type_map.update({'flux': '(memref<?xf64>, !llvm.ptr, !llvm.ptr, f64, f64, i32, memref<?xf64>) -> ()'})
    type_map.update({'nonconservativeProduct': '(memref<?xf64>, memref<?xf64>, !llvm.ptr, !llvm.ptr, f64, f64, i32, memref<?xf64>) -> ()'})
    type_map.update({'sourceTerm': '(memref<?xf64>, !llvm.ptr, !llvm.ptr, f64, f64, memref<?xf64>) -> ()'})
    type_map.update({'volumetricSource': '(memref<?xf64>, !llvm.ptr, f64, memref<?xf64>) -> ()'})
    type_map.update({'maxEigenvalue': '(memref<?xf64>, !llvm.ptr, !llvm.ptr, f64, f64, i32, memref<?xf64>) -> ()'})    

    # Local variable storage types (for runtime-allocated temporary variables)
    type_map.update({'source': 'memref<?xmemref<?xf64>>'})
    type_map.update({'maxEigenvaluesX': 'memref<?xmemref<?xf64>>'})
    type_map.update({'maxEigenvaluesY': 'memref<?xmemref<?xf64>>'})
    type_map.update({'maxEigenvaluesZ': 'memref<?xmemref<?xf64>>'})
    type_map.update({'maxEigenvaluesX2': 'memref<?xmemref<?xf64>>'})
    type_map.update({'maxEigenvaluesY2': 'memref<?xmemref<?xf64>>'})
    type_map.update({'maxEigenvaluesZ2': 'memref<?xmemref<?xf64>>'})
    type_map.update({'maxEigenvaluesXY': 'memref<?xmemref<?xf64>>'})
    type_map.update({'maxEigenvaluesPerVolume': 'memref<?xmemref<?xf64>>'})
    
    # Lambda/flux arrays (2D double** patterns) - These are ASSIGNED to with nested subscripts
    # so they need nested memref type mapping
    type_map.update({'lambdaLeft': 'memref<?xmemref<?xf64>>'})
    type_map.update({'lambdaRight': 'memref<?xmemref<?xf64>>'})
    type_map.update({'lambdaBottom': 'memref<?xmemref<?xf64>>'})
    type_map.update({'lambdaTop': 'memref<?xmemref<?xf64>>'})
    type_map.update({'lambdaNear': 'memref<?xmemref<?xf64>>'})
    type_map.update({'lambdaFar': 'memref<?xmemref<?xf64>>'})
    type_map.update({'fluxLeft': 'memref<?xmemref<?xf64>>'})
    type_map.update({'fluxRight': 'memref<?xmemref<?xf64>>'})
    type_map.update({'fluxBottom': 'memref<?xmemref<?xf64>>'})
    type_map.update({'fluxTop': 'memref<?xmemref<?xf64>>'})
    type_map.update({'fluxNear': 'memref<?xmemref<?xf64>>'})
    type_map.update({'fluxFar': 'memref<?xmemref<?xf64>>'})
    
    # Bridge arrays: Used as temporary storage, allocated with outer dimension then inner arrays
    # Like lambdaLeft, these are semantically "array of arrays" in the C++ kernel
    # Must use nested memref type to match the allocation pattern
    type_map.update({'flux_x': 'memref<?xmemref<?xf64>>'})
    type_map.update({'flux_y': 'memref<?xmemref<?xf64>>'})
    type_map.update({'flux_z': 'memref<?xmemref<?xf64>>'})
    type_map.update({'average_x': 'memref<?xmemref<?xf64>>'})
    type_map.update({'average_y': 'memref<?xmemref<?xf64>>'})
    type_map.update({'average_z': 'memref<?xmemref<?xf64>>'})
    type_map.update({'delta_x': 'memref<?xmemref<?xf64>>'})
    type_map.update({'delta_y': 'memref<?xmemref<?xf64>>'})
    type_map.update({'delta_z': 'memref<?xmemref<?xf64>>'})
    type_map.update({'nonconservative_x': 'memref<?xmemref<?xf64>>'})
    type_map.update({'nonconservative_y': 'memref<?xmemref<?xf64>>'})
    type_map.update({'nonconservative_z': 'memref<?xmemref<?xf64>>'})

    # MLIR to C++ type mappings for function signature generation
    type_map.update({'memref<?xf64>': 'void*'})
    type_map.update({'!llvm.ptr': 'void*'})
    type_map.update({'f64': 'double'})
    type_map.update({'i32': 'int'})
    type_map.update({'i64': 'long'})
    type_map.update({'i1': 'bool'})

    # MLIRCellData struct layout with MemRefDescriptor fields
    descriptor2DType = "!llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>"
    descriptor1DType = "!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>"
    
    # Updated patchTypes: 7 descriptors + 3 i64 scalars + 2 ptrs = 12 fields
    patchTypes = [descriptor2DType, descriptor2DType, descriptor1DType, descriptor1DType, 
                  descriptor1DType, descriptor1DType, descriptor1DType, 'i64', 'i64', 'i64', 'ptr', 'ptr']
    
    struct_map.update({'patchData': [ 
            ('QIn' , f"{descriptor2DType}|||!llvm.struct<({', '.join(patchTypes)})>"),                    # Field 0: 2D descriptor
            ('QOut' , f"{descriptor2DType}|||!llvm.struct<({', '.join(patchTypes)})>"),                   # Field 1: 2D descriptor
            ('cellCentre' , f"{descriptor1DType}|||!llvm.struct<({', '.join(patchTypes)})>"),               # Field 2: 1D descriptor
            ('cellSize' , f"{descriptor1DType}|||!llvm.struct<({', '.join(patchTypes)})>"),                 # Field 3: 1D descriptor
            ('t' , f"{descriptor1DType}|||!llvm.struct<({', '.join(patchTypes)})>"),                        # Field 4: 1D descriptor
            ('dt' , f"{descriptor1DType}|||!llvm.struct<({', '.join(patchTypes)})>"),                       # Field 5: 1D descriptor
            ('id' , f"{descriptor1DType}|||!llvm.struct<({', '.join(patchTypes)})>"),                       # Field 6: 1D descriptor
            ('numberOfCells' , f"i64|||!llvm.struct<({', '.join(patchTypes)})>"),                      # Field 7: i64
            ('memoryLocation' , f"i64|||!llvm.struct<({', '.join(patchTypes)})>"),                     # Field 8: i64
            ('targetDevice' , f"i64|||!llvm.struct<({', '.join(patchTypes)})>"),                       # Field 9: i64
            ('QOut_legacy' , f"!llvm.ptr|||!llvm.struct<({', '.join(patchTypes)})>"),                  # Field 10: function pointer
            ('maxEigenvalue' , f"memref<?xf64>|||!llvm.struct<({', '.join(patchTypes)})>")]})              # Field 11: function pointer

    # Populate type_map with descriptor field types from struct_map
    # This maps field names to their converted MLIR memref types
    for struct_name, fields in struct_map.items():
        for field_name, field_info in fields:
            target_type = field_info.split('|||')[0]
            # Convert descriptor types to their memref equivalents
            if target_type == descriptor2DType:
                type_map.update({field_name: 'memref<?x?xf64>'})
            elif target_type == descriptor1DType:
                # Distinguish based on field name:
                # - cellCentre, cellSize, h, x: these are Vector<2,double> fields (stored as i64 pointers)
                # - t, dt, id: these are scalar f64 fields
                if field_name in ['cellCentre', 'cellSize', 'h', 'x']:
                    type_map.update({field_name: 'memref<?xi64>'})  # Vector fields (pointers to vectors)
                else:
                    type_map.update({field_name: 'memref<?xf64>'})  # Scalar fields (f64 values)
            elif target_type == 'i64':
                type_map.update({field_name: 'i64'})
            elif target_type == '!llvm.ptr':
                # For MLIR pointer types, keep them as-is (they're function pointers, not data pointers)
                type_map.update({field_name: '!llvm.ptr'})
            else:
                # For other types, use the target_type directly
                type_map.update({field_name: target_type})

    def __init__(self):
        self._parent = None
        self._on_left = False
        self._hoist = False 
        self._index_type = False
        
    def parent(self, parent = None):
        if parent is not None:
          self._parent = parent
        return self._parent
                  
    def on_left(self, left = None):
        if left is not None:
          self._on_left = left
        return self._on_left 

    def hoist(self, flag = None):
        if flag is not None:
          self._hoist = flag
        return self._hoist
              
    @classmethod    
    def get_mlir_id(cls):
        current_reg = cls.mlir_id_count
        cls.mlir_id_count += 1       
        return current_reg
   
    @classmethod    
    def reset_mlir_ids(cls):
        cls.mlir_id_count = 0
 
    @staticmethod
    def get_mlir_type(type_string):
        # Handle string representations by creating temporary type objects
        if isinstance(type_string, str):
            if type_string == "int":
                return TInteger().print_mlir(0)
            elif type_string == "double":
                return TFloat().print_mlir(0)
            elif type_string == "bool":
                return TBoolean().print_mlir(0)
        return None

    @staticmethod
    def generate_pointer_to_2d_element(memref_mlir, row_index, col_index, is_nested, indent_level=0):
        indent = ' ' * indent_level * Node.spaces_per_tab
        mlir_code = ""
        
        if is_nested:
            # Load inner memref from outer array, then offset within it
            inner_memref_id = f"%{Node.get_mlir_id()}"
            mlir_code += indent + f"{inner_memref_id} = memref.load {memref_mlir}[{row_index}] {{name = \"{memref_mlir.strip('%')}\"}} : memref<?xmemref<?xf64>>\n"
            
            # Extract pointer as index from inner memref
            ptr_idx_id = f"%{Node.get_mlir_id()}"
            mlir_code += indent + f"{ptr_idx_id} = memref.extract_aligned_pointer_as_index {inner_memref_id} : memref<?xf64> -> index {{name = \"{memref_mlir.strip('%')}\"}}\n"
            
            # Byte offset calculation (col * 8 for f64 elements)
            element_size_bytes = f"%{Node.get_mlir_id()}"
            mlir_code += indent + f"{element_size_bytes} = arith.constant 8 : index\n"
            
            offset_bytes_id = f"%{Node.get_mlir_id()}"
            mlir_code += indent + f"{offset_bytes_id} = arith.muli {col_index}, {element_size_bytes} : index\n"
            
            # Add offset to base pointer
            ptr_with_offset = f"%{Node.get_mlir_id()}"
            mlir_code += indent + f"{ptr_with_offset} = arith.addi {ptr_idx_id}, {offset_bytes_id} : index\n"
            
            # Convert to i64, then to llvm.ptr
            ptr_i64 = f"%{Node.get_mlir_id()}"
            mlir_code += indent + f"{ptr_i64} = arith.index_cast {ptr_with_offset} : index to i64\n"
            
            result_ptr = f"%{Node.get_mlir_id()}"
            mlir_code += indent + f"{result_ptr} = llvm.inttoptr {ptr_i64} : i64 to !llvm.ptr\n"
        else:
            # Flat 2D memref: compute (row * width + col) offset
            col_width_id = f"%{Node.get_mlir_id()}"
            one_const = f"%{Node.get_mlir_id()}"
            mlir_code += indent + f"{one_const} = arith.constant 1 : index\n"
            mlir_code += indent + f"{col_width_id} = memref.dim {memref_mlir}, {one_const} : memref<?x?xf64>\n"
            
            # Calculate row offset (row * width)
            row_offset = f"%{Node.get_mlir_id()}"
            mlir_code += indent + f"{row_offset} = arith.muli {row_index}, {col_width_id} : index\n"
            
            # Add column to row offset
            elem_offset = f"%{Node.get_mlir_id()}"
            mlir_code += indent + f"{elem_offset} = arith.addi {row_offset}, {col_index} : index\n"
            
            # Convert element offset to bytes
            element_size_bytes = f"%{Node.get_mlir_id()}"
            mlir_code += indent + f"{element_size_bytes} = arith.constant 8 : index\n"
            
            offset_bytes_id = f"%{Node.get_mlir_id()}"
            mlir_code += indent + f"{offset_bytes_id} = arith.muli {elem_offset}, {element_size_bytes} : index\n"
            
            # Extract base pointer from memref
            base_ptr_idx = f"%{Node.get_mlir_id()}"
            mlir_code += indent + f"{base_ptr_idx} = memref.extract_aligned_pointer_as_index {memref_mlir} : memref<?x?xf64> -> index {{name = \"{memref_mlir.strip('%')}\"}}\n"
            
            # Add byte offset to base
            ptr_with_offset = f"%{Node.get_mlir_id()}"
            mlir_code += indent + f"{ptr_with_offset} = arith.addi {base_ptr_idx}, {offset_bytes_id} : index\n"
            
            # Convert to i64, then to llvm.ptr
            ptr_i64 = f"%{Node.get_mlir_id()}"
            mlir_code += indent + f"{ptr_i64} = arith.index_cast {ptr_with_offset} : index to i64\n"
            
            result_ptr = f"%{Node.get_mlir_id()}"
            mlir_code += indent + f"{result_ptr} = llvm.inttoptr {ptr_i64} {{name = \"{memref_mlir.strip('%')}\"}} : i64 to !llvm.ptr\n"
        
        return mlir_code, result_ptr

    @classmethod
    def create_memref_from_ptr(cls, ptr_id, target_type, indent_level=0, size_hint=None, target_var_name=None):
        indent = ' ' * indent_level * cls.spaces_per_tab
        
        # Generate unique IDs for all the operations
        desc_id = f"%{cls.get_mlir_id()}"
        desc_0_id = f"%{cls.get_mlir_id()}"
        desc_1_id = f"%{cls.get_mlir_id()}"
        desc_2_id = f"%{cls.get_mlir_id()}"
        desc_3_id = f"%{cls.get_mlir_id()}"
        desc_4_id = f"%{cls.get_mlir_id()}"
        
        # Use target variable name if provided, otherwise generate a new one
        if target_var_name:
            memref_id = target_var_name
        else:
            memref_id = f"%{cls.get_mlir_id()}"
        
        # Create constants we need
        c0_i64_id = f"%{cls.get_mlir_id()}"
        c1_i64_id = f"%{cls.get_mlir_id()}"
        
        # Use standard LLVM memref descriptor format - this is part of the fix supplied by Nick Brown
        # For 2D memrefs, use 2D descriptor; for 1D memrefs, use 1D descriptor
        if target_type.startswith('memref<?x?'):
            # 2D memref needs 2D descriptor
            struct_type = '!llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>'
            num_dimensions = 2
        else:
            # 1D memref needs 1D descriptor
            struct_type = '!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>'
            num_dimensions = 1
        
        # Determine size based on target type or hint
        if target_type.startswith('memref<?xmemref') or size_hint == 1:
            size_val = "1"
        elif target_type.startswith('memref<?xi64>') and size_hint == 2:
            size_val = "2"
        elif target_type.startswith('memref<?xi64>'):
            size_val = "2"  # Default for vector types
        else:
            size_val = "1"  # Default for scalar-like types
            
        mlir_str = ""
        
        # Create the required constants
        mlir_str += indent + f"{c0_i64_id} = arith.constant 0 : i64\n"
        mlir_str += indent + f"{c1_i64_id} = arith.constant 1 : i64\n"
        
        # If we need a size constant, create it
        if size_val != "1":
            size_id = f"%{cls.get_mlir_id()}"
            size_i64_id = f"%{cls.get_mlir_id()}"
            mlir_str += indent + f"{size_id} = arith.constant {size_val} : index\n"
            mlir_str += indent + f"{size_i64_id} = arith.index_cast {size_id} : index to i64\n"
            size_ref = size_i64_id
        else:
            size_ref = c1_i64_id
            
        # Create the LLVM memref descriptor
        mlir_str += indent + f"{desc_id} = llvm.mlir.undef : {struct_type}\n"
        mlir_str += indent + f"{desc_0_id} = llvm.insertvalue {ptr_id}, {desc_id}[0] : {struct_type}\n"
        mlir_str += indent + f"{desc_1_id} = llvm.insertvalue {ptr_id}, {desc_0_id}[1] : {struct_type}\n"
        mlir_str += indent + f"{desc_2_id} = llvm.insertvalue {c0_i64_id}, {desc_1_id}[2] : {struct_type}\n"
        mlir_str += indent + f"{desc_3_id} = llvm.insertvalue {size_ref}, {desc_2_id}[3, 0] : {struct_type}\n"
        if num_dimensions == 2:
            # For 2D descriptor, set [3, 1], [4, 0] and [4, 1]
            desc_3_1_id = f"%{cls.get_mlir_id()}"
            mlir_str += indent + f"{desc_3_1_id} = llvm.insertvalue {size_ref}, {desc_3_id}[3, 1] : {struct_type}\n"
            mlir_str += indent + f"{desc_4_id} = llvm.insertvalue {c1_i64_id}, {desc_3_1_id}[4, 0] : {struct_type}\n"
            desc_final_id = f"%{cls.get_mlir_id()}"
            mlir_str += indent + f"{desc_final_id} = llvm.insertvalue {c1_i64_id}, {desc_4_id}[4, 1] : {struct_type}\n"
        else:
            # For 1D descriptor, just set [4, 0]
            mlir_str += indent + f"{desc_4_id} = llvm.insertvalue {c1_i64_id}, {desc_3_id}[4, 0] : {struct_type}\n"
            desc_final_id = desc_4_id
        mlir_str += indent + f"{memref_id} = builtin.unrealized_conversion_cast {desc_final_id} : {struct_type} to {target_type} {{name = \"{memref_id.strip('%')}\"}}\n"
        
        return mlir_str, memref_id

    @classmethod
    def create_memref_from_extracted_descriptor(cls, alloc_ptr_id, aligned_ptr_id, offset_id, sizes_0_id, sizes_1_id, strides_0_id, strides_1_id, target_type, struct_type, indent_level=0, target_var_name=None, is_2d=False):
        indent = ' ' * indent_level * cls.spaces_per_tab
        
        # Use target variable name if provided
        if target_var_name:
            memref_id = target_var_name
        else:
            memref_id = f"%{cls.get_mlir_id()}"
        
        # Generate unique IDs for all the operations
        desc_id = f"%{cls.get_mlir_id()}"
        desc_0_id = f"%{cls.get_mlir_id()}"
        desc_1_id = f"%{cls.get_mlir_id()}"
        desc_2_id = f"%{cls.get_mlir_id()}"
        desc_3_id = f"%{cls.get_mlir_id()}"
        desc_4_id = f"%{cls.get_mlir_id()}"
        
        c1_i64_id = f"%{cls.get_mlir_id()}"
        
        mlir_str = ""
        
        # Create constants we need
        mlir_str += indent + f"{c1_i64_id} = arith.constant 1 : i64\n"
        
        # Create the LLVM memref descriptor using the extracted fields
        # We preserve the extracted pointers but hardcode sizes and strides to 1
        mlir_str += indent + f"{desc_id} = llvm.mlir.undef : {struct_type}\n"
        mlir_str += indent + f"{desc_0_id} = llvm.insertvalue {alloc_ptr_id}, {desc_id}[0] : {struct_type}\n"
        mlir_str += indent + f"{desc_1_id} = llvm.insertvalue {aligned_ptr_id}, {desc_0_id}[1] : {struct_type}\n"
        mlir_str += indent + f"{desc_2_id} = llvm.insertvalue {offset_id}, {desc_1_id}[2] : {struct_type}\n"
        mlir_str += indent + f"{desc_3_id} = llvm.insertvalue {sizes_0_id}, {desc_2_id}[3, 0] : {struct_type}\n"
        
        # Check if 2D or 1D 
        if is_2d:
            # 2D descriptor - use all extracted components
            desc_3_1_id = f"%{cls.get_mlir_id()}"
            mlir_str += indent + f"{desc_3_1_id} = llvm.insertvalue {sizes_1_id}, {desc_3_id}[3, 1] : {struct_type}\n"
            mlir_str += indent + f"{desc_4_id} = llvm.insertvalue {strides_0_id}, {desc_3_1_id}[4, 0] : {struct_type}\n"
            desc_final_id = f"%{cls.get_mlir_id()}"
            mlir_str += indent + f"{desc_final_id} = llvm.insertvalue {strides_1_id}, {desc_4_id}[4, 1] : {struct_type}\n"
        else:
            # 1D descriptor - use strides_0_id
            mlir_str += indent + f"{desc_4_id} = llvm.insertvalue {strides_0_id}, {desc_3_id}[4, 0] : {struct_type}\n"
            desc_final_id = desc_4_id
        
        mlir_str += indent + f"{memref_id} = builtin.unrealized_conversion_cast {desc_final_id} : {struct_type} to {target_type} {{name = \"{memref_id.strip('%')}\"}}\n"
        
        return mlir_str, memref_id
 
    @abstractmethod
    def print_cpp(self, indent_level = 0):
        pass

    @abstractmethod
    def print_mlir(self, indent_level = 0):
        pass

    @abstractmethod
    def print_omp(self, indent_level = 0):
        pass

    @abstractmethod
    def print_sycl(self, indent_level = 0):
        pass
    
    @abstractmethod
    def print_tree(self, indent_level = 0):
        pass

class Statement(Node):
    def __init__(self):
        super().__init__()


class Expression(Node):
    def __init__(self):
        super().__init__()
        self._mlir_id = None

    def mlir_id(self, id = None):
        if id is not None:
            self._mlir_id = id
        return self._mlir_id
        
    def multidimensional_index(self, index_list, start_index=0, dimensions = []):
        return self
    
    @abstractmethod
    def get_type(self):
        pass


class Type(Node):

    def print_tree(self, indent_level=0):
        return ' ' 

    def get_element_type(self):
        return None


class TVoid(Type):
    def print_cpp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + 'void'

    def print_mlir(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + 'void'

    def print_omp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + 'void'

    def print_sycl(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + 'void'

    def is_function(self):
        return False
    
 

class Argument(Node):
    def __init__(self, id, argument_type: Type, namespace = None, is_function = False):
        super().__init__()
        self.id = id
        self._type = argument_type
        self._type.parent(self)
        self._namespace = namespace
        self._is_function = is_function
        if Node.use_accelerator == True:
            if type(self._type) is TCustom and self._type._type.find("CellData") != -1:
                index = self._type._type.find("CellData")
                self._type._type = self._type._type[0:index] + "CopyCellDataGPU" + self._type._type[index + 8:]
    
    def print_cpp(self, indent_level = 0):
        return self._type.print_cpp() + " " + self.id
    
    def print_omp(self, indent_level = 0):
        return self._type.print_omp() + " " + self.id
    
    def print_sycl(self, indent_level = 0):
        return self._type.print_sycl() + " " + self.id

    def print_mlir(self, indent_level = 0):
        # NOTE: Arguments are inserted in-line into 'func.func' op
        return f"%{self.id} : {self._type.print_mlir(0)}"
            
    def print_tree(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + f"Argument: {self.id}"


class FunctionDefinition(Node):
    _stateless = False
    _syclDataToCopy = []

    def __init__(self, id, return_type: Type, template = None, namespaces = [], stateless = False, top_level = False):
        super().__init__()
        self.id = id
        self._return_type = return_type
        self._return_type.parent(self)
        self._template = template
        self._arguments = []
        self._body = []
        self._namespaces = namespaces
        FunctionDefinition._stateless = stateless
        self._top_level = top_level

    def top_level(self, top = None):
        if top is not None:
          self._top_level = top
        return self._top_level                

    def add_statement(self, statement: Statement):
        if statement is not None:
            self._body.append(statement)

    def add_argument(self, argument: Argument):
        self._arguments.append(argument)
        if self._top_level:
            # NOTE: For now, we assume that the top-level function is always the main entry point
            #       and set global values in Node here - 'namespace' is MLIR type here
            Node.type_map.update({argument.id: argument._namespace})
            if Node.type_map.get(argument._type.print_cpp()) is None:
                Node.type_map.update({argument._type.print_cpp(): argument._namespace})
            
    def print_declaration(self, indent_level = 0):
        namespace_header = f"""namespace {"::".join(self._namespaces)} {{"""
        namespace_footer = "}"
        template_string = ""
        if self._template is not None:
            template_strings = [argument.print_cpp() for argument in self._template]
            template_string = "template <" + ",".join(template_strings) + ">\n"

        current_indent = indent_level + 1  # Indentation increases inside namespace
        argument_prints = [argument.print_cpp() for argument in self._arguments]

        return f"""{namespace_header}
{template_string}{' ' * current_indent * Node.spaces_per_tab}{self._return_type.print_cpp()} {self.id}({', '.join(argument_prints)});
{namespace_footer}
"""

    def print_cpp(self, indent_level=0):
        namespace_header = f"""namespace {"::".join(self._namespaces)} {{"""
        namespace_footer = "}"
        template_string = ""
        if self._template is not None:
            template_strings = [argument.print_cpp() for argument in self._template]
            template_string = "template <" + ",".join(template_strings) + ">\n"

        current_indent = indent_level + 1  # Indentation increases inside namespace
        statement_prints = [statement.print_cpp(current_indent + 1) for statement in self._body]
        argument_prints = [argument.print_cpp() for argument in self._arguments]

        return f"""{namespace_header}
{template_string}{' ' * current_indent * Node.spaces_per_tab}{self._return_type.print_cpp()} {self.id}({', '.join(argument_prints)}) {{
{os.linesep.join(statement_prints)}
{' ' * current_indent * Node.spaces_per_tab}}}
{namespace_footer}
"""
    
    def print_omp(self, indent_level=0):
        namespace_header = f"""namespace {"::".join(self._namespaces)} {{"""
        namespace_footer = "}"
        template_string = ""
        if self._template is not None:
            template_strings = [argument.print_omp() for argument in self._template]
            template_string = "template <" + ",".join(template_strings) + ">\n"

        current_indent = indent_level + 1  # Indentation increases inside namespace
        statement_prints = [statement.print_omp(current_indent + 1) for statement in self._body]
        argument_prints = [argument.print_omp() for argument in self._arguments]

        return f"""{namespace_header}
{template_string}{' ' * current_indent * Node.spaces_per_tab}{self._return_type.print_omp()} {self.id}({', '.join(argument_prints)}) {{
{os.linesep.join(statement_prints)}
{' ' * current_indent * Node.spaces_per_tab}}}
{namespace_footer}
"""
    
    def print_sycl(self, indent_level=0):
        namespace_header = f"""namespace {"::".join(self._namespaces)} {{"""
        namespace_footer = "}"
        template_string = ""
        if self._template is not None:
            template_strings = [argument.print_sycl() for argument in self._template]
            template_string = "template <" + ",".join(template_strings) + ">\n"

        current_indent = indent_level + 1  # Indentation increases inside namespace
        statement_prints = [statement.print_sycl(current_indent + 1) for statement in self._body]
        argument_prints = [argument.print_sycl() for argument in self._arguments]
        statement_prints.insert(0, "::sycl::queue& queue = tarch::accelerator::getSYCLQueue(targetDevice);")
        statement_prints.insert(1, "size_t range0, range1, range2;")

        data_copy_prints = []
        for dataBlockCreation in FunctionDefinition._syclDataToCopy:
            if len(dataBlockCreation._dataBlock._iteration_range) > 1:
                step_size = Integer(1)
                for d in dataBlockCreation._dataBlock._iteration_range[0:-1]:
                    step_size = step_size * (d[1] - d[0])
                size = step_size * (dataBlockCreation._dataBlock._iteration_range[-1][1] - dataBlockCreation._dataBlock._iteration_range[-1][0])

        return f"""{namespace_header}
{template_string}{' ' * current_indent * Node.spaces_per_tab}{self._return_type.print_sycl()} {self.id}({', '.join(argument_prints)}) {{
{os.linesep.join(statement_prints)}
{os.linesep.join(data_copy_prints)}
{' ' * current_indent * Node.spaces_per_tab}}}
{namespace_footer}
"""
    
    def print_mlir(self, indent_level = 0):
        indent_level += 1  # Increase indentation for MLIR
        
        if self._top_level:
            Node.reset_mlir_ids()  # Reset the MLIR id count for the top-level function

            # NOTE: We used to generate MLIRbridge.cpp using the dedicated MLIRBridgeGenerator module
            globals_mlir = []
            
            # Add external function declarations for bridge functions using dynamic signatures
            # Extract function signatures directly from self._template (functors only, not constants)
            if self._template is not None:
                for argument in self._arguments:
                    mlir_signature = argument._namespace #Node.type_map[cpp_type_str]
                    
                    # Only process functors, skip template constants like "NumberOfVolumesPerAxisInPatch"
                    if argument._is_function:
                        # Create the bridge function declaration in MLIR with _bridge suffix
                        # This matches the C++ wrapper function names (flux_bridge, etc.)
                        func_decl = f"func.func private @{argument.id}_bridge{mlir_signature}"
                        globals_mlir.append(' ' * (indent_level) * Node.spaces_per_tab + func_decl)

                    
            else:
                # No template means no MLIR bridge functions are needed
                raise ValueError(
                    "MLIR code generation requires template parameters to determine bridge function signatures. "
                    "Ensure the function has template parameters that specify the required functors."
                )
            
            # Add addressof/loads for globals inside the function
            function_global_loads = []
            if self._template is not None:
                for argument in self._template:
                    # Generate globals for ALL template parameters (including boolean evaluation flags)
                    new_id = Node.get_mlir_id()
                    globals_mlir.append(' ' * (indent_level) * Node.spaces_per_tab +f"llvm.mlir.global external @{argument.id}() : {argument._type.print_mlir(0)}")
                    function_global_loads.append(' ' * (indent_level + 1) * Node.spaces_per_tab + f"%{new_id} = llvm.mlir.addressof @{argument.id} : !llvm.ptr")
                    
                    # Get the MLIR type for the argument
                    arg_mlir_type = argument._type.print_mlir(0)
                    
                    # Skip global loading for variables with empty types (be more thorough)
                    if arg_mlir_type and arg_mlir_type.strip() != '' and len(arg_mlir_type.strip()) > 0:
                        function_global_loads.append(' ' * (indent_level + 1) * Node.spaces_per_tab + f"%{argument.id} = llvm.load %{new_id} {{name = \"{argument.id.strip('%')}\"}} : !llvm.ptr -> {arg_mlir_type}")
                    else:
                        # For variables with empty types, just create a comment
                        function_global_loads.append(' ' * (indent_level + 1) * Node.spaces_per_tab + f"// Skipping load for {argument.id} (empty type: '{arg_mlir_type}')")
            else:
                globals_mlir = []

            Node.context.push_block()
            argument_prints = [argument.print_mlir(indent_level).strip('') for argument in self._arguments]
            Node.context.push_block()

            statement_prints = [statement.print_mlir(indent_level + 1) for statement in self._body]
            statements = Node.context.pop_block().get_mlir()
            statements += statement_prints
            
            return_type = self._return_type.print_mlir(0)
            return_type_str = f" -> ({return_type})" if return_type != 'void' else ''

            return f"""
builtin.module {{
{os.linesep.join(globals_mlir)}
    func.func @{self.id + "_omp" if Node.use_mlir_omp else self.id}({', '.join(argument_prints)}){return_type_str} {{
{os.linesep.join(function_global_loads)}
{os.linesep.join(statements)}
func.return
    }}
}}
"""
        # Non-top-level case
        Node.context.push_block()
        argument_prints = [argument.print_mlir(indent_level) for argument in self._arguments]
        Node.context.push_block()
        statement_prints = [statement.print_mlir(indent_level + 1) for statement in self._body]
        statements = Node.context.pop_block().get_mlir()
        statements += statement_prints
        return_type = self._return_type.print_mlir(indent_level)
        return_type_str = f" -> ({return_type})" if return_type != 'void' else ''
        return f"""
    func.func @{self.id}({', '.join(argument_prints)}){return_type_str} {{
{os.linesep.join(statements)}
func.return
    }}
"""

    def print_tree(self, indent_level = 0):
        statement_prints = [statement.print_tree(indent_level + 1) for statement in self._body]
        argument_prints = [argument.print_tree(indent_level + 1) for argument in self._arguments]
        return f"""
FunctionDefinition: {self.id}:
{os.linesep.join(argument_prints)}
{os.linesep.join(statement_prints)}"""
    
    def print_definition_with_timer(self, indent_level = 0):
        namespace_header = f"""namespace {"::".join(self._namespaces)} {{"""
        namespace_footer = "}"
        template_string = ""
        function_call_string = self.id
        if self._template is not None:
            template_strings = [argument.print_cpp() for argument in self._template]
            template_string = "template <" + ",".join(template_strings) + ">\n"
            function_call_string += "<" + ",".join([template.split(' ')[1] for template in template_strings]) + ">"

        arguments = copy.deepcopy(self._arguments)
        arguments.append(Argument("measurement", TCustom("tarch::timing::Measurement&")))

        current_indent = indent_level + 1  # Indentation increases inside namespace
        argument_prints = [argument.print_cpp() for argument in arguments]

        return f"""{namespace_header}
{template_string}{' ' * current_indent * Node.spaces_per_tab}{self._return_type.print_cpp()} {self.id}({', '.join(argument_prints)}) {{
tarch::timing::Watch watch("{"::".join(self._namespaces)}", "{self.id}", false, true);
{function_call_string}({",".join([argument.print_cpp().split(' ')[-1] for argument in self._arguments])});
watch.stop();
measurement.setValue(watch.getCalendarTime());
{' ' * current_indent * Node.spaces_per_tab}}}
{namespace_footer}
"""
    
    def print_declaration_with_timer(self, indent_level = 0):
        namespace_header = f"""namespace {"::".join(self._namespaces)} {{"""
        namespace_footer = "}"
        template_string = ""
        if self._template is not None:
            template_strings = [argument.print_cpp() for argument in self._template]
            template_string = "template <" + ",".join(template_strings) + ">\n"

        arguments = copy.deepcopy(self._arguments)
        arguments.append(Argument("measurement", TCustom("tarch::timing::Measurement&")))

        current_indent = indent_level + 1  # Indentation increases inside namespace
        argument_prints = [argument.print_cpp() for argument in arguments]

        return f"""{namespace_header}
{template_string}{' ' * current_indent * Node.spaces_per_tab}{self._return_type.print_cpp()} {self.id}({', '.join(argument_prints)});
{namespace_footer}
"""

class LogToFile(Statement):
    def __init__(self, statementToLog, outputStream):
        self._statement = statementToLog
        self._output = outputStream

    def print_cpp(self, indent_level=0):
        return f"""{self._output.print_cpp()} << {self._statement.print_cpp()};"""
    
    def print_omp(self, indent_level=0):
        return f"""{self._output.print_omp()} << {self._statement.print_omp()};"""
    
    def print_sycl(self, indent_level=0):
        return f"""{self._output.print_sycl()} << {self._statement.print_sycl()};"""
    
    def print_mlir(self, indent_level=0):
        pass

    def print_tree(self, indent_level=0):
        return "LogToFile"


# Expressions
class Name(Expression):
    variables = dict()

    def __init__(self, id, type = None):
        super().__init__()
        self.id = id
        self.type = type
        if self.type is None:
            self.type = Name.variables[self.id].get_type()
        self.type._parent = self

    def multidimensional_index(self, index_list, start_index=0, dimensions = []):
        if type(Name.variables[self.id].get_type()) is TDataBlock or type(Name.variables[self.id].get_type()) is TDiagonalMatrix:
            return Name.variables[self.id].multidimensional_index(index_list, start_index, dimensions)
        else:
            return self

    def index(self, i):
        return Name.variables[self.id].index(i)
    
    def index(self, i, j):
        return Name.variables[self.id].index(i, j)

    def __add__(self, rhs):
        return BinaryOperation("+", self, rhs)

    def __sub__(self, rhs):
        return BinaryOperation("-", self, rhs)
    
    def __mul__(self, rhs):
        return BinaryOperation("*", self, rhs)
    
    def __div__(self, rhs):
        return BinaryOperation("/", self, rhs)
    
    def __neg__(self):
        return UnaryOperation("-", self)

    def __gt__(self, rhs):
        if type(rhs) is Name:
            return (Name.variables[self.id] > Name.variables[rhs.id])
        return (Name.variables[self.id] > rhs)

    def __eq__(self, rhs):
        if type(rhs) is Name:
            if rhs.id in Name.variables and self.id in Name.variables:
                return (self._value == Name.variables[rhs.id])
            else:
                return False
        else:
            if self.id in Name.variables:
                return (Name.variables[self.id] == rhs)
            else:
                return False

    def get_type(self):
        return self.type
    
    def print_cpp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + self.id
    
    def print_omp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + self.id
    
    def print_sycl(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + self.id
    
    def print_mlir(self, indent_level=0):
        # NOTE: Names are returned and used in-linei
        return "%" + self.id 
    
    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "Name:" + self.id

class Index:
    pass

class Integer(Expression):
    def __init__(self, value, string = None):
        super().__init__()
        self._value = int(value)
        self._string = string
    
    def __add__(self, rhs):
        if self._value == 0:
            return rhs
        if type(rhs) is Integer:
            return Integer(self._value + rhs._value)
        return BinaryOperation("+", self, rhs)

    def __sub__(self, rhs):
        if self._value == 0:
            if type(rhs) is Integer:
                return Integer(-rhs._value)
            if type(rhs) is UnaryOperation and rhs._operation == "-":
                return rhs._operand
            return -rhs
        if type(rhs) is Integer:
            return Integer(self._value - rhs._value)
        return BinaryOperation("-", self, rhs)
    
    def __mul__(self, rhs):
        return BinaryOperation("*", self, rhs)
    
    def __div__(self, rhs):
        return BinaryOperation("/", self, rhs)

    def __neg__(self):
        if self._value == 0:
            return self
        else:
            return UnaryOperation("-", self)

    def __lt__(self, rhs):
        if type(rhs) is Name:
            if rhs.id in Name.variables:
                return (self._value < Name.variables[rhs.id])
            else:
                return False
        return (self._value < rhs)
    
    def __gt__(self, rhs):
        if type(rhs) is Name:
            if rhs.id in Name.variables:
                return (self._value > Name.variables[rhs.id])
            else:
                return False
        return (self._value > rhs)

    def __eq__(self, rhs):
        if type(rhs) is Integer:
            return (self._value == rhs._value)

    def __int__(self):
        return self._value

    def get_type(self):
        # Check if this Integer references a struct field in struct_map
        if self._string is not None and '.' in self._string:
            struct, field = self._string.split('.')
            if struct in Node.struct_map:
                struct_list = Node.struct_map[struct]
                # Find the field and get its type
                for item in struct_list:
                    if item[0] == field:
                        # Extract the target type using '|||' separator (struct_map format: "target_type|||outer_struct_type")
                        mlir_type = item[1]
                        target_type = mlir_type.split('|||')[0] if '|||' in mlir_type else mlir_type
                        # Create a type object based on the target type
                        if target_type == 'i64':
                            # Return a wrapper that will generate i64
                            return TLong()
                        elif target_type == 'i32':
                            return TInteger()
                        elif target_type == 'f64':
                            return TFloat()
        # Default: return regular i32 TInteger
        return TInteger()
    
    def print_cpp(self, indent_level=0):
        if self._string is None:
            return ' ' * indent_level * Node.spaces_per_tab + str(self._value)
        else:
            return ' ' * indent_level * Node.spaces_per_tab + str(self._string)
        
    def print_omp(self, indent_level=0):
        if self._string is None:
            return ' ' * indent_level * Node.spaces_per_tab + str(self._value)
        else:
            return ' ' * indent_level * Node.spaces_per_tab + str(self._string)
    
    def print_sycl(self, indent_level=0):
        if self._string is None:
            return ' ' * indent_level * Node.spaces_per_tab + str(self._value)
        else:
            return ' ' * indent_level * Node.spaces_per_tab + str(self._string)
    
    def print_mlir(self, indent_level=0):
        indent = ' ' * indent_level * Node.spaces_per_tab 

        if self._string is None:
            value = str(self._value)
            if self.hoist():
                self.mlir_id(Node.get_mlir_id())
                mlir_str = "".join(Node.context.get_block().pop_mlir())
                mlir_str += indent + f"%{self.mlir_id()} = arith.constant {self._value} : {self.get_type().print_mlir(0)}\n"
                Node.context.push_block()
                Node.context.get_block().mlir_append(mlir_str)
                id = f"%{self.mlir_id()}"
                return id
            return value
        else:
            # NOTE: We use Strings for the CellData structs (patchData).
            #	So, we check the name here and we generate the MLIR to 
            #	extract the struct from LLVM pointer
            # TODO: We need to raise this to where we make the assignment 
            match self._string:
                case x if '.' in x:
                    struct, field = x.split('.')
                    if struct in Node.struct_map:
                        struct_list = Node.struct_map[struct]
                        idx = [item[0] for item in struct_list].index(field)
                        mlir_type = struct_list[idx][1]
                        # Extract just the struct type (everything after the separator)
                        mlir_type_struct = mlir_type.split('|||')[1] if '|||' in mlir_type else mlir_type
                        # getelementptr returns !llvm.ptr and takes struct type as parameter
                        mlir_str = f"llvm.getelementptr %{struct}[0, {idx}] : (!llvm.ptr) -> !llvm.ptr, {mlir_type_struct}"
                        return mlir_str
            return self._string
        
    def print_tree(self, indent_level=0):
        if self._string is None:
            return ' ' * indent_level * Node.spaces_per_tab + "Integer: " + str(self._value)
        else:
            return ' ' * indent_level * Node.spaces_per_tab + "Integer: " + str(self._string)


class Boolean(Expression):
    def __init__(self, value):
        super().__init__()
        self._value = value
    
    def __str__(self):
        return str(self._value)
    
    def get_type(self):
        return TBoolean()

    def print_cpp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + str(self._value)
    
    def print_omp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + str(self._value)

    def print_sycl(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + str(self._value)

    def print_mlir(self, indent_level=0):
        # NOTE: Literals are returned and used in-line, no hoisting required
        return str(self._value) 
    
    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "Boolean: " + str(self._value)

class String(Expression):
    def __init__(self, value):
        super().__init__()
        self._value = value
    
    def __str__(self):
        return self._value

    def get_type(self):
        return TString()
    
    def print_cpp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + self._value
    
    def print_omp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + self._value

    def print_sycl(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + self._value

    def print_mlir(self, indent_level=0):
        indent = ' ' * indent_level * Node.spaces_per_tab 

        match self._value:
            case x if '.' in x:
                struct, field = x.split('.')
                if struct in Node.struct_map:
                    struct_list = Node.struct_map[struct]
                    idx = [item[0] for item in struct_list].index(field)
                    mlir_type_target, mlir_type_struct = struct_list[idx][1].split('|||')

                    # Generate the MLIR operations
                    struct_val_id = f"%{Node.get_mlir_id()}"
                    
                    # TODO: Special case: maxEigenvalue is also a function parameter, so we have to hack it here
                    if field == "maxEigenvalue":
                        field_id = f"%{field}s"
                    else:
                        field_id = f"%{Node.get_mlir_id()}"
                    
                    # Build the basic MLIR operations
                    mlir_str = indent + f"{struct_val_id} = llvm.load %{struct} {{name = \"{struct.strip('%')}\"}}: !llvm.ptr -> {mlir_type_struct}\n"
                    
                    # If the extracted field is a raw pointer (not a descriptor struct),
                    # extract it directly with a _ptr suffix to distinguish from the final memref
                    if not mlir_type_target.startswith("!llvm.struct<"):
                        # For raw pointers (like maxEigenvalue field [11]), use _ptr naming
                        ptr_field_id = f"%{field}{'_ptr' if field != 'maxEigenvalue' else '_ptr'}"
                        mlir_str += indent + f"{ptr_field_id} = llvm.extractvalue {struct_val_id}[{idx}] : {mlir_type_struct}\n"
                        field_id = ptr_field_id
                    else:
                        # For descriptor structs, extract normally (will be processed below)
                        mlir_str += indent + f"{field_id} = llvm.extractvalue {struct_val_id}[{idx}] : {mlir_type_struct}\n"

                    final_id = field_id

                    # If the extracted field is a descriptor struct, convert it to appropriate memref
                    if mlir_type_target.startswith("!llvm.struct<"):
                        # Use getelementptr[0, field_idx] to properly access struct fields
                        ptr_getelementptr_id = f"%{Node.get_mlir_id()}"
                        ptr_load_id = f"%{Node.get_mlir_id()}"
                        mlir_str += indent + f"{ptr_getelementptr_id} = llvm.getelementptr %{struct}[0, {idx}] : (!llvm.ptr) -> !llvm.ptr, {mlir_type_struct}\n"
                        mlir_str += indent + f"{ptr_load_id} = llvm.load {ptr_getelementptr_id} {{name = \"{struct.strip('%')}\"}}: !llvm.ptr -> {mlir_type_target}\n"
                        
                        # Extract all 5 descriptor fields
                        alloc_ptr_id = f"%{Node.get_mlir_id()}"
                        aligned_ptr_id = f"%{Node.get_mlir_id()}"
                        offset_id = f"%{Node.get_mlir_id()}"
                        sizes_0_id = f"%{Node.get_mlir_id()}"
                        strides_0_id = f"%{Node.get_mlir_id()}"
                        
                        mlir_str += indent + f"{alloc_ptr_id} = llvm.extractvalue {ptr_load_id}[0] : {mlir_type_target}\n"
                        mlir_str += indent + f"{aligned_ptr_id} = llvm.extractvalue {ptr_load_id}[1] : {mlir_type_target}\n"
                        mlir_str += indent + f"{offset_id} = llvm.extractvalue {ptr_load_id}[2] : {mlir_type_target}\n"
                        mlir_str += indent + f"{sizes_0_id} = llvm.extractvalue {ptr_load_id}[3, 0] : {mlir_type_target}\n"
                        mlir_str += indent + f"{strides_0_id} = llvm.extractvalue {ptr_load_id}[4, 0] : {mlir_type_target}\n"
                        
                        # Determine target memref type based on descriptor dimensions
                        is_2d = "array<2 x i64>" in mlir_type_target
                        
                        # For 2D descriptors, also extract sizes[1] and strides[1]
                        sizes_1_id = None
                        strides_1_id = None
                        
                        if is_2d:
                            sizes_1_id = f"%{Node.get_mlir_id()}"
                            strides_1_id = f"%{Node.get_mlir_id()}"
                            mlir_str += indent + f"{sizes_1_id} = llvm.extractvalue {ptr_load_id}[3, 1] : {mlir_type_target}\n"
                            mlir_str += indent + f"{strides_1_id} = llvm.extractvalue {ptr_load_id}[4, 1] : {mlir_type_target}\n"
                        
                        # Use type_map to get the correct memref type for this field
                        # For fields like cellCentre, cellSize, t, dt, we need to use their correct types from type_map
                        if field in Node.type_map:
                            type_entry = Node.type_map[field]
                            # Handle both Type objects and strings
                            if isinstance(type_entry, str):
                                target_memref_type = type_entry
                            else:
                                # Type object - call print_mlir()
                                target_memref_type = type_entry.print_mlir(0).strip()
                        else:
                            # Fallback to descriptor-based inference
                            if is_2d:
                                # 2D descriptor
                                target_memref_type = "memref<?x?xf64>"
                            else:
                                # 1D descriptor (assume array<1 x i64>)
                                target_memref_type = "memref<?xf64>"

                        # Create memref descriptor from the extracted fields using the new method
                        conversion_mlir, memref_id = Node.create_memref_from_extracted_descriptor(
                            alloc_ptr_id,
                            aligned_ptr_id,
                            offset_id,
                            sizes_0_id,
                            sizes_1_id,
                            strides_0_id,
                            strides_1_id,
                            target_memref_type,
                            mlir_type_target,
                            indent_level,
                            target_var_name=Node.target_variable_name,
                            is_2d=is_2d)
                        mlir_str += conversion_mlir
                        final_id = memref_id
                    elif 'memref' in mlir_type_target or mlir_type_target == '!llvm.ptr':
                        # For explicit memref types or raw pointers (like maxEigenvalue), 
                        # use create_memref_from_ptr with the extracted pointer (_ptr version)
                        clean_target_type = mlir_type_target.strip('()')
                        # Use the _ptr field_id for raw pointer types
                        conversion_mlir, memref_id = Node.create_memref_from_ptr(
                            field_id,  # This is now %maxEigenvalues_ptr for raw pointers
                            clean_target_type, 
                            indent_level, 
                            target_var_name=Node.target_variable_name)
                        mlir_str += conversion_mlir
                        final_id = memref_id

                    # Always append to context and return variable ID
                    Node.context.get_block().mlir_append(mlir_str)
                    return final_id
        return self._value            
    
    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "String: " + self._value


class Float(Expression):
    def __init__(self, value,  string = None, reference = False):
        super().__init__()
        self._value = value
        self._string = string
        self._reference = reference

    
    def __float__(self):
        return self._value

    def get_type(self):
        return TFloat(self._reference)
    
    def print_cpp(self, indent_level=0):
        if self._string is None:
            return ' ' * indent_level * Node.spaces_per_tab + str(self._value)
        else:
            return ' ' * indent_level * Node.spaces_per_tab + str(self._string)
    
    def print_omp(self, indent_level=0):
        if self._string is None:
            return ' ' * indent_level * Node.spaces_per_tab + str(self._value)
        else:
            return ' ' * indent_level * Node.spaces_per_tab + str(self._string)
    
    def print_sycl(self, indent_level=0):
        if self._string is None:
            return ' ' * indent_level * Node.spaces_per_tab + str(self._value)
        else:
            return ' ' * indent_level * Node.spaces_per_tab + str(self._string)
    
    def print_mlir(self, indent_level=0):
        indent = ' ' * indent_level * Node.spaces_per_tab 

        if self._string is None:
            value = str(self._value)
            if self.hoist():
                self.mlir_id(Node.get_mlir_id())
                mlir_str = "".join(Node.context.get_block().pop_mlir())
                mlir_str += indent + f"%{self.mlir_id()} = arith.constant {self._value} : {self.get_type().print_mlir(0)}\n"
                Node.context.push_block()
                Node.context.get_block().mlir_append(mlir_str)
                id = f"%{self.mlir_id()}"
                return id
            return value
        else:
            return str(self._string) 
    
    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "Float: " + str(self._value)
    

class DataBlock(Expression):
    def __init__(self, iteration_range, internal, requires_memory_allocation, id = None, underlying_type = String("double")):
        super().__init__()
        self._iteration_range = iteration_range
        self._internal = internal
        self.requires_memory_allocation = requires_memory_allocation
        self._underlying_type = underlying_type
        self.id = id
        self._offset = []
        self._memory_range = []
        for [x, y] in self._iteration_range:
            self._offset.append(Integer(0))
            self._memory_range.append([Integer(0), y - x])
    
    def offset(self, offset):
        output = DataBlock(copy.deepcopy(self._iteration_range), copy.deepcopy(self._internal), True)
        output.id = self.id

        for i in range(len(offset)):
            if type(offset[i]) is list:
                if offset[i][0] is None:
                    offset[i][0] = self._iteration_range[i][0]
                if offset[i][1] is None:
                    offset[i][1] = self._iteration_range[i][1]

                output._offset[i] = offset[i][0] - self._iteration_range[i][0]
                output._iteration_range[i][1] = offset[i][1]
                output._iteration_range[i][0] = offset[i][0]
            else:
                raise Exception("Invalid index")
        
        for i in range(len(output._offset), len(output._iteration_range)):
            output._offset.append(Integer(0))

        return output

    def linearise_index(self, indices, dimensions, offset_start_index = 0):
        factors = [Integer(1)]
        for i in range(0, len(dimensions) - 1):
            factors.append(factors[-1] * (self._memory_range[i][1]))
        if Node.use_accelerator == False and type(self) is not FaceDataBlock:
            index_list = indices[0:-1]
        else:
            index_list = indices
            
        if type(dimensions[-len(index_list)][1] - dimensions[-len(index_list)][0]) is Integer and (dimensions[-len(index_list)][1] - dimensions[-len(index_list)][0])._value == 1:
            index = self._offset[offset_start_index]
        else:
            if type(self._offset[offset_start_index + 0].get_type()) is TDataBlock:
                index = index_list[0] + self._offset[offset_start_index + 0].multidimensional_index(index_list, offset_start_index)
            else:
                index = index_list[0] + self._offset[offset_start_index + 0]

        index = factors[-len(index_list)] * index
        for i in range(1, len(index_list)):
            if type(self._offset[offset_start_index + i].get_type()) is TDataBlock:
                temp = index_list[i] + self._offset[offset_start_index + i].multidimensional_index(indices, offset_start_index)
            else:
                temp = index_list[i] + self._offset[offset_start_index + i]
            index = index + factors[-len(index_list) + i] * temp
        return index
    
    def multidimensional_index(self, index_list, start_index = 0, dimensions = []):
        if len(self._iteration_range) == 1:
            return Subscript(self, index_list[-1])
        else:
            if Node.use_accelerator == False:
                return Subscript(Subscript(self, index_list[-1]), self.linearise_index(index_list, self._iteration_range[0:-1], start_index))
            else:
                return Subscript(self, self.linearise_index(index_list, self._iteration_range, start_index))

    def get_type(self):
        db_type = TDataBlock(len(self._iteration_range), self._underlying_type)
        return db_type
    
    def print_cpp(self, indent_level=0):
        if self.id is None:
            return ' ' * indent_level * Node.spaces_per_tab + self._internal.print_cpp()
        else:
            return ' ' * indent_level * Node.spaces_per_tab + self.id
        
    def print_omp(self, indent_level=0):
        if self.id is None:
            return ' ' * indent_level * Node.spaces_per_tab + self._internal.print_omp()
        else:
            return ' ' * indent_level * Node.spaces_per_tab + self.id
    
    def print_sycl(self, indent_level=0):
        if self.id is None:
            return ' ' * indent_level * Node.spaces_per_tab + self._internal.print_sycl()
        else:
            return ' ' * indent_level * Node.spaces_per_tab + self.id
    

    def print_mlir(self, indent_level=0):
        if self.id is None:
            return self._internal.print_mlir(indent_level)
        else:
            return "%" + self.id
        
    def print_tree(self, indent_level=0):
        if self.id is None:
            return ' ' * indent_level * Node.spaces_per_tab + f"""DataBlock:
{self._internal.print_tree(indent_level + 1)}"""
        else:
            return ' ' * indent_level * Node.spaces_per_tab + "DataBlock: " + self.id

"""
FaceDataBlock distinguishes itself from normal DataBlocks by the fact that its internal array is always 1d.
"""
class FaceDataBlock(DataBlock):
    def __init__(self, iteration_range, internal, requires_memory_allocation, id = None, underlying_type = String("double")):
        super().__init__(iteration_range, internal, requires_memory_allocation, id, underlying_type)

    def offset(self, offset):
        output = FaceDataBlock(copy.deepcopy(self._iteration_range), copy.deepcopy(self._internal), True)
        output.id = self.id

        for i in range(len(offset)):
            if type(offset[i]) is list:
                if offset[i][0] is None:
                    offset[i][0] = self._iteration_range[i][0]
                if offset[i][1] is None:
                    offset[i][1] = self._iteration_range[i][1]

                output._offset[i] = offset[i][0] - self._iteration_range[i][0]
                output._iteration_range[i][1] = offset[i][1]
                output._iteration_range[i][0] = offset[i][0]
            else:
                raise Exception("Invalid index")
        
        for i in range(len(output._offset), len(output._iteration_range)):
            output._offset.append(Integer(0))

        return output
    
    def multidimensional_index(self, index_list, start_index = 0, dimensions = []):
        return Subscript(self, self.linearise_index(index_list, self._iteration_range, start_index))
    

class LogDataBlock(Statement):
    def __init__(self, dataBlock, filename):
        self._filename = filename

        self._dataBlock = dataBlock

        loops = []
        index = []

        loops.append(For([Integer(0), self._dataBlock._iteration_range[-1][1], - self._dataBlock._iteration_range[-1][0]]))
        for i in range(len(self._dataBlock._memory_range) - 2, -1, -1):
            loops.append(For([Integer(0), self._dataBlock._iteration_range[i][1] - self._dataBlock._iteration_range[i][0]]))
       
        for i in range(len(self._dataBlock._memory_range) - 1, -1, -1):
            index.append(loops[i].get_iteration_variable())                            

        loops[-1].add_statement(LogToFile(self._dataBlock.multidimensional_index(index), String("log")))
        loops[-1].add_statement(LogToFile(String("\" \""), String("log")))
        for i in range(0, len(loops) - 1):
            loops[i].add_statement(loops[i + 1])

        loops[-2].add_statement(LogToFile(String("\", \""), String("log")))
        for i in range(0, len(self._dataBlock._memory_range) - 2):
            loops[i].add_statement(LogToFile(String("\"\\n\""), String("log")))

        for loop in loops[::-1]:
            loop.close_scope()
        self._loop = loops[0]
    
    def print_cpp(self, indent_level=0):
        return f"""log.open("{self._filename.print_cpp()}");
{self._loop.print_cpp()}
log.close();
"""
    
    def print_omp(self, indent_level=0):
        return ""

    def print_sycl(self, indent_level=0):
        return ""

    def print_mlir(self, indent_level=0):
        return ""

    def print_tree(self, indent_level=0):
        return ""

class Subscript(Expression):
    def __init__(self, value: Expression, index: Expression):
        super().__init__()
        self._value = value
        self._index = index
    
    def __neg__(self):
        return UnaryOperation("-", self)

    def is_nested_subscript(self):
        return isinstance(self._value, Subscript)

    def get_type(self):
        return self._value.get_type()

    def print_cpp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + f"{self._value.print_cpp()}[{self._index.print_cpp()}]"
    
    def print_omp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + f"{self._value.print_omp()}[{self._index.print_omp()}]"

    def print_sycl(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + f"{self._value.print_sycl()}[{self._index.print_sycl()}]"

    def print_mlir(self, indent_level=0):
        indent = ' ' * indent_level * Node.spaces_per_tab
        dims = 1
        tmp = self._value
        while isinstance(tmp, Subscript):
            dims += 1
            tmp = tmp._value

        self._dimensions = dims
        value_type = self._value.get_type()
        type_str = value_type.print_cpp().strip()

        mlir_type = Node.type_map.get(type_str)
        if mlir_type is None:
            # Fallback: try to get the MLIR type from the type object itself, or use a default
            mlir_type = '' + value_type.print_mlir(0)
                
        mlir_str = "".join(Node.context.get_block().pop_mlir())

        self._index.hoist(True)
        index = self._index.print_mlir(indent_level)
        mlir_str += "".join(Node.context.get_block().pop_mlir())

        if not getattr(self._index, '_index_type', False):
            new_index = f"%{Node.get_mlir_id()}"
            # Determine the actual type of the index expression
            index_expr_type = self._index.get_type()
            mlir_type_cast = index_expr_type.print_mlir(0).strip()
            mlir_str += indent + new_index + f" = arith.index_cast {index.strip(' ')}: {mlir_type_cast} to index\n"
            self._new_index = new_index
        else:
            self._new_index = index

        # NOTE: special handling: Strings might represent struct field access
        if isinstance(self._value, String):
            # When indexing a String field (like patchData.input[i]), we need to handle the index
            # Get the field type to determine if it's 2D or 1D
            base_mlir = self._value.print_mlir(indent_level)
            base_type = self._value.get_type()
            
            if isinstance(base_type, TDataBlock):
                memref_type = base_type.print_mlir(0).strip()
                
                # Check if it's a 2D flat or 1D nested memref
                if "x?x" in memref_type:
                    # 2D flat memref - but we only have one index, which means we need a second one
                    # This shouldn't happen for single-level subscripts on 2D arrays
                    result_id = f"%{Node.get_mlir_id()}"
                    mlir_str += indent + f"{result_id} = memref.load {base_mlir}[%0, {self._new_index}] {{name = \"{base_mlir.strip('%')}\"}} : {memref_type}\n"
                    Node.context.push_block()
                    Node.context.get_block().mlir_append(mlir_str)
                    return result_id
                else:
                    # 1D memref - can be nested (memref<?xmemref<?xf64>>) or plain (memref<?xf64>)
                    result_id = f"%{Node.get_mlir_id()}"
                    mlir_str += indent + f"{result_id} = memref.load {base_mlir}[{self._new_index}] {{name = \"{base_mlir.strip('%')}\"}} : {memref_type}\n"
                    Node.context.push_block()
                    Node.context.get_block().mlir_append(mlir_str)
                    return result_id
            else:
                # Not a memref type, just return the base
                return base_mlir

        Node.context.push_block()
        Node.context.get_block().mlir_append(mlir_str)

        self._value.hoist(True)
        base_value = self._value.print_mlir(indent_level)

        # Handle nested subscripts (like source[i][j]) by generating proper load operations
        if self.is_nested_subscript():
            result_id = f"%{Node.get_mlir_id()}"

            base_expr = tmp  # The base expression without subscripts
            outer_base = base_expr.print_mlir(indent_level)  # The base array
            
            # Get the actual type of the base expression - check mlir_memref_types FIRST
            if outer_base in Node.mlir_memref_types:
                actual_memref_type = Node.mlir_memref_types[outer_base]
            else:
                # Fallback to computing from type
                base_expr_type = base_expr.get_type()
                if isinstance(base_expr_type, TDataBlock):
                    actual_memref_type = base_expr_type.print_mlir(0).strip()
                else:
                    actual_memref_type = "memref<?x?xf64>"  # Fallback
            
            # Check if base expression has an explicit type in type_map (nested memref override)
            base_name = None
            if isinstance(base_expr, Name):
                base_name = base_expr.id
            elif hasattr(base_expr, 'id'):
                # DataBlock or similar object with id attribute
                base_name = base_expr.id
            
            if base_name and base_name in Node.type_map and _is_nested_memref(Node.type_map[base_name]):
                actual_memref_type = Node.type_map[base_name]
            
            # Collect both indices from the nested structure
            indices = []
            current = self
            while isinstance(current, Subscript):
                indices.insert(0, current._new_index)
                current = current._value
            
            # Generate load based on memref type structure
            indices_str = ", ".join(indices)
            mlir_str = "".join(Node.context.get_block().pop_mlir())
            
            # Check if it's a nested memref (memref<?xmemref<...>>) vs flat multi-D (memref<?x?x...>)
            # Handle both Type objects and strings
            if isinstance(actual_memref_type, str):
                is_nested = actual_memref_type.count("memref<") > 1
                actual_memref_type_str = actual_memref_type
            else:
                # Type object (e.g., TNestedMemref)
                is_nested = actual_memref_type.is_nested_memref()
                actual_memref_type_str = actual_memref_type.print_mlir(0).strip()
            
            if is_nested and len(indices) >= 2:
                # This is a nested memref (memref<?xmemref<?xf64>>)
                # Need to load in two steps: first get the inner memref, then load from it
                inner_memref_id = f"%{Node.get_mlir_id()}"
                # Extract the inner memref type from the outer type
                if actual_memref_type_str.startswith("memref<?x") and actual_memref_type_str.endswith(">>"):
                    inner_type = actual_memref_type_str[9:-1]  # Remove "memref<?x" and final ">"
                else:
                    inner_type = "memref<?xf64>"  # Fallback
                
                mlir_str += indent + f"{inner_memref_id} = memref.load {outer_base}[{indices[0]}] {{name = \"{outer_base.strip('%')}\"}} : {actual_memref_type_str}\n"
                # Now load from the inner memref with remaining indices
                remaining_indices = ", ".join(indices[1:])
                mlir_str += indent + f"{result_id} = memref.load {inner_memref_id}[{remaining_indices}] {{name = \"{outer_base.strip('%')}\"}} : {inner_type}\n"
            else:
                # This is a flat multi-D memref (memref<?x?xf64>, memref<?x?x?xf64>, etc.)
                mlir_str += indent + f"{result_id} = memref.load {outer_base}[{indices_str}] {{name = \"{outer_base.strip('%')}\"}} : {actual_memref_type_str}\n"

            Node.context.push_block()
            Node.context.get_block().mlir_append(mlir_str)

            return result_id

        # Default: just return the base value (may be a memref, pointer, or value)
        return base_value
        
    def print_tree(self, indent_level=0):
        return f"""Subscript:
{self._value.print_tree(indent_level + 1)}
{self._index.print_tree(indent_level + 1)}"""


class Reference(Expression):
    def __init__(self, expression: Expression):
        self._expression = expression
    
    def get_type(self):
        return self._expression.get_type()
    
    def print_cpp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "&" + self._expression.print_cpp()
    
    def print_omp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "&" + self._expression.print_omp()

    def print_sycl(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "&" + self._expression.print_sycl()

    def print_mlir(self, indent_level=0):
        result_id = None
        indent = ' ' * indent_level * Node.spaces_per_tab
        mlir_str = "".join(Node.context.get_block().pop_mlir())
        
        self._expression.hoist(True)
               
        # Handle references to subscripted expressions (array elements)
        if isinstance(self._expression, Subscript):
            base_expr = self._expression._value
            index_expr = self._expression._index

            base_expr.hoist(True)
            base_mlir = base_expr.print_mlir(indent_level)
            
            mlir_str += "".join(Node.context.get_block().pop_mlir())
            
            if not hasattr(self._expression, '_new_index'):
                # If _new_index doesn't exist, process index and generate it now
                index_expr.hoist(True)
                index_mlir = index_expr.print_mlir(indent_level)
                mlir_str += "".join(Node.context.get_block().pop_mlir())
                
                # Create new index with proper type cast
                new_index = f"%{Node.get_mlir_id()}"
                mlir_type = ('index' if hasattr(index_expr, '_index_type') and index_expr._index_type else 'i32')
                mlir_str += indent + new_index + f" = arith.index_cast {index_mlir.strip(' ')}: {mlir_type} to index\n"
                self._expression._new_index = new_index
            else:
                new_index = self._expression._new_index
            
            # Determine dimensions and type of the memref
            base_type = base_expr.get_type()          
            if isinstance(base_type, TDataBlock):
                result_id = f"%{Node.get_mlir_id()}"
                
                # For nested memrefs (like double**), handle the inner memref properly
                if isinstance(base_expr, Subscript):
                    # base_expr is Subscript[DataBlock, i] 
                    # We need to check if the base_expr itself is a nested subscript or just single-level
                    if isinstance(base_expr._value, Subscript):
                        result_id = base_mlir
                    else:
                        # Check if this variable has an explicit nested type in `type_map`
                        base_mlir_name = base_mlir.lstrip('%')
                        is_nested_memref = False
                        if base_mlir_name in Node.type_map and _is_nested_memref(Node.type_map[base_mlir_name]):
                            is_nested_memref = True
                        
                        outer_index = base_expr._new_index
                        
                        # Use unified method from Node to generate pointer arithmetic
                        ptr_code, result_id = Node.generate_pointer_to_2d_element(
                            base_mlir, outer_index, new_index, is_nested_memref, indent_level
                        )

                        mlir_str += ptr_code
                else:
                    element_type = "f64"  # Default to f64 - modify if needed for other types
                    mlir_str += indent + f"{result_id} = memref.subview {base_mlir}[{new_index}][1][1] : memref<?x{element_type}> to memref<?x{element_type}>\n"
        
        # Default behavior for non-subscript references
        Node.context.push_block()
        Node.context.get_block().mlir_append(mlir_str)
        
        # Process the expression and return its ID
        return result_id if result_id is not None else self._expression.print_mlir(indent_level)
    

    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + f"""Reference:
{self._expression.print_tree(indent_level + 1)}"""


class Vector(Expression):
    def __init__(self, elements, element_type, string = None):
        self._elements = elements
        self._type = element_type
        self._string = string

    def print_cpp(self, indent_level=0):
        if self._string is None:
            element_prints = [element.print_cpp() for element in self._elements]
            return ' ' * indent_level * Node.spaces_per_tab + f"""tarch::la::Vector<{len(self._elements)}, {self._type.print_cpp()}>{{{", ".join(element_prints)}}}"""
        return self._string
    
    def print_omp(self, indent_level=0):
        element_prints = [element.print_cpp() for element in self._elements]
        return ' ' * indent_level * Node.spaces_per_tab + f"""tarch::la::Vector<{len(self._elements)}, {self._type.print_cpp()}>{{{", ".join(element_prints)}}}"""
    
    def print_sycl(self, indent_level=0):
        element_prints = [element.print_cpp() for element in self._elements]
        return ' ' * indent_level * Node.spaces_per_tab + f"""tarch::la::Vector<{len(self._elements)}, {self._type.print_cpp()}>{{{", ".join(element_prints)}}}"""
    
    def print_mlir(self, indent_level=0):
        pass

    def print_tree(self, indent_level=0):
        return "Vector"
    
    def get_type(self):
        if self._string is not None:
            return TCustom("auto")
        return TCustom(f"tarch::la::Vector<{len(self._elements)}, {self._type.print_cpp()}>")


class DiagonalMatrix(Expression):
    def __init__(self, width, internal, index_dimensions):
        self._internal = internal
        self._width = width
        self._index_dimensions = index_dimensions
        self._id = None

    def multidimensional_index(self, index, start_index = 0, dimensions = []):
        diagonal_index = index[1]
        factor = Integer(1)
        for i in range(2, len(index) - 1):
            factor = factor * (dimensions[i - 1][1] - dimensions[i - 1][0])
            diagonal_index = diagonal_index + factor * index[i]
        return Subscript(self, diagonal_index)

    def print_cpp(self, indent_level=0):
        if self._id is not None:
            return self._id.print_cpp(indent_level)
        else:
            return self._internal.print_cpp(indent_level)

    def print_omp(self, indent_level=0):
        if self._id is not None:
            return self._id.print_cpp(indent_level)
        else:
            return self._internal.print_cpp(indent_level)

    def print_sycl(self, indent_level=0):
        pass

    def print_mlir(self, indent_level=0):
        pass

    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "DiagonalMatrix"

    def get_type(self):
        return TDiagonalMatrix()


class Matrix(Expression):
    def __init__(self, dimensions, internal, index_dimensions):
        self._internal = internal
        self._dimensions = dimensions
        self._index_dimensions = index_dimensions
        self._id = None

    def index(self, row_index, column_index):
        return Subscript(self, row_index * self._dimensions[1] + column_index)

    def print_cpp(self, indent_level=0):
        if self._id is not None:
            return self._id.print_cpp(indent_level)
        else:
            return self._internal.print_cpp(indent_level)

    def print_omp(self, indent_level=0):
        if self._id is not None:
            return self._id.print_cpp(indent_level)
        else:
            return self._internal.print_cpp(indent_level)

    def print_sycl(self, indent_level=0):
        pass

    def print_mlir(self, indent_level=0):
        pass

    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "Matrix"

    def get_type(self):
        return TMatrix()
    

class DiagonalMatrixConstructionFromKroneckerProduct(Expression):
    def __init__(self, id, matrix):
        self._id = String(id)
        if type(matrix) is Name:
            self._matrix = Name.variables[matrix.id]
        else:
            self._matrix = matrix

        self._memory_allocation = MemoryAllocation(self._id, TFloat(), [self._matrix._width])

        lhs_matrix = self._matrix._internal._arguments[0]
        if type(lhs_matrix) is Name:
            lhs_matrix = Name.variables[lhs_matrix.id]

        rhs_matrix = self._matrix._internal._arguments[1]
        if type(rhs_matrix) is Name:
            rhs_matrix = Name.variables[rhs_matrix.id]

        self._assignment = For([Integer(0), lhs_matrix._width])
        internal_loop = For([Integer(0), rhs_matrix._width])
        index = self._assignment.get_iteration_variable() * rhs_matrix._width + internal_loop.get_iteration_variable()
        lhs_index = Subscript(lhs_matrix, self._assignment.get_iteration_variable())
        if type(lhs_matrix._internal) is Float or type(lhs_matrix._internal) is Integer:
            lhs_index = lhs_matrix._internal
        rhs_index = Subscript(rhs_matrix, internal_loop.get_iteration_variable())
        if type(rhs_matrix._internal) is Float or type(rhs_matrix._internal) is Integer:
            rhs_index = rhs_matrix._internal
        internal_loop.add_statement(Assignment(Subscript(self._id, index), BinaryOperation("*", lhs_index, rhs_index)))
        self._assignment.add_statement(internal_loop)
        internal_loop.close_scope()
        self._assignment.close_scope()

    def print_cpp(self, indent_level=0):
        return f"""{self._memory_allocation.print_cpp(indent_level)}
{self._assignment.print_cpp(indent_level)}"""

    def print_omp(self, indent_level=0):
        return f"""{self._memory_allocation.print_omp(indent_level)}
{self._assignment.print_omp(indent_level)}"""

    def print_sycl(self, indent_level=0):
        pass

    def print_mlir(self, indent_level=0):
        pass

    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "DiagonalKroneckerProduct"

    def get_type(self):
        return TDiagonalMatrix()


class DiagonalMatrixConstructionFromValue(Expression):
    def __init__(self, id, matrix):
        self._id = String(id)
        if type(matrix) is Name:
            self._matrix = Name.variables[matrix.id]
        else:
            self._matrix = matrix

        self._memory_allocation = MemoryAllocation(self._id, TFloat(), [self._matrix._width])

        self._assignment = For([Integer(0), self._matrix._width])
        self._assignment.add_statement(Assignment(Subscript(self._id, self._assignment.get_iteration_variable()), self._matrix._internal))
        self._assignment.close_scope()

    def print_cpp(self, indent_level=0):
        return f"""{self._memory_allocation.print_cpp(indent_level)}
{self._assignment.print_cpp(indent_level)}"""

    def print_omp(self, indent_level=0):
        return f"""{self._memory_allocation.print_omp(indent_level)}
{self._assignment.print_omp(indent_level)}"""

    def print_sycl(self, indent_level=0):
        pass

    def print_mlir(self, indent_level=0):
        pass

    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "DiagonalKroneckerProduct"

    def get_type(self):
        return TDiagonalMatrix()


class MatrixConstructionFromKroneckerProduct(Expression):
    def __init__(self, id, matrix):
        self._id = String(id)
        if type(matrix) is Name:
            self._matrix = Name.variables[matrix.id]
        else:
            self._matrix = matrix

        self._memory_allocation = MemoryAllocation(self._id, TFloat(), [self._matrix._dimensions[0] * self._matrix._dimensions[1]])

        self._initialisation = For([Integer(0), self._matrix._dimensions[0] * self._matrix._dimensions[1]])
        self._initialisation.add_statement(Assignment(Subscript(self._id, self._initialisation.get_iteration_variable()), Float(0.0)))
        self._initialisation.close_scope()

        lhs_matrix = self._matrix._internal._arguments[0]
        if type(lhs_matrix) is Name:
            lhs_matrix = Name.variables[lhs_matrix.id]

        rhs_matrix = self._matrix._internal._arguments[1]
        if type(rhs_matrix) is Name:
            rhs_matrix = Name.variables[rhs_matrix.id]

        if type(lhs_matrix.get_type()) is TMatrix:
            lhs_dimensions = lhs_matrix._dimensions
        else:
            lhs_dimensions = [lhs_matrix._width, lhs_matrix._width]

        if type(rhs_matrix.get_type()) is TMatrix:
            rhs_dimensions = rhs_matrix._dimensions
        else:
            rhs_dimensions = [rhs_matrix._width, rhs_matrix._width]

        loops = []
        if type(lhs_matrix.get_type()) is TMatrix:
            loops.append(For([Integer(0), lhs_matrix._dimensions[0]]))
            loops.append(For([Integer(0), lhs_matrix._dimensions[1]]))
            lhs_index = loops[0].get_iteration_variable() * lhs_dimensions[1] + loops[1].get_iteration_variable()
            startIndex = loops[0].get_iteration_variable() * lhs_dimensions[1] * rhs_dimensions[1] * rhs_dimensions[0] + loops[1].get_iteration_variable() * rhs_dimensions[1]
        else:
            loops.append(For([Integer(0), lhs_matrix._width]))
            lhs_index = loops[0].get_iteration_variable()
            startIndex = loops[0].get_iteration_variable() * lhs_dimensions[1] * rhs_dimensions[1] * rhs_dimensions[0] + loops[0].get_iteration_variable() * rhs_dimensions[1]

        if type(rhs_matrix.get_type()) is TMatrix:
            loops.append(For([Integer(0), rhs_matrix._dimensions[0]]))
            loops.append(For([Integer(0), rhs_matrix._dimensions[1]]))
            rhs_index = loops[-2].get_iteration_variable() * rhs_dimensions[1] + loops[-1].get_iteration_variable()
            index = startIndex + loops[-1].get_iteration_variable() + loops[-2].get_iteration_variable() * lhs_dimensions[1] * rhs_dimensions[1]
        else:
            loops.append(For([Integer(0), rhs_matrix._width]))
            rhs_index = loops[-1].get_iteration_variable()
            index = startIndex + loops[-1].get_iteration_variable() + loops[-1].get_iteration_variable() * lhs_dimensions[1] * rhs_dimensions[1]

        lhs_subscript = Subscript(lhs_matrix, lhs_index)
        if type(lhs_matrix._internal) is Float or type(lhs_matrix._internal) is Integer:
            lhs_subscript = lhs_matrix._internal
        rhs_subscript = Subscript(rhs_matrix, rhs_index)
        if type(rhs_matrix._internal) is Float or type(rhs_matrix._internal) is Integer:
            rhs_subscript = rhs_matrix._internal

        loops[-1].add_statement(Assignment(Subscript(self._id, index), BinaryOperation("*", lhs_subscript, rhs_subscript)))

        for i in range(len(loops) - 1):                    
            loops[i].add_statement(loops[i + 1])
            loops[i].close_scope()
        loops[-1].close_scope()
        self._assignment = loops[0]

    def print_cpp(self, indent_level=0):
        return f"""{self._memory_allocation.print_cpp(indent_level)}
{self._initialisation.print_cpp(indent_level)}
{self._assignment.print_cpp(indent_level)}"""

    def print_omp(self, indent_level=0):
        return f"""{self._memory_allocation.print_omp(indent_level)}
{self._initialisation.print_omp(indent_level)}
{self._assignment.print_omp(indent_level)}"""

    def print_sycl(self, indent_level=0):
        pass

    def print_mlir(self, indent_level=0):
        pass

    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "KroneckerProduct"

    def get_type(self):
        return TMatrix()


class VectorIndex(Expression):
    def __init__(self, value: Expression, index: Expression):
        super().__init__()
        self._value = value
        self._index = index
    
    def get_type(self):
        return self._value.get_type()
    
    def print_cpp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + f"{self._value.print_cpp()}({self._index.print_cpp()})"

    def print_omp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + f"{self._value.print_omp()}({self._index.print_omp()})"

    def print_sycl(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + f"{self._value.print_sycl()}({self._index.print_sycl()})"

    def print_mlir(self, indent_level=0):
        indent = ' ' * indent_level * Node.spaces_per_tab
        mlir_str = ""

        if isinstance(self._value, Subscript):
            array_base = self._value._value
            array_index = self._value._index
            array_base_mlir = array_base.print_mlir(indent_level)
            array_index_mlir = array_index.print_mlir(indent_level)
            # Ensure array_index is of type 'index' with correct source type
            if not getattr(array_index, '_index_type', False):
                array_index_cast = f"%{Node.get_mlir_id()}"
                array_index_type = array_index.get_type()
                array_index_mlir_type = array_index_type.print_mlir(0).strip()
                mlir_str += indent + f"{array_index_cast} = arith.index_cast {array_index_mlir} : {array_index_mlir_type} to index\n"
            else:
                array_index_cast = array_index_mlir

        raw_ptr = f"%{Node.get_mlir_id()}"
        mlir_str += indent + f"{raw_ptr} = memref.load {array_base_mlir}[{array_index_cast}] {{name = \"{array_base_mlir.strip('%')}\"}} : memref<?xi64>\n"

        vec_ptr = f"%{Node.get_mlir_id()}"
        mlir_str += indent + f"{vec_ptr} = llvm.inttoptr {raw_ptr} {{name = \"{array_base_mlir.strip('%')}\"}} : i64 to !llvm.ptr\n"

        func_param_mlir = self._index.print_mlir(indent_level)
        # Ensure func_param is i32 (emit arith.constant for Integer, cast otherwise)
        if isinstance(self._index, Integer):
            func_param_i32 = f"%{Node.get_mlir_id()}"
            mlir_str += indent + f"{func_param_i32} = arith.constant {int(self._index._value)} : i32\n"
        elif not getattr(self._index, '_index_type', False):
            func_param_i32 = f"%{Node.get_mlir_id()}"
            mlir_str += indent + f"{func_param_i32} = arith.index_cast {func_param_mlir} : index to i32\n"
        else:
            func_param_i32 = func_param_mlir

        result = f"%{Node.get_mlir_id()}"
        # Convert i32 index to i64 for pointer arithmetic
        func_param_i64 = f"%{Node.get_mlir_id()}"
        mlir_str += indent + f"{func_param_i64} = arith.extsi {func_param_i32} : i32 to i64\n"
        # Compute pointer to array element
        velem_ptr = f"%{Node.get_mlir_id()}"
        mlir_str += indent + f"{velem_ptr} = llvm.getelementptr {vec_ptr}[{func_param_i64}] {{name = \"{array_base_mlir.strip('%')}\"}} : (!llvm.ptr, i64) -> !llvm.ptr, f64\n"
        # Load the value from the computed pointer
        mlir_str += indent + f"{result} = llvm.load {velem_ptr} {{name = \"{array_base_mlir.strip('%')}\"}} : !llvm.ptr -> f64\n"
        Node.context.get_block().mlir_append(mlir_str)

        return result
        
    def print_tree(self, indent_level=0):
        return f"""
VectorIndex: 
{self.id}
{self._value.print_tree(indent_level + 1)}
{self._index.print_tree(indent_level + 1)}"""
    

class Comparison(Expression):
    def __init__(self, operation, lhs, rhs):
        self._operation = operation
        self._lhs = lhs
        self._rhs = rhs

    def get_type(self):
        return self._lhs.get_type()

    def print_cpp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + f"({self._lhs.print_cpp()} {self._operation} {self._rhs.print_cpp()})"
    
    def print_omp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + f"({self._lhs.print_omp()} {self._operation} {self._rhs.print_omp()})"
    
    def print_sycl(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + f"({self._lhs.print_sycl()} {self._operation} {self._rhs.print_sycl()})"
    
    @staticmethod
    def get_mlir_comparison_op(operation, op_type_class):
        if op_type_class is TFloat:
            op_map = {
                "<": "arith.cmpf olt",
                ">": "arith.cmpf ogt",
                "<=": "arith.cmpf ole",
                ">=": "arith.cmpf oge", 
                "==": "arith.cmpf oeq",
                "!=": "arith.cmpf one"
            }
        else:  # TInteger or default
            op_map = {
                "<": "arith.cmpi slt",
                ">": "arith.cmpi sgt", 
                "<=": "arith.cmpi sle",
                ">=": "arith.cmpi sge",
                "==": "arith.cmpi eq",
                "!=": "arith.cmpi ne"
            }
        return op_map.get(operation, None)
    
    def print_mlir(self, indent_level = 0):
        op_type_class = TDataBlock.get_mlir_type_for_operation(self._lhs, self._rhs)
        
        mlir_operation = Comparison.get_mlir_comparison_op(self._operation, op_type_class)
        result_type = "i1"  # Comparisons always return boolean
        
        if mlir_operation is None:
            raise ValueError(f"Unsupported comparison operation: {self._operation}")
        
        const_zero_id = None
        
        lhs_mlir = self._lhs.print_mlir(indent_level)
        if TDataBlock.needs_memref_load(self._lhs):
            const_zero_id = TDataBlock.ensure_const_zero(const_zero_id, indent_level)
            lhs_mlir = TDataBlock.generate_memref_load(self._lhs, lhs_mlir, const_zero_id, indent_level)
            
        rhs_mlir = self._rhs.print_mlir(indent_level)
        if TDataBlock.needs_memref_load(self._rhs):
            const_zero_id = TDataBlock.ensure_const_zero(const_zero_id, indent_level)
            rhs_mlir = TDataBlock.generate_memref_load(self._rhs, rhs_mlir, const_zero_id, indent_level)

        return ' ' * indent_level * Node.spaces_per_tab + f"""{mlir_operation} {lhs_mlir}, {rhs_mlir} : {result_type}"""

    def print_tree(self, indent_level=0):
        return f"""
Comparison {self._operation}:
{self._lhs.print_tree(indent_level + 1)}
{self._rhs.print_tree(indent_level + 1)}"""

class BinaryOperation(Expression):
    def __init__(self, operation, lhs, rhs):
        super().__init__()
        self._operation = operation
        self._lhs = lhs
        self._rhs = rhs 

        if type(self._lhs) is BinaryOperation:
            if self._lhs._operation == "+":
                if self._lhs._rhs == Integer(0):
                    self._lhs = self._lhs._lhs
                elif self._lhs._lhs == Integer(0):
                    self._lhs = self._lhs._rhs
            elif self._lhs._operation == "-":
                if self._lhs._rhs == Integer(0):
                    self._lhs = self._lhs._lhs
            elif self._lhs._operation == "*":
                if self._lhs._rhs == Integer(1):
                    self._lhs = self._lhs._lhs
                elif self._lhs._lhs == Integer(1):
                    self._lhs = self._lhs._rhs

        if type(self._rhs) is BinaryOperation:
            if self._rhs._operation == "+":
                if self._rhs._rhs == Integer(0):
                    self._rhs = self._rhs._lhs
                elif self._rhs._lhs == Integer(0):
                    self._rhs = self._rhs._rhs
            elif self._rhs._operation == "-":
                if self._rhs._rhs == Integer(0):
                    self._rhs = self._rhs._lhs
            elif self._rhs._operation == "*":
                if self._rhs._rhs == Integer(1):
                    self._rhs = self._rhs._lhs
                elif self._rhs._lhs == Integer(1):
                    self._rhs = self._rhs._rhs

        if self._operation == "-" and type(self._rhs) is UnaryOperation and self._rhs._operation == "-":
            self._operation = "+"
            self._rhs = self._rhs._operand
        if self._operation == "+" and type(self._rhs) is UnaryOperation and self._rhs._operation == "-":
            self._operation = "-"
            self._rhs = self._rhs._operand

    def index(self, index):
        return BinaryOperation(self._operation, self._lhs.index(index), self._rhs.index(index))

    def multidimensional_index(self, index, start_index = 0, dimensions = []):
        return BinaryOperation(self._operation, self._lhs.multidimensional_index(index, start_index, dimensions), self._rhs.multidimensional_index(index, start_index, dimensions))


    def __add__(self, rhs):
        return BinaryOperation("+", self, rhs)

    def __sub__(self, rhs):
        return BinaryOperation("-", self, rhs)
    
    def __mul__(self, rhs):
        return BinaryOperation("*", self, rhs)

    def __neg__(self):
        return UnaryOperation("-", self)

    def get_type(self):
        # For arithmetic operations, return the element type, not the pointer type
        op_type_class = TDataBlock.get_mlir_type_for_operation(self._lhs, self._rhs)
        return op_type_class()

    def print_cpp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + f"({self._lhs.print_cpp()} {self._operation} {self._rhs.print_cpp()})"
    
    def print_omp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + f"({self._lhs.print_omp()} {self._operation} {self._rhs.print_omp()})"
    
    def print_sycl(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + f"({self._lhs.print_sycl()} {self._operation} {self._rhs.print_sycl()})"
    
    @staticmethod
    def get_mlir_arithmetic_op(operation, op_type_class):
        if op_type_class is TFloat:
            op_map = {
                "*": "arith.mulf",
                "/": "arith.divf", 
                "+": "arith.addf",
                "-": "arith.subf"
            }
        else:  # TInteger, TLong, or default
            op_map = {
                "*": "arith.muli",
                "/": "arith.divi",
                "+": "arith.addi", 
                "-": "arith.subi"
            }
        return op_map.get(operation, None)
    
    @staticmethod
    def cast_operand(operand_mlir, operand, result_type, indent_level):
        indent = ' ' * indent_level * Node.spaces_per_tab
        operand_type = operand.get_type()
                
        if type(operand_type) is TDataBlock:
            operand_type = operand_type.get_element_type()
            
        if isinstance(operand_type, TInteger) and result_type == TFloat:
            new_op = f"%{Node.get_mlir_id()}"
            Node.context.get_block().mlir_append(indent + f"{new_op} = arith.sitofp {operand_mlir.strip(' ')} : i32 to f64\n")
            return new_op
        elif isinstance(operand_type, TLong) and result_type == TFloat:
            new_op = f"%{Node.get_mlir_id()}"
            Node.context.get_block().mlir_append(indent + f"{new_op} = arith.sitofp {operand_mlir.strip(' ')} : i64 to f64\n")
            return new_op
        elif isinstance(operand_type, TFloat) and result_type == TInteger:
            new_op = f"%{Node.get_mlir_id()}"
            Node.context.get_block().mlir_append(indent + f"{new_op} = arith.fptosi {operand_mlir.strip(' ')} : f64 to i32\n")
            return new_op
        elif isinstance(operand_type, TFloat) and result_type == TLong:
            new_op = f"%{Node.get_mlir_id()}"
            Node.context.get_block().mlir_append(indent + f"{new_op} = arith.fptosi {operand_mlir.strip(' ')} : f64 to i64\n")
            return new_op
        elif isinstance(operand_type, TInteger) and result_type == TLong:
            new_op = f"%{Node.get_mlir_id()}"
            Node.context.get_block().mlir_append(indent + f"{new_op} = arith.extsi {operand_mlir.strip(' ')} : i32 to i64\n")
            return new_op
        elif isinstance(operand_type, TLong) and result_type == TInteger:
            new_op = f"%{Node.get_mlir_id()}"
            Node.context.get_block().mlir_append(indent + f"{new_op} = arith.trunci {operand_mlir.strip(' ')} : i64 to i32\n")
            return new_op
        elif operand._index_type and result_type == TFloat:
            new_op = f"%{Node.get_mlir_id()}"
            Node.context.get_block().mlir_append(indent + f"{new_op} = arith.index_cast {operand_mlir.strip(' ')} : index to f64\n")
            return new_op
        elif operand._index_type and result_type == TInteger:
            new_op = f"%{Node.get_mlir_id()}"
            Node.context.get_block().mlir_append(indent + f"{new_op} = arith.index_cast {operand_mlir.strip(' ')} : index to i32\n")
            return new_op
        elif operand._index_type and result_type == TLong:
            new_op = f"%{Node.get_mlir_id()}"
            Node.context.get_block().mlir_append(indent + f"{new_op} = arith.index_cast {operand_mlir.strip(' ')} : index to i64\n")
            return new_op    
        return operand_mlir  # No casting needed, return original operand


    def get_mlir_op(self):
        op_type_class = TDataBlock.get_mlir_type_for_operation(self._lhs, self._rhs)
        return BinaryOperation.get_mlir_arithmetic_op(self._operation, op_type_class)    

    def print_mlir(self, indent_level = 0):
        indent = ' ' * indent_level * Node.spaces_per_tab
        self._lhs.hoist(True)
        self._rhs.hoist(True)  
        
        op_type_class = TDataBlock.get_mlir_type_for_operation(self._lhs, self._rhs)
        result_type = op_type_class.print_mlir(0)
        
        op_type_class = TDataBlock.get_mlir_type_for_operation(self._lhs, self._rhs)
        result_type = op_type_class().print_mlir(0)

        const_zero_id = None

        self._lhs.hoist(True)
        lhs = self._lhs.print_mlir(indent_level).strip(' ')
        
        if TDataBlock.needs_memref_load(self._lhs):
            const_zero_id = TDataBlock.ensure_const_zero(const_zero_id, indent_level)
            lhs = TDataBlock.generate_memref_load(self._lhs, lhs, const_zero_id, indent_level)
            
        lhs = BinaryOperation.cast_operand(lhs, self._lhs, op_type_class, indent_level)

        self._rhs.hoist(True)
        rhs = self._rhs.print_mlir(indent_level).strip(' ')
        
        if TDataBlock.needs_memref_load(self._rhs):
            const_zero_id = TDataBlock.ensure_const_zero(const_zero_id, indent_level)
            rhs = TDataBlock.generate_memref_load(self._rhs, rhs, const_zero_id, indent_level)

        rhs = BinaryOperation.cast_operand(rhs, self._rhs, op_type_class, indent_level)

        mlir_str = ''
        for mlir in Node.context.pop_block().get_mlir():
            mlir_str += indent + mlir.strip(' ')
        
        self.mlir_id(f"%{Node.get_mlir_id()}")
        
        mlir_str += indent + f"""{self.mlir_id()} = {self.get_mlir_op()} {lhs}, {rhs} : {result_type}\n"""        
        Node.context.push_block()
        Node.context.get_block().mlir_append(mlir_str)
            
        return self.mlir_id()

    def print_tree(self, indent_level=0):
        return f"""
BinaryOperation {self._operation}:
{self._lhs.print_tree(indent_level + 1)}
{self._rhs.print_tree(indent_level + 1)}"""
    

class UnaryOperation(Expression):
    def __init__(self, operation, operand):
        super().__init__()
        self._operation = operation
        self._operand = operand
    
    def __add__(self, rhs):
        return BinaryOperation("+", self, rhs)

    def __sub__(self, rhs):
        return BinaryOperation("-", self, rhs)
    
    def __mul__(self, rhs):
        return BinaryOperation("*", self, rhs)

    def __neg__(self):
        if self._operation == "-":
            return self._operand
        else:
            return UnaryOperation("-", self._operand)

    def get_type(self):
        return self._operand.get_type()

    def print_cpp(self, indent_level = 0):
        return f"({self._operation} {self._operand.print_cpp()})"
    
    def print_omp(self, indent_level = 0):
        return f"({self._operation} {self._operand.print_omp()})"
    
    def print_sycl(self, indent_level = 0):
        return f"({self._operation} {self._operand.print_sycl()})"
    
    def print_mlir(self, indent_level=0):
        indent = ' ' * indent_level * Node.spaces_per_tab
        self._operand.hoist(True)

        operand_type = self._operand.get_type()
        if isinstance(operand_type, TDataBlock):
            element_type_class = operand_type.get_element_type()
            operand_type = element_type_class() if element_type_class else operand_type

        if isinstance(operand_type, TFloat):
            if self._operation == "+":
                Node.context.get_block().mlir_append(self._operand.print_mlir(indent_level))
                return self.mlir_id()
            elif self._operation == "-":
                op_mlir = self._operand.print_mlir(indent_level).strip(' ')
                mlir_str = "".join(Node.context.get_block().pop_mlir())
                neg_one_id = f"%{Node.get_mlir_id()}"
                mlir_str += indent + f"{neg_one_id} = arith.constant -1.0 : f64\n"

                self.mlir_id(f"%{Node.get_mlir_id()}")
                mlir_str += indent + f"{self.mlir_id()} = arith.mulf {op_mlir}, {neg_one_id} : f64\n"
                Node.context.push_block()
                Node.context.get_block().mlir_append(mlir_str)
                return self.mlir_id()
        else:
            if self._operation == "+":
                Node.context.get_block().mlir_append(self._operand.print_mlir(indent_level))
                return self.mlir_id()
            elif self._operation == "-":
                op_mlir = self._operand.print_mlir(indent_level).strip(' ')
                mlir_str = "".join(Node.context.get_block().pop_mlir())
                neg_one_id = f"%{Node.get_mlir_id()}"
                mlir_str += indent + f"{neg_one_id} = arith.constant -1 : i32\n"

                self.mlir_id(f"%{Node.get_mlir_id()}")
                mlir_str += indent + f"{self.mlir_id()} = arith.muli {op_mlir}, {neg_one_id} : i32\n"
                Node.context.push_block()
                Node.context.get_block().mlir_append(mlir_str)
                return self.mlir_id()
                
    def print_tree(self, indent_level=0):
        return f"""
UnaryOperation {self._operation}:
{self._operand.print_tree(indent_level + 1)}"""

class DataBlockIndex(Expression):
    def __init__(self, dataBlock, index):
        super().__init__()
        if type(dataBlock) is Name:
            self._dataBlock = Name.variables[dataBlock.id]
        else:
            self._dataBlock = dataBlock
        
        self._index = index
        self._iteration_range = self._dataBlock._iteration_range
        self._memory_range = self._dataBlock._memory_range
    
    def index(self, index):
        return VectorIndex(self._dataBlock.index(index), self._index)

    def multidimensional_index(self, index, start_index = 0, dimensions = []):
        return VectorIndex(self._dataBlock.multidimensional_index(index, start_index, dimensions), self._index)

    def get_type(self):
        return TDataBlock(self._iteration_range, String("double"))

    def print_cpp(self, indent_level = 0):
        pass

    def print_omp(self, indent_level = 0):
        pass

    def print_sycl(self, indent_level = 0):
        pass

    def print_mlir(self, indent_level = 0):
        pass

    def print_tree(self, indent_level=0):
        pass


class DataBlockComparison(Expression):
    def __init__(self, operation, lhs, rhs):
        if type(lhs) is Name and type(lhs.get_type()) is TDataBlock:
            self._lhs = Name.variables[lhs.id]
        else:
            self._lhs = lhs
        
        if type(rhs) is Name and type(rhs.get_type()) is TDataBlock:
            self._rhs = Name.variables[rhs.id]
        else:
            self._rhs = rhs
        
        self._operation = operation
        self.requires_memory_allocation = True
        self._offset = None

        if type(self._lhs.get_type()) is TDataBlock and type(self._rhs.get_type()) is TDataBlock:
            self._iteration_range = copy.deepcopy(self._lhs._iteration_range)
            self._memory_range = copy.deepcopy(self._lhs._memory_range)
            if type(self._rhs._iteration_range[0][1]) is not BinaryOperation and  type(self._iteration_range[0][1]) is not BinaryOperation and self._rhs._iteration_range[0][1] > self._iteration_range[0][1]:
                self._iteration_range[0][1] = self._rhs._iteration_range[0][1]
            elif type(self._iteration_range[0][1]) is not BinaryOperation and self._iteration_range[0][1] == Integer(1):
                self._iteration_range[0][1] = self._rhs._iteration_range[0][1]
            if type(self._rhs._memory_range[0]) is not BinaryOperation and  type(self._memory_range[0]) is not BinaryOperation and self._rhs._memory_range[0][1] > self._memory_range[0][1]:
                self._memory_range[0] = self._rhs._memory_range[0]
            elif type(self._memory_range[0]) is not BinaryOperation and self._memory_range[0] == Integer(1):
                self._memory_range[0] = self._rhs._memory_range[0]

        elif type(self._lhs.get_type()) is TDataBlock:
            self._iteration_range = copy.deepcopy(self._lhs._iteration_range)
            self._memory_range = copy.deepcopy(self._lhs._memory_range) 
        else:
            self._iteration_range = copy.deepcopy(self._rhs._iteration_range)
            self._memory_range = copy.deepcopy(self._rhs._memory_range)
    
    def index(self, index):
        return Comparison(self._operation, self._lhs.index(index), self._rhs.index(index))

    def multidimensional_index(self, index, start_index = 0, dimensions = []):
        return Comparison(self._operation, self._lhs.multidimensional_index(index, start_index, dimensions), self._rhs.multidimensional_index(index, start_index, dimensions))

    def get_type(self):
        return TDataBlock(self._iteration_range, String("int"))

    def print_cpp(self, indent_level = 0):
        pass

    def print_omp(self, indent_level = 0):
        pass

    def print_sycl(self, indent_level = 0):
        pass

    def print_mlir(self, indent_level = 0):
        pass

    def print_tree(self, indent_level=0):
        pass

class DataBlockBinaryOperation(Expression):
    def __init__(self, operation, lhs, rhs, useFunctionSyntax=False):
        super().__init__()
        if type(lhs) is Name and type(lhs.get_type()) is TDataBlock:
            self._lhs = Name.variables[lhs.id]
        else:
            self._lhs = lhs
        
        if type(rhs) is Name and type(rhs.get_type()) is TDataBlock:
            self._rhs = Name.variables[rhs.id]
        else:
            self._rhs = rhs
        
        self._operation = operation
        self.requires_memory_allocation = True
        self._useFunctionSyntax = useFunctionSyntax

        if type(self._lhs.get_type()) is TDataBlock and type(self._rhs.get_type()) is TDataBlock:
            self._iteration_range = copy.deepcopy(self._lhs._iteration_range)
            self._memory_range = copy.deepcopy(self._lhs._memory_range)
            if len(self._iteration_range) == 1:
                self._iteration_range = copy.deepcopy(self._rhs._iteration_range)
            elif type(self._rhs._iteration_range[0][1]) is not BinaryOperation and  type(self._iteration_range[0][1]) is not BinaryOperation and self._rhs._iteration_range[0][1] > self._iteration_range[0][1]:
                self._iteration_range[0][1] = self._rhs._iteration_range[0][1]
            elif type(self._iteration_range[0][1]) is not BinaryOperation and self._iteration_range[0][1] == Integer(1):
                self._iteration_range[0][1] = self._rhs._iteration_range[0][1]

            self._memory_range[0] = [Integer(0), self._iteration_range[0][1] - self._iteration_range[0][0]]

        elif type(self._lhs.get_type()) is TDataBlock:
            self._iteration_range = copy.deepcopy(self._lhs._iteration_range)
            self._memory_range = copy.deepcopy(self._lhs._memory_range) 
        else:
            self._iteration_range = copy.deepcopy(self._rhs._iteration_range)
            self._memory_range = copy.deepcopy(self._rhs._memory_range)

        if (type(self._iteration_range[0][0]) is Integer and type(self._iteration_range[0][1]) is Integer and (self._iteration_range[0][1] - self._iteration_range[0][0])._value == 1):
            self._iteration_range[0][0] = Integer(0)
            self._iteration_range[0][1] = Integer(1)

        self._offset = []
        for [x, y] in self._iteration_range:
            self._offset.append(x)

    
    def index(self, index):
        if self._useFunctionSyntax:
            return FunctionCall(self._operation, [self._lhs.index(index), self._rhs.index(index)])
        else:
            return BinaryOperation(self._operation, self._lhs.index(index), self._rhs.index(index))

    def multidimensional_index(self, index, start_index = 0, dimensions = []):
        if self._useFunctionSyntax:
            return FunctionCall(self._operation, [self._lhs.multidimensional_index(index, start_index, dimensions), self._rhs.multidimensional_index(index, start_index, dimensions)])
        else:
            return BinaryOperation(self._operation, self._lhs.multidimensional_index(index, start_index, dimensions), self._rhs.multidimensional_index(index, start_index, dimensions))

    def get_type(self):
        lhs_is_double = False
        rhs_is_double = False
        if type(self._lhs.get_type()) is TDataBlock:
            lhs_is_double = (str(self._lhs.get_type().get_single()) == "double")
        else:
            lhs_is_double = (str(self._lhs.get_type()) == "double")

        if type(self._rhs.get_type()) is TDataBlock:
            rhs_is_double = (str(self._rhs.get_type().get_single()) == "double")
        else:
            rhs_is_double = (str(self._rhs.get_type()) == "double")

        if (lhs_is_double or rhs_is_double):
            return TDataBlock(self._iteration_range, String("double"))
        else:
            return TDataBlock(self._iteration_range, String("int"))

    def __sub__(self, rhs):
        return DataBlockBinaryOperation("-", self, rhs)
    
    def __add__(self, rhs):
        return DataBlockBinaryOperation("+", self, rhs)

    def print_cpp(self, indent_level = 0):
        pass

    def print_omp(self, indent_level = 0):
        pass

    def print_sycl(self, indent_level = 0):
        pass

    def print_mlir(self, indent_level = 0):
        pass

    def print_tree(self, indent_level=0):
        pass


class DataBlockUnaryOperation(Expression):
    def __init__(self, operation, dataBlock):
        super().__init__()
        if type(dataBlock) is Name and type(dataBlock.get_type()) is TDataBlock:
            self._dataBlock = Name.variables[dataBlock.id]
        else:
            self._dataBlock = dataBlock
        
        self._operation = operation
        self.requires_memory_allocation = True
        self._offset = None

        self._iteration_range = copy.deepcopy(self._dataBlock._iteration_range)
        self._memory_range = copy.deepcopy(self._dataBlock._memory_range)
    
    def index(self, index):
        return UnaryOperation(self._operation, self._dataBlock.index(index))

    def multidimensional_index(self, index, start_index = 0, dimensions = []):
        return UnaryOperation(self._operation, self._dataBlock.multidimensional_index(index, start_index, dimensions))

    def get_type(self):
        return TDataBlock(self._iteration_range, String("double"))

    def print_cpp(self, indent_level = 0):
        pass

    def print_omp(self, indent_level = 0):
        pass

    def print_sycl(self, indent_level = 0):
        pass

    def print_mlir(self, indent_level = 0):
        pass

    def print_tree(self, indent_level=0):
        pass


class DataBlockUnaryMax:
    def __init__(self, dataBlock):
        if type(dataBlock) is Name:
            self._dataBlock = Name.variables[dataBlock.id]
        else:
            self._dataBlock = dataBlock

    def set_output_variable(self, outputVariable):
        self._outputVariable = Name.variables[outputVariable.id]
        self._loop = For(self._dataBlock._iteration_range[-1])
        self._loops = [self._loop]
        self._loop.add_statement(Assignment(self._outputVariable.multidimensional_index([self._loop.get_iteration_variable()]), Float(0.0)))
        for i in range(len(self._dataBlock._iteration_range) - 2, -1, -1):
            self._loops.append(For(self._dataBlock._iteration_range[i]))
            self._loops[-2].add_statement(self._loops[-1])
        index = []
        for loop in self._loops[::-1]:
            index.append(loop.get_iteration_variable())
        self._loops[-1].add_statement(Assignment(self._outputVariable.multidimensional_index(index), Max(self._outputVariable.multidimensional_index(index), self._dataBlock.multidimensional_index(index))))
        for loop in self._loops:
            loop.close_scope()
    
    def get_type(self):
        return TDataBlock(self._dataBlock._iteration_range, TFloat())

    def print_cpp(self, indent_level = 0):
        return self._loop.print_cpp(indent_level)
    
    def print_omp(self, indent_level = 0):
        if Node.use_accelerator:
            return f"""{' ' * indent_level * Node.spaces_per_tab}omp_target_memset(maxEigenvalues, 0, N * sizeof(double), targetDevice);
{' ' * indent_level * Node.spaces_per_tab}for (int {self._loop._iteration_variable.print_cpp()} = {self._loop._iteration_range[0].print_cpp()}; {self._loop._iteration_variable.print_cpp()} < {self._loop._iteration_range[1].print_cpp()}; {self._loop._iteration_variable.print_cpp()}++) {{
{' ' * (indent_level + 1) * Node.spaces_per_tab}#pragma omp target teams loop collapse(Dimensions) reduction(max:{self._outputVariable.multidimensional_index([self._loop.get_iteration_variable()]).print_omp()})
{self._loop._statements[1].print_omp(indent_level + 1)}
{' ' * indent_level * Node.spaces_per_tab}}}
"""
        return self._loop.print_omp(indent_level)

    
    def print_sycl(self, indent_level = 0):
        ranges = [f"range{i} = {loop.get_interval_size().print_sycl()};" for i, loop in enumerate(self._loops[0:3])]
        indexing = [f"int {loop.get_iteration_variable().print_sycl()} = index[{i}];" for i, loop in enumerate(self._loops[0:3])]
        statement_prints = [statement.print_sycl(indent_level + 1) for statement in self._loops[2]._statements]
        return os.linesep.join(ranges) + "\n" + ' ' * indent_level * Node.spaces_per_tab + f"""queue.submit([&](::sycl::handler& handler) {{handler.memset({self._outputVariable.print_sycl()}, 0, sizeof(double) * {self._loop.get_interval_size().print_sycl()}); }}).wait();
{' ' * indent_level * Node.spaces_per_tab}queue.submit([&](::sycl::handler& handler) {{
{' ' * indent_level * Node.spaces_per_tab}handler.parallel_for(::sycl::range<3>{{range0, range1, range2}}, [=](::sycl::item<3> index) {{
{os.linesep.join(indexing)}
{self._loops[3].print_sycl(indent_level + 1)}
{' ' * (indent_level + 1) * Node.spaces_per_tab}}});
{' ' * indent_level * Node.spaces_per_tab}}}).wait();"""
    
    def print_mlir(self, indent_level = 0):
        return self._loop.print_mlir(indent_level)
    
    def print_tree(self, indent_level=0):
        return f"""DataBlockUnaryMax:
{self._dataBlock.print_tree(indent_level + 1)}"""
        

class DataBlockMax(DataBlockBinaryOperation):
    def __init__(self, lhs, rhs):
        super().__init__("max", lhs, rhs)

    def index(self, index):
        return Max(self._lhs.index(index), self._rhs.index(index))
        
    def multidimensional_index(self, index, start_index = 0, dimensions = []):
        return Max(self._lhs.multidimensional_index(index, start_index, dimensions), self._rhs.multidimensional_index(index, start_index, dimensions))
    
    def print_cpp(self, indent_level = 0):
        pass

    def print_omp(self, indent_level = 0):
        pass

    def print_sycl(self, indent_level = 0):
        pass

    def print_mlir(self, indent_level = 0):
        pass

    def print_tree(self, indent_level=0):
        pass

# Types
class TCustom(Type):
    def __init__(self, type_name):
        self._type = type_name
    
    def print_cpp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + self._type
    
    def print_omp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + self._type

    def print_sycl(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + self._type

    def print_mlir(self, indent_level=0):
        mlir_type = Node.get_mlir_type(self._type)
        if mlir_type is not None:
            return ' ' * indent_level * Node.spaces_per_tab + mlir_type
        else:
            # Fallback to type_map for complex types
            type_entry = Node.type_map[f"{self._type}"]
            if isinstance(type_entry, str):
                return ' ' * indent_level * Node.spaces_per_tab + type_entry
            else:
                # Type object (e.g., TVoid, TFloat, etc.)
                return ' ' * indent_level * Node.spaces_per_tab + type_entry.print_mlir(0)
    
    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "TCustom: " + self._type
    
    def is_nested_memref(self):
        try:
            return isinstance(self._type, str) and ('xmemref<' in self._type or self._type.count('memref<') > 1)
        except Exception:
            return False


class TInteger(Type):
    def print_cpp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "int"
    
    def print_omp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "int"
    
    def print_sycl(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "int"

    def print_mlir(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "i32"
    
    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "TInteger"


class TLong(Type):
    """Type for 64-bit integers (long) used in MLIRCellData struct fields"""
    def print_cpp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "long"
    
    def print_omp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "long"
    
    def print_sycl(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "long"

    def print_mlir(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "i64"
    
    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "TLong"


class TBoolean(Type):
    def print_cpp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "bool"
    
    def print_omp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "bool"
    
    def print_sycl(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "bool"

    def print_mlir(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "i1"
    
    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "TBoolean"


class TFloat(Type):
    def __init__(self, reference = False):
        self._reference = reference
    def print_cpp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "double" + ("&" if self._reference else "")
    
    def print_omp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "double" + ("&" if self._reference else "")
    
    def print_sycl(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "double" + ("&" if self._reference else "")

    def print_mlir(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "f64"
    
    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "TFloat"


class TString(Type):
    def print_cpp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "const char*"
    
    def print_omp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "const char*"
    
    def print_sycl(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "const char*"
    
    def print_mlir(self, indent_level=0):
        pass

    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "TString"

   
class TNestedMemref(Type):
    def __init__(self, element_type=TFloat()):
        super().__init__()
        self._element_type = element_type
    
    def print_cpp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + f'double**'  # C++ representation
    
    def print_mlir(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + f'memref<?xmemref<?x{self._element_type.print_mlir(0).strip()}>>'
    
    def print_omp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + f'double**'
    
    def print_sycl(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + f'double**'
    
    def is_nested_memref(self):
        return True
    
    def get_element_type(self):
        return self._element_type


class TFunctionType(Type):
    def __init__(self, signature_string):
        super().__init__()
        self._signature = signature_string
    
    def print_cpp(self, indent_level=0):
        # C++ function pointers don't have direct equivalent
        return ' ' * indent_level * Node.spaces_per_tab + 'void*'
    
    def print_mlir(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + self._signature
    
    def print_omp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + 'void*'
    
    def print_sycl(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + 'void*'


class TSimpleMemref(Type):
    def __init__(self, element_type=TFloat(), dimensions=1):
        super().__init__()
        self._element_type = element_type
        self._dimensions = dimensions
    
    def print_cpp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + 'void*'
    
    def print_mlir(self, indent_level=0):
        # Build memref<?x...xf64> with appropriate number of dimensions
        dims = 'x'.join(['?'] * self._dimensions)
        return ' ' * indent_level * Node.spaces_per_tab + f'memref<{dims}x{self._element_type.print_mlir(0).strip()}>'  # Use print_mlir for element type
    
    def print_omp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + 'void*'
    
    def print_sycl(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + 'void*'
    
    def get_element_type(self):
        return self._element_type


class TDataBlock(Type):
    def __init__(self, dimensions, underlying_type):
        self._dimensions = dimensions
        self._underlying_type = underlying_type
    
    @staticmethod
    def needs_memref_load(operand):
        if isinstance(operand, VectorIndex):
            return False
        
        operand_type = operand.get_type()
        if isinstance(operand_type, TDataBlock):
            if (isinstance(operand_type._dimensions, int) and operand_type._dimensions == 1) or \
               (isinstance(operand_type._dimensions, list) and len(operand_type._dimensions) == 1):
                return True
            
            if isinstance(operand, Subscript) and operand.is_nested_subscript():
                return True
            
        return False
    
    @staticmethod
    def generate_memref_load(operand, operand_mlir, const_zero_id, indent_level):
        indent = ' ' * indent_level * Node.spaces_per_tab
        load_id = f"%{Node.get_mlir_id()}"
        operand_type = operand.get_type()
        
        if isinstance(operand_type, TDataBlock):
            # Handle nested array access (double** style arrays)
            if isinstance(operand, Subscript) and operand.is_nested_subscript():
                return operand_mlir
            
            element_type_class = operand_type.get_element_type()
            if element_type_class is not None:
                element_type = element_type_class().print_mlir(0)
            else:
                element_type = "f64"  # Default fallback
            
            load_str = indent + f"{load_id} = memref.load {operand_mlir}[{const_zero_id}] {{name = \"{operand_mlir.strip('%')}\"}} : memref<?x{element_type}>\n"
            Node.context.get_block().mlir_append(load_str)
            return load_id
            
        return operand_mlir
    
    @staticmethod
    def ensure_const_zero(const_zero_id, indent_level):
        if const_zero_id is None:
            const_zero_id = f"%{Node.get_mlir_id()}"
            indent = ' ' * indent_level * Node.spaces_per_tab
            Node.context.get_block().mlir_append(indent + f"{const_zero_id} = arith.constant 0 : index\n")
        return const_zero_id

    def get_single(self):
        return self._underlying_type
    
    def get_element_type(self):
        if isinstance(self._underlying_type, String):
            if self._underlying_type._value == "double":
                return TFloat
            elif self._underlying_type._value == "int":
                return TInteger
            else:
                return None
        else:
            return type(self._underlying_type)
    
    @staticmethod
    def get_mlir_type_for_operation(lhs, rhs):
        lhs_operand_type = lhs.get_type()
        if type(lhs_operand_type) is TDataBlock:
            lhs_type = lhs_operand_type.get_element_type()
        else:
            lhs_type = type(lhs_operand_type)
            
        rhs_operand_type = rhs.get_type()
        if type(rhs_operand_type) is TDataBlock:
            rhs_type = rhs_operand_type.get_element_type()
        else:
            rhs_type = type(rhs_operand_type)
        
        # Use float if either operand is float
        if lhs_type is TFloat or rhs_type is TFloat:
            return TFloat
        elif lhs_type is TLong or rhs_type is TLong:
            return TLong
        elif lhs_type is TInteger and rhs_type is TInteger:
            return TInteger
        elif lhs_type is TBoolean and rhs_type is TBoolean:
            return TBoolean

        return None
    
    def get_array_element_info(self):
        element_type_class = self.get_element_type()
        
        # Check for vector types by examining the underlying type
        if isinstance(self._underlying_type, String):
            is_vector = "tarch::la::Vector" in self._underlying_type._value
        elif hasattr(self._underlying_type, '_type'):
            is_vector = "tarch::la::Vector" in self._underlying_type._type
        else:
            is_vector = "tarch::la::Vector" in self._underlying_type
        
        # Determine MLIR element type based on vector vs scalar
        if is_vector:
            mlir_element_type = "i64"  # Vectors stored as pointers (i64)
        else:
            mlir_element_type = element_type_class().print_mlir(0)
            
        return element_type_class or TFloat, is_vector, mlir_element_type

    
    def print_cpp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + self._underlying_type.print_cpp() + "*"
    
    def print_omp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + self._underlying_type.print_omp() + "*"

    def print_sycl(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + self._underlying_type.print_sycl() + "*"

    def print_mlir(self, indent_level=0):
            dims = []
            
            if isinstance(self._dimensions, int):
                dims.append(str(self._dimensions))
            else:
                for dim in self._dimensions:
                    match dim:
                        case int():
                            dims.append(str(dim))                    
                        case Expression():
                            dims.append(dim.print_mlir(indent_level))
                        case list():
                            Node.context.push_block()
                            for op in dim:
                                op.print_mlir(indent_level)
                            dims.append(''.join(Node.context.pop_block().get_mlir()))
                                        
            # NOTE: The underlying type isn't an MLIR type here but a string representing the C++ type. So...
            # 	We look up the C++ type string in a types map in Node and get the MLIR type
            if isinstance(self._underlying_type, Type):
                type_name = self._underlying_type.print_mlir(indent_level)
            else:
                # We have a String type here
                type_name = Node.get_mlir_type(self._underlying_type._value)
                if type_name is None:
                    # Fallback to type_map for complex types
                    type_name = Node.type_map[f"{self._underlying_type}"]
            
            # If type_name is already a memref type (e.g., from Vector types that map to 'memref<?xi64>'),
            # return it directly instead of wrapping it again. This prevents creating invalid nested memrefs like 'memref<?xmemref<?xi64>>'.
            if isinstance(type_name, str) and type_name.startswith("memref<"):
                return ' ' * indent_level * Node.spaces_per_tab + type_name
            
            # Build the memref type with the correct number of dimensions
            # NOTE: self._dimensions may include halo dimensions, so we need to infer actual data dimensions
            # For now, we use a heuristic: if 1D field, return 1D; if multi-D, assume 2D (common case)
            # The type_map provides the correct types for named fields, so this should rarely be used
            
            if isinstance(self._dimensions, int):
                if self._dimensions == 1:
                    actual_dims = 1
                else:
                    # Multi-dimensional iteration ranges with halo are typically 2D or 3D data
                    # For now, assume 2D as fallback
                    actual_dims = 2
            elif isinstance(self._dimensions, list):
                if len(self._dimensions) <= 1:
                    actual_dims = 1
                else:
                    actual_dims = 2  # Fallback for multi-dim
            else:
                actual_dims = 1  # Fallback
            
            # Build memref with ? for each actual dimension
            dimension_list = ["?"] * actual_dims
            memref_dims = "x".join(dimension_list)
            
            return ' ' * indent_level * Node.spaces_per_tab + f"""memref<{memref_dims}x{type_name}>"""
    
    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "TDataBlock"


def _is_nested_memref(type_map_entry):
    if isinstance(type_map_entry, str):
        # String: check for 'xmemref<' pattern
        return 'xmemref<' in type_map_entry
    elif hasattr(type_map_entry, 'is_nested_memref'):
        # Type object: call the method
        return type_map_entry.is_nested_memref()
    else:
        return False


class TMatrix(Type):
    def print_cpp(self, indent_level = 0):
        pass
    
    def print_omp(self, indent_level = 0):
        pass
    
    def print_sycl(self, indent_level = 0):
        pass
    
    def print_mlir(self, indent_level=0):
        pass

    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "TMatrix"


class TDiagonalMatrix(Type):
    def print_cpp(self, indent_level = 0):
        pass
    
    def print_omp(self, indent_level = 0):
        pass
    
    def print_sycl(self, indent_level = 0):
        pass
    
    def print_mlir(self, indent_level=0):
        pass

    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "TDiagonalMatrix"


class TVector(Type):
    def print_cpp(self, indent_level = 0):
        pass
    
    def print_omp(self, indent_level = 0):
        pass
    
    def print_sycl(self, indent_level = 0):
        pass
    
    def print_mlir(self, indent_level=0):
        pass

    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "TVector"


# Statements
class If(Statement):
    def __init__(self, boolean):
        super().__init__()
        self._boolean = boolean
        self._statements = []
    
    def add_statement(self, statement):
        if statement is not None:
            self._statements.append(statement)

    def print_cpp(self, indent_level=0):
        statement_prints = [statement.print_cpp(indent_level + 1) for statement in self._statements]
        return ' ' * indent_level * Node.spaces_per_tab + f"""if constexpr ({self._boolean.print_cpp()}) {{
{os.linesep.join(statement_prints)}
""" + ' ' * indent_level * Node.spaces_per_tab + "}"
    
    def print_omp(self, indent_level=0):
        statement_prints = [statement.print_omp(indent_level + 1) for statement in self._statements]
        return ' ' * indent_level * Node.spaces_per_tab + f"""if constexpr ({self._boolean.print_omp()}) {{
{os.linesep.join(statement_prints)}
""" + ' ' * indent_level * Node.spaces_per_tab + "}"
    
    def print_sycl(self, indent_level=0):
        statement_prints = [statement.print_sycl(indent_level + 1) for statement in self._statements]
        return ' ' * indent_level * Node.spaces_per_tab + f"""if constexpr ({self._boolean.print_sycl()}) {{
{os.linesep.join(statement_prints)}
""" + ' ' * indent_level * Node.spaces_per_tab + "}"
    
    def print_mlir(self, indent_level=0):
        indent = ' ' * indent_level * Node.spaces_per_tab
        Node.context.push_block()
        statement_prints = [statement.print_mlir(indent_level + 1) for statement in self._statements]
        body_statements = ''.join(Node.context.pop_block().get_mlir())
        body_statements += ''.join(statement_prints)
        return indent + f"""scf.if {self._boolean.print_mlir(indent_level)} {{
{body_statements}""" + indent + "}"
        

    def print_tree(self, indent_level=0):
        statement_prints = [statement.print_tree(indent_level + 1) for statement in self._statements]
        return ' ' * indent_level * Node.spaces_per_tab + f"""If: 
{os.linesep.join(statement_prints)}"""

class For(Statement):
    _inuse_iteration_variables = set()
    _iteration_variable_names = ['i', 'j', 'k', 'l', 'n', 'm', 'a', 'b', 'c', 'd']

    def __init__(self, iteration_range, iteration_variable_name = None, use_scheduler = False):
        super().__init__()
        self._iteration_range = iteration_range
        self._statements = []
        self._use_scheduler = False# use_scheduler
        
        if iteration_variable_name == None:
            for variable_name in For._iteration_variable_names:
                if variable_name not in For._inuse_iteration_variables:
                    self._iteration_variable = Name(variable_name, TInteger())
                    self._iteration_variable._index_type = True
                    For._inuse_iteration_variables.add(variable_name)
                    break
        else:
            self._iteration_variable = Name(iteration_variable_name, TInteger())
            For._inuse_iteration_variables.update({iteration_variable_name : self._iteration_variable})
        
    def get_interval_size(self):
        return self._iteration_range[1] - self._iteration_range[0]

    def get_iteration_variable(self):
        return self._iteration_variable

    def add_statement(self, statement):
        self._statements.append(statement)
    
    def close_scope(self):
        For._inuse_iteration_variables.remove(self._iteration_variable.id)
    
    def print_cpp(self, indent_level = 0):
        statement_prints = [statement.print_cpp(indent_level + 1) for statement in self._statements]
        if self._use_scheduler:
            return ' ' * indent_level * Node.spaces_per_tab + f"""parallelForWithSchedulerInstructions({self._iteration_variable.print_cpp()}, {self._iteration_range[1].print_cpp()}, loopParallelism) {{
{os.linesep.join(statement_prints)}
{' ' * indent_level * Node.spaces_per_tab}}}
endParallelFor"""
        else:
            return ' ' * indent_level * Node.spaces_per_tab + f"""for (int {self._iteration_variable.print_cpp()} = {self._iteration_range[0].print_cpp()}; {self._iteration_variable.print_cpp()} < {self._iteration_range[1].print_cpp()}; {self._iteration_variable.print_cpp()}++) {{
{os.linesep.join(statement_prints)}
""" + ' ' * indent_level * Node.spaces_per_tab + "}"
        
    
    def print_omp(self, indent_level = 0):
        statement_prints = [statement.print_omp(indent_level + 1) for statement in self._statements]
        return ' ' * indent_level * Node.spaces_per_tab + f"""for (int {self._iteration_variable.print_omp()} = {self._iteration_range[0].print_omp()}; {self._iteration_variable.print_omp()} < {self._iteration_range[1].print_omp()}; {self._iteration_variable.print_omp()}++) {{
{os.linesep.join(statement_prints)}
""" + ' ' * indent_level * Node.spaces_per_tab + "}"

    def print_sycl(self, indent_level = 0):
        statement_prints = [statement.print_sycl(indent_level + 1) for statement in self._statements]
        return ' ' * indent_level * Node.spaces_per_tab + f"""for (int {self._iteration_variable.print_sycl()} = {self._iteration_range[0].print_sycl()}; {self._iteration_variable.print_sycl()} < {self._iteration_range[1].print_sycl()}; {self._iteration_variable.print_sycl()}++) {{
{os.linesep.join(statement_prints)}
""" + ' ' * indent_level * Node.spaces_per_tab + "}"

    def print_mlir(self, indent_level=0):
        indent = ' ' * indent_level * Node.spaces_per_tab
        bounds_casting = ''
            
        if isinstance(self._iteration_range[0], list) or isinstance(self._iteration_range[0], tuple):
            dims = len(self._iteration_range)
            iter_vars = []
            lwbs = []
            upbs = []
            steps = []
            for d in range(dims):
                var = For._iteration_variable_names[d]
                iter_vars.append(f"%{var}")
                self._iteration_range[d][0].hoist(True)
                self._iteration_range[d][1].hoist(True)
                if isinstance(self._iteration_range[d][0], Integer):
                    mlir_id = f"%{Node.get_mlir_id()}"
                    bounds_casting += indent + f"{mlir_id} = arith.constant {self._iteration_range[d][0].print_mlir(indent_level)} : index\n"
                    lwb = mlir_id
                else:
                    lwb = self._iteration_range[d][0].print_mlir(indent_level).strip()
                if isinstance(self._iteration_range[d][1], Integer):
                    mlir_id = f"%{Node.get_mlir_id()}"
                    bounds_casting += indent + f"{mlir_id} = arith.constant {self._iteration_range[d][1].print_mlir(indent_level)} : index\n"
                    upb = mlir_id
                else:
                    upb = self._iteration_range[d][1].print_mlir(indent_level).strip()
                lwbs.append(lwb)
                upbs.append(upb)
                mlir_id = f"%{Node.get_mlir_id()}"
                bounds_casting += indent + f"{mlir_id} = arith.constant 1 : index\n"
                steps.append(mlir_id)
        else:
            self._iteration_range[0].hoist(True)
            self._iteration_range[1].hoist(True)
            if not self._iteration_variable._index_type:
                self._iteration_variable.hoist(True)
                mlir_str = self._iteration_variable.print_mlir(indent_level).strip()
                mlir_id = f"%{Node.get_mlir_id()}"
                bounds_casting += ''.join(Node.context.pop_block().get_mlir())
                Node.context.push_block()
                bounds_casting += indent + f"{mlir_id} = arith.index_cast {mlir_str} : {self._iteration_variable.get_type().print_mlir(0)} to index\n"
                iter_vars = [mlir_id]
            else:
                iter_vars = [self._iteration_variable.print_mlir(indent_level).strip()]
                
            if not self._iteration_range[0]._index_type:
                self._iteration_range[0].hoist(True)
                mlir_str = self._iteration_range[0].print_mlir(indent_level).strip()
                mlir_id = f"%{Node.get_mlir_id()}"
                bounds_casting += ''.join(Node.context.pop_block().get_mlir())
                Node.context.push_block()
                bounds_casting += indent + f"{mlir_id} = arith.index_cast {mlir_str} : {self._iteration_range[0].get_type().print_mlir(indent_level).strip()} to index\n"
                lwbs = [mlir_id]
            else:
                lwbs += [self._iteration_range[0].print_mlir(indent_level).strip()]
                
            if not self._iteration_range[1]._index_type:
                self._iteration_range[1].hoist(True)
                mlir_str = self._iteration_range[1].print_mlir(indent_level).strip()
                mlir_id = f"%{Node.get_mlir_id()}"
                bounds_casting += ''.join(Node.context.pop_block().get_mlir())
                Node.context.push_block()
                bounds_casting += indent + f"{mlir_id} = arith.index_cast {mlir_str} : {self._iteration_range[1].get_type().print_mlir(indent_level).strip()} to index\n"
                upbs = [mlir_id]
            else:
                upbs += [self._iteration_range[1].print_mlir(indent_level).strip()]
                
            bounds_casting += ''.join(Node.context.pop_block().get_mlir())
            mlir_id = f"%{Node.get_mlir_id()}"
            bounds_casting += indent + f"{mlir_id} = arith.constant 1 : index\n"
            steps = [mlir_id]

        # Collect body
        Node.context.push_block()
        statement_prints = [statement.print_mlir(indent_level + 1) for statement in self._statements]
        statements = ''.join(Node.context.pop_block().get_mlir())
        statements += ''.join(statement_prints)

        # Check if loop contains memory allocation (nested memref allocation must be sequential)
        # Use scf.for for allocation loops, scf.parallel for computation loops
        has_allocation = any(isinstance(stmt, MemoryAllocation) for stmt in self._statements)
        
        if has_allocation:
            # Use sequential scf.for for memory allocation (cannot parallelize)
            loop_header = (
                f"{indent}scf.for {', '.join(iter_vars)} = "
                f"{', '.join(lwbs)} to {', '.join(upbs)} step {', '.join(steps)} {{\n"
            )
        else:
            # Use scf.parallel for computation loops (safe to parallelize)
            loop_header = (
                f"{indent}scf.parallel ({', '.join(iter_vars)}) = "
                f"({', '.join(lwbs)}) to ({', '.join(upbs)}) step ({', '.join(steps)}) {{\n"
            )
        loop_footer = "\n" + indent + "}\n"

        return bounds_casting + loop_header + statements + loop_footer

    def print_tree(self, indent_level = 0):
        statement_prints = [statement.print_tree(indent_level + 1) for statement in self._statements]
        return ' ' * indent_level * Node.spaces_per_tab + f"""For: 
{os.linesep.join(statement_prints)}"""
    

class Comment(Statement):
    def __init__(self, comment):
        super().__init__()
        self._comment = comment

    def get_type(self):
        return None

    def print_cpp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "//" + self._comment[1:]
    
    def print_omp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "//" + self._comment[1:]
    
    def print_sycl(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "//" + self._comment[1:]

    def print_mlir(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + "//" + self._comment[1:]
    
    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + "Comment"


class FunctionCall(Statement):
    def __init__(self, id, arguments, is_offloadable = False):
        self.id = id
        self._arguments = arguments
        self._is_offloadable = is_offloadable
    
    def add_argument(self, argument: Expression):
        self._arguments.append(argument)

    def print_cpp(self, indent_level=0):
        argument_prints = [argument.print_cpp() for argument in self._arguments]
        return ' ' * indent_level * Node.spaces_per_tab + f"""{self.id}({", ".join(argument_prints)});"""
    
    def print_omp(self, indent_level=0):
        argument_prints = [argument.print_omp() for argument in self._arguments]
        return ' ' * indent_level * Node.spaces_per_tab + f"""{self.id}({", ".join(argument_prints)});"""

    def print_sycl(self, indent_level=0):
        argument_prints = [argument.print_sycl() for argument in self._arguments]
        return ' ' * indent_level * Node.spaces_per_tab + f"""{self.id}({", ".join(argument_prints)});"""

    def print_mlir(self, indent_level=0):
        indent = ' ' * indent_level * Node.spaces_per_tab
        Node.context.push_block()
        
        # Detect if this is a bridge function call
        is_bridge_call = self.id in ['sourceTerm', 'flux', 'nonconservativeProduct']
        
        # Process arguments and handle array subscripts
        argument_prints = []
        for i, argument in enumerate(self._arguments):
            argument.hoist(True)
            
            # Check if this is a Reference to a 2D subscript for a bridge call
            is_reference_to_2d_subscript = (
                is_bridge_call and
                isinstance(argument, Reference) and
                isinstance(argument._expression, Subscript) and
                isinstance(argument._expression._value, Subscript)
            )
            
            if is_reference_to_2d_subscript:
                # This is Reference(Subscript[Subscript[2D_memref]])
                # Extract the nested structure
                outer_subscript = argument._expression  # Subscript[Subscript[...], col_index]
                inner_subscript = outer_subscript._value  # Subscript[2D_memref, row_index]
                memref_2d = inner_subscript._value  

                outer_subscript.hoist(True)
                outer_mlir = outer_subscript.print_mlir(indent_level)
                mlir_str_from_outer = "".join(Node.context.get_block().pop_mlir())
                Node.context.get_block().mlir_append(mlir_str_from_outer)
                
                # Get MLIR representations
                memref_2d.hoist(True)
                memref_2d_mlir = memref_2d.print_mlir(indent_level).strip(' ')

                # Get the base memref variable name (strip % prefix)
                base_mlir_name = memref_2d_mlir.lstrip('%')
                is_nested_base = False
                if base_mlir_name in Node.type_map and _is_nested_memref(Node.type_map[base_mlir_name]):
                    is_nested_base = True
                
                # Use unified method from Node to generate pointer arithmetic
                ptr_code, final_ptr = Node.generate_pointer_to_2d_element(
                    memref_2d_mlir, inner_subscript._new_index, outer_subscript._new_index, is_nested_base, indent_level
                )
                Node.context.get_block().mlir_append(ptr_code)
                arg_mlir = final_ptr
            else:
                # Not a 2D subscript for bridge call - use standard approach
                arg_mlir = argument.print_mlir(indent_level).strip(' ')
            
            # Regular subscript (array indexing) that needs loading
            if not is_reference_to_2d_subscript and isinstance(argument, Subscript):
                # Get the base array and index
                base_value = arg_mlir  # This is the array reference
                index = argument._new_index  # This is the converted index
                base_type = argument._value.get_type()
                
                if isinstance(base_type, TDataBlock):
                    # Check if this is a multi-dimensional (2D+) DataBlock
                    if isinstance(base_type._dimensions, int) and base_type._dimensions >= 2:
                        # This is a flat multi-dimensional memref (like memref<?x?xf64>)
                        # When passed to a function expecting memref<?xf64>, just pass it as-is
                        arg_mlir = f"{base_value}[{index}]"
                    else:
                        # Single-dimensional DataBlock
                        element_type_class, is_vector, mlir_element_type = base_type.get_array_element_info()

                        if is_vector:
                            # This is an array of vectors - generate a load to get the vector pointer as i64, then cast to !llvm.ptr
                            result_id = f"%{Node.get_mlir_id()}"
                            load_op = indent + f"{result_id} = memref.load {base_value}[{index}] {{name = \"{base_value.strip('%')}\"}} : memref<?x{mlir_element_type}>\n"
                            Node.context.get_block().mlir_append(load_op)
                            # Now cast the i64 to !llvm.ptr
                            ptr_id = f"%{Node.get_mlir_id()}"
                            cast_op = indent + f"{ptr_id} = llvm.inttoptr {result_id} : {mlir_element_type} to !llvm.ptr\n"
                            Node.context.get_block().mlir_append(cast_op)
                            arg_mlir = ptr_id
                        else:
                            # Regular array of scalars - generate a load to get the scalar value
                            result_id = f"%{Node.get_mlir_id()}"
                            load_op = indent + f"{result_id} = memref.load {base_value}[{index}] {{name = \"{base_value.strip('%')}\"}} : memref<?x{mlir_element_type}>\n"
                            Node.context.get_block().mlir_append(load_op)
                            arg_mlir = result_id
                else:
                    # Handle non-TDataBlock types (like nested memrefs)
                    if str(base_type).startswith("memref<?xmemref<?xf64>>"):
                        # This is an array of vectors (stored as memref of memrefs)
                        result_id = f"%{Node.get_mlir_id()}"
                        load_op = indent + f"{result_id} = memref.load {base_value}[{index}] {{name = \"{base_value.strip('%')}\"}} : memref<?xmemref<?xf64>>\n"
                        Node.context.get_block().mlir_append(load_op)
                        arg_mlir = result_id
                    else:
                        # For other types, just pass the reference
                        arg_mlir = f"{base_value}[{index}]"
                
            argument_prints.append(arg_mlir)
        
        argument_block = Node.context.pop_block().get_mlir()
        for op in argument_block:
            Node.context.get_block().mlir_append(' ' * indent_level * Node.spaces_per_tab + op.strip(' ')) 
        
        # NOTE: we lookup the function pointer name in type_map to get the type
        id = self.id[0].upper() + self.id[1:]
        type_entry = Node.type_map[self.id]
        # Handle both Type objects and strings
        if isinstance(type_entry, str):
            fn_type = type_entry
        else:
            fn_type = type_entry.print_mlir(0).strip()
        
        bridge_function_name = f"{self.id}_bridge"
        call_statement = f"""func.call @{bridge_function_name}({", ".join(argument_prints)}) : {fn_type.strip(' ')}"""
        
        Node.context.get_block().mlir_append(indent + call_statement)
        return '' 

    def print_tree(self, indent_level=0):
        argument_prints = [argument.print_tree() for argument in self._arguments]
        return ' ' * indent_level * Node.spaces_per_tab + f"""FunctionCall:
{' ' * (indent_level + 1) * Node.spaces_per_tab + ",".join(argument_prints)}"""


class Max(Statement):
    def __init__(self, lhs, rhs):
        super().__init__()
        self._lhs = lhs
        self._rhs = rhs
    
    def print_cpp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + f"""std::max({self._lhs.print_cpp()}, {self._rhs.print_cpp()});"""

    def print_omp(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + f"""std::max({self._lhs.print_omp()}, {self._rhs.print_omp()});"""

    def print_sycl(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + f"""::sycl::max({self._lhs.print_sycl()}, {self._rhs.print_sycl()});"""

    def print_mlir(self, indent_level=0):
        indent = ' ' * indent_level * Node.spaces_per_tab
        
        def unpack_nested_memref_access(expr, indent_level):
            indent = ' ' * indent_level * Node.spaces_per_tab
            
            if not (isinstance(expr, Subscript) and isinstance(expr._value, Subscript)):
                return expr.print_mlir(indent_level).strip(' ')
            
            outer_subscript = expr._value
            
            if not hasattr(outer_subscript, '_new_index'):
                outer_subscript.print_mlir(indent_level)
            if not hasattr(expr, '_new_index'):
                expr.print_mlir(indent_level)
            
            outer_index = getattr(outer_subscript, '_new_index', outer_subscript._index.print_mlir(indent_level))
            inner_index = getattr(expr, '_new_index', expr._index.print_mlir(indent_level))
            
            inner_memref_id = f"%{Node.get_mlir_id()}"

            # Get the outer memref's MLIR variable name and check type_map
            outer_memref_var = outer_subscript._value.print_mlir(indent_level).strip(' ')
            
            # Try to determine the outer memref type
            # First, check if this variable is registered in type_map
            try:
                # Get the cpp name to check type_map
                outer_var_cpp = outer_subscript._value.print_cpp().strip()
                if outer_var_cpp in Node.type_map:
                    type_entry = Node.type_map[outer_var_cpp]
                    # Handle both Type objects and strings
                    if isinstance(type_entry, str):
                        outer_memref_type = type_entry
                    else:
                        # Type object - call print_mlir()
                        outer_memref_type = type_entry.print_mlir(0).strip()
                else:
                    # Fallback to default nested memref type
                    outer_memref_type = "memref<?xmemref<?xf64>>"
            except:
                outer_memref_type = "memref<?xmemref<?xf64>>"

            mlir_str = indent + f"{inner_memref_id} = memref.load {outer_memref_var}[{outer_index}] {{name = \"{outer_memref_var.strip('%')}\"}} : {outer_memref_type}\n"
            value_id = f"%{Node.get_mlir_id()}"
            mlir_str += indent + f"{value_id} = memref.load {inner_memref_id}[{inner_index}] {{name = \"{outer_memref_var.strip('%')}\"}} : memref<?xf64>\n"
            Node.context.get_block().mlir_append(mlir_str)
  
            return value_id
        
        Node.context.push_block()
        
        if type(self._lhs.get_type()) is TDataBlock:
            self._lhs.hoist(True)
        if type(self._rhs.get_type()) is TDataBlock:
            self._rhs.hoist(True)
            
        const_zero_id = None
        if isinstance(self._lhs, Subscript) and isinstance(self._lhs._value, Subscript):
            lhs = unpack_nested_memref_access(self._lhs, indent_level)
        else:
            lhs = self._lhs.print_mlir(indent_level).strip(' ')
            if TDataBlock.needs_memref_load(self._lhs):
                const_zero_id = TDataBlock.ensure_const_zero(const_zero_id, indent_level)
                lhs = TDataBlock.generate_memref_load(self._lhs, lhs, const_zero_id, indent_level)
            
        if isinstance(self._rhs, Subscript) and isinstance(self._rhs._value, Subscript):
            rhs = unpack_nested_memref_access(self._rhs, indent_level)
        else:
            rhs = self._rhs.print_mlir(indent_level).strip(' ')
            if TDataBlock.needs_memref_load(self._rhs):
                const_zero_id = TDataBlock.ensure_const_zero(const_zero_id, indent_level)
                rhs = TDataBlock.generate_memref_load(self._rhs, rhs, const_zero_id, indent_level)

        argument_block = Node.context.pop_block().get_mlir()
        for op in argument_block:
            Node.context.get_block().mlir_append(op)  
  
        # Use the unified type determination and operation mapping from TDataBlock
        op_type_class = TDataBlock.get_mlir_type_for_operation(self._lhs, self._rhs)
        result_type = op_type_class.print_mlir(0)  

        if op_type_class is TInteger:
            mlir_op = "arith.maxsi"
        else:
            mlir_op = "arith.maximumf"  # Default to float
            
        new_id = f"%{Node.get_mlir_id()}"
        mlir_str = indent + f"{new_id} = {mlir_op} {lhs}, {rhs} : {result_type}\n"
        Node.context.get_block().mlir_append(mlir_str)         
           
        return new_id

    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + f"""Max:
{self._lhs.print_tree(indent_level + 1)}
{self._rhs.print_tree(indent_level + 1)}"""


class MemoryAllocation(Statement):
    memoryAllocated = []
    allocated_types = {}  # Map from MLIR variable name to allocated memref type string

    def __init__(self,  name: Name, object_type: Type, dimensions, specify_type = True, add_to_stack=True):
        super().__init__()
        self._type = object_type
        if type(dimensions[0]) is list:
            self._dimensions = [dimension[1] for dimension in dimensions]
        else:
            self._dimensions = dimensions
        self._name = name
        self._specify_type = specify_type

        if add_to_stack == True:
            MemoryAllocation.memoryAllocated[-1].append(self)

        if len(self._dimensions) > 1:
            self._size = self._dimensions[-2]
            for i in range(len(self._dimensions) - 3, -1, -1):
                self._size = self._size * self._dimensions[i]

            self._loop = For([Integer(0), self._dimensions[-1]])
            self._loop.add_statement(MemoryAllocation(Subscript(self._name, self._loop.get_iteration_variable()), self._type, [self._size], False, False))
            self._loop.close_scope()
    
    def print_cpp(self, indent_level=0):
        if len(self._dimensions) == 1:
            return ' ' * indent_level * Node.spaces_per_tab + f"""{self._type.print_cpp() + "* " if self._specify_type else ""}{self._name.print_cpp()} = new {self._type.print_cpp()}[{self._dimensions[0].print_cpp()}];"""
        else:
            if Node.use_accelerator == False and type(Name.variables[self._name.id]) is not FaceDataBlock:
                return ' ' * indent_level * Node.spaces_per_tab + f"""{self._type.print_cpp()}** {self._name.print_cpp()} = new {self._type.print_cpp()}*[{self._dimensions[-1].print_cpp()}];
{self._loop.print_cpp(indent_level)}"""
            else:
                return ' ' * indent_level * Node.spaces_per_tab + f"""{self._type.print_cpp()}* {self._name.print_cpp()} = new {self._type.print_cpp()}[{(self._size * self._dimensions[-1]).print_cpp()}];"""

        
    def print_omp(self, indent_level=0):
        if Node.use_accelerator == False:
            if len(self._dimensions) == 1:
                return ' ' * indent_level * Node.spaces_per_tab + f"""{self._type.print_omp() + "* " if self._specify_type else ""}{self._name.print_omp()} = new {self._type.print_omp()}[{self._dimensions[0].print_omp()}];"""
            else:
                return ' ' * indent_level * Node.spaces_per_tab + f"""{self._type.print_omp()}** {self._name.print_omp()} = new {self._type.print_omp()}*[{self._dimensions[-1].print_omp()}];
{self._loop.print_omp(indent_level)}"""
        else:   
            if len(self._dimensions) == 1:
                if Node.use_memory_manager == True:
                    return ' ' * indent_level * Node.spaces_per_tab + f"""{self._type.print_omp() + "* " if self._specify_type else ""}{self._name.print_omp()} = tarch::accelerator::GPUMemoryManager::getInstance().allocate<{self._type.print_omp()}>({self._dimensions[0].print_omp()}, targetDevice);"""
                else:
                    return ' ' * indent_level * Node.spaces_per_tab + f"""{self._type.print_omp() + "* " if self._specify_type else ""}{self._name.print_omp()} = ({self._type.print_omp()}*)omp_target_alloc(sizeof({self._type.print_omp()}) * {self._dimensions[0].print_omp()}, targetDevice);"""
            else:
                if Node.use_memory_manager == True:
                    return ' ' * indent_level * Node.spaces_per_tab + f"""{self._type.print_omp()}* {self._name.print_omp()} = tarch::accelerator::GPUMemoryManager::getInstance().allocate<{self._type.print_omp()}>({(self._size * self._dimensions[-1]).print_omp()}, targetDevice);"""
                else:
                    return ' ' * indent_level * Node.spaces_per_tab + f"""{self._type.print_omp()}* {self._name.print_omp()} = ({self._type.print_omp()}*)omp_target_alloc(sizeof({self._type.print_omp()}) * {(self._size * self._dimensions[-1]).print_omp()}, targetDevice);"""

    def print_sycl(self, indent_level=0):
        if len(self._dimensions) == 1:
            return ' ' * indent_level * Node.spaces_per_tab + f"""{self._type.print_sycl() + "* " if self._specify_type else ""}{self._name.print_sycl()} = new {self._type.print_sycl()}[{self._dimensions[0].print_sycl()}];"""
        else:
            return ' ' * indent_level * Node.spaces_per_tab + f"""{self._type.print_sycl()}** {self._name.print_sycl()} = new {self._type.print_sycl()}*[{self._dimensions[-1].print_sycl()}];
{self._loop.print_sycl(indent_level)}"""
        
    def print_mlir(self, indent_level=0):
        indent = ' ' * indent_level * Node.spaces_per_tab
        
        # Comprehensive tracing for MemoryAllocation
        name_mlir = self._name.print_mlir(0)
        name_type = self._name.get_type()
        is_datablock = isinstance(name_type, TDataBlock)
        var_name_for_map = self._name.print_cpp().strip()
        in_type_map = var_name_for_map in Node.type_map


        if len(self._dimensions) == 1:
            # Process the dimension expression first and collect any intermediate operations
            Node.context.push_block()
            index_val = self._dimensions[0].print_mlir(indent_level)
            intermediate_ops = Node.context.pop_block().get_mlir()
            
            # Now build the allocation operations
            index_id = f"%{Node.get_mlir_id()}"
            decl = ''.join(intermediate_ops)
            
            # Add the index cast and allocation with correct source type
            index_type = self._dimensions[0].get_type()
            index_mlir_type = index_type.print_mlir(0).strip()
            decl += indent + f"{index_id} = arith.index_cast {index_val} : {index_mlir_type} to index\n"
            new_id = f"%{Node.get_mlir_id()}"
            
            # For 1D allocation, we should always generate 1D memref type
            # Get the element type correctly
            name_type = self._name.get_type()
            if isinstance(name_type, TDataBlock):
                # Get the element type of the DataBlock
                element_type_class = name_type.get_element_type()
                element_type = element_type_class().print_mlir(0)
                mlir_type = f"memref<?x{element_type}>"
            else:
                mlir_type = name_type.print_mlir(0)

            decl += indent + f"{new_id} = memref.alloc({index_id}) {{name = \"{name_mlir.strip('%')}\"}} : {mlir_type}\n"
            # Track the allocated type for deallocation
            MemoryAllocation.allocated_types[new_id] = mlir_type
            
            # Check if this is an inner allocation that needs to be stored in an outer array
            # This happens when self._name is a Subscript (indicating arr[i] = allocated_inner_array)
            if isinstance(self._name, Subscript) and not self._specify_type:
                # This is an inner allocation for nested memref, add the store operation
                # Get the outer array name and index
                outer_array = self._name._value  # This should be the outer array name
                outer_index = self._name._index  
                
                # Process the outer array and index expressions to collect any additional operations
                Node.context.push_block()
                outer_array_mlir = outer_array.print_mlir(indent_level)
                outer_index_mlir = outer_index.print_mlir(indent_level)
                additional_ops = Node.context.pop_block().get_mlir()
                
                # Add any additional operations
                decl += ''.join(additional_ops)
                
                # When storing a memref into a nested memref array, we need to use memref<?xmemref<?xf64>>
                type_mlir = Node.get_mlir_type(self._type)
                if type_mlir is None:
                    type_entry = Node.type_map[self._type.print_cpp()]
                    # Handle both Type objects and strings
                    if isinstance(type_entry, str):
                        type_mlir = type_entry.strip(' ')
                    else:
                        # Type object - call print_mlir()
                        type_mlir = type_entry.print_mlir(0).strip()
                outer_type = f"memref<?xmemref<?x{type_mlir}>>"
                
                # Add the store operation
                decl += indent + f"memref.store {new_id}, {outer_array_mlir}[{outer_index_mlir}] {{name = \"{outer_array_mlir.strip('%')}\"}} : {outer_type}"
            
            return decl
        else:
            # For multi-dimensional arrays, check type_map first (ordering is important)
            # This allows overriding default DataBlock behavior for nested memref types
            var_name_for_type_check = self._name.print_cpp().strip()
            in_type_map = var_name_for_type_check in Node.type_map
            
            # Use nested memref if in type_map, otherwise check if DataBlock flat
            if in_type_map:
                # Variable has explicit type specification - use nested memref path
                is_datablock = False  # Force nested path
                actual_dims_calc = -1
            else:
                # For multi-dimensional arrays, check if this is a DataBlock
                # If so, use flat memref with the correct number of dimensions instead of nested memref
                is_datablock = isinstance(self._name.get_type(), TDataBlock)
                
                # Calculate actual_dims based on DataBlock iteration_range
                if is_datablock:
                    datablock_obj = self._name.get_type()

                    if hasattr(datablock_obj, '_iteration_range'):
                        actual_dims_calc = len(datablock_obj._iteration_range)
                    elif isinstance(datablock_obj._dimensions, int):
                        actual_dims_calc = 1 if datablock_obj._dimensions == 1 else 2
                    elif isinstance(datablock_obj._dimensions, list):
                        actual_dims_calc = len(datablock_obj._dimensions) if len(datablock_obj._dimensions) > 0 else 1
                    else:
                        actual_dims_calc = 1
                else:
                    actual_dims_calc = -1  # Not a DataBlock
                
            if is_datablock and actual_dims_calc >= 2:
                # Multi-dimensional DataBlock case: allocate as flat N-D memref
                element_type_class = self._name.get_type().get_element_type()
                element_type = element_type_class().print_mlir(0)

                # We need actual_dims_calc dimensions from self._dimensions
                # For a 2D array: _dimensions might have 2, 3, or 4 elements (depends on iteration_range structure)
                # Take the first actual_dims_calc elements
                dim_exprs = self._dimensions[:actual_dims_calc]
                if len(dim_exprs) < actual_dims_calc:
                    # Fallback: if we don't have enough dimensions, use what we have
                    dim_exprs = self._dimensions
                
                intermediate_decl = ""
                dim_ids = []
                
                for i, dim_expr in enumerate(dim_exprs):
                    # Evaluate the dimension expression - this appends its operations to context
                    dim_val = dim_expr.print_mlir(indent_level)
                    dim_id = f"%{Node.get_mlir_id()}"
                    dim_ids.append(dim_id)
                    dim_type = dim_expr.get_type()
                    dim_mlir_type = dim_type.print_mlir(0).strip()
                    
                    # Generate the index cast (will be appended after dimension expressions)
                    intermediate_decl += indent + f"{dim_id} = arith.index_cast {dim_val} : {dim_mlir_type} to index\n"
                
                # These must come before the index_cast operations
                dim_expr_ops = Node.context.pop_block().get_mlir()
                intermediate_decl = ''.join(dim_expr_ops) + intermediate_decl
                
                # Re-push a new block for remaining operations
                Node.context.push_block()
                
                # Process name and allocate
                name_mlir = self._name.print_mlir(indent_level)
                
                # Build memref type: memref<?x?x...xf64> with actual_dims_calc '?'s
                dimension_list = ["?"] * actual_dims_calc
                memref_dims = "x".join(dimension_list)
                alloc_args = ", ".join(dim_ids)
                
                # Allocate as flat N-D memref
                alloc_line = indent + f"{name_mlir} = memref.alloc({alloc_args}) {{name = \"{name_mlir.strip('%')}\"}} : memref<{memref_dims}x{element_type}>\n"
                
                # Track the allocated memref type so it can be used in load/store operations
                memref_type_full = f"memref<{memref_dims}x{element_type}>"
                Node.mlir_memref_types[name_mlir] = memref_type_full
                MemoryAllocation.allocated_types[name_mlir] = memref_type_full
                
                return intermediate_decl + alloc_line
            else:
                # For non-DataBlock multi-dimensional arrays, use nested memref with loop
                # Use the existing loop structure created in the constructor
                size = self._dimensions[-2]
                for i in range(len(self._dimensions) - 3, -1, -1):
                    size = size * self._dimensions[i]
                    
                # Process the outer dimension expression first and collect any intermediate operations
                Node.context.push_block()
                outer_size_val = self._dimensions[-1].print_mlir(indent_level)
                intermediate_ops = Node.context.pop_block().get_mlir()
                
                # Build the allocation operations
                decl = ''.join(intermediate_ops)
                
                # Add the index cast and allocation with correct source type
                outer_size_id = f"%{Node.get_mlir_id()}"
                outer_size_type = self._dimensions[-1].get_type()
                outer_size_mlir_type = outer_size_type.print_mlir(0).strip()
                decl += indent + f"{outer_size_id} = arith.index_cast {outer_size_val} : {outer_size_mlir_type} to index\n"
                
                # Determine the correct type for the outer array
                # Check if this variable name is in type_map (e.g., maxEigenvaluesX, maxEigenvaluesY)
                var_name = self._name.print_cpp().strip()

                # NOTE: We need to check if the variable name is in type_map first
                if var_name in Node.type_map:
                    # Use the explicitly registered type from type_map
                    type_entry = Node.type_map[var_name]

                    # Handle both Type objects and strings
                    if isinstance(type_entry, str):
                        outer_type = type_entry
                    else:
                        # Type object (e.g., TNestedMemref) 
                        outer_type = type_entry.print_mlir(0).strip()
                elif isinstance(self._name.get_type(), TDataBlock):
                    # Get the element type correctly for the inner memref
                    element_type_class = self._name.get_type().get_element_type()
                    element_type = element_type_class().print_mlir(0)
                    outer_type = f"memref<?xmemref<?x{element_type}>>"
                else:
                    # Fallback for non-DataBlock types
                    name_type_mlir = self._name.get_type().print_mlir(0) 

                    if name_type_mlir is None:
                        type_entry = Node.type_map[self._name.get_type().print_cpp()]
                        # Handle both Type objects and strings
                        if isinstance(type_entry, str):
                            name_type_mlir = type_entry.strip(' ')
                        else:
                            # Type object 
                            name_type_mlir = type_entry.print_mlir(0).strip()
                    outer_type = f"memref<?x{name_type_mlir}>"
                
                # Process the name expression to collect any additional operations
                Node.context.push_block()
                name_mlir = self._name.print_mlir(indent_level)
                name_ops = Node.context.pop_block().get_mlir()
                
                # Add any name-related operations
                decl += ''.join(name_ops)
                decl += indent + f"{name_mlir} = memref.alloc({outer_size_id}) {{name = \"{name_mlir.strip('%')}\"}} : {outer_type}\n"
                # Track the allocated type for deallocation
                MemoryAllocation.allocated_types[name_mlir] = outer_type
                
                # Use the existing loop structure for inner allocations
                loop_mlir = self._loop.print_mlir(indent_level)
                
                return decl + loop_mlir

    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + f"""MemoryAllocation:
{self._name.print_tree(indent_level + 1)}""" 


class MemoryDeallocation(Statement):
    def __init__(self, allocation: MemoryAllocation):
        super().__init__()
        self._dimensions = allocation._dimensions
        self._name = allocation._name
        self._allocated_type = None  # Will store the MLIR type string when known

        if len(self._dimensions) > 1:
            size = self._dimensions[-2]
            for i in range(len(self._dimensions) - 3, -1, -1):
                size = size * self._dimensions[i]

            self._loop = For([Integer(0), self._dimensions[-1]])
            self._loop.add_statement(MemoryDeallocation(MemoryAllocation(Subscript(self._name, self._loop.get_iteration_variable()), None, [size], False, False)))
            self._loop.close_scope()

    def print_cpp(self, indent_level=0):
        if len(self._dimensions) == 1:
            return ' ' * indent_level * Node.spaces_per_tab + f"""delete[] {self._name.print_cpp()};"""
        else:
            if Node.use_accelerator == False and type(Name.variables[self._name.id]) is not FaceDataBlock:
                return f"""{self._loop.print_cpp(indent_level)}
{' ' * indent_level * Node.spaces_per_tab}delete[] {self._name.print_cpp()};"""
            else:
                return f"""{' ' * indent_level * Node.spaces_per_tab}delete[] {self._name.print_cpp()};"""

    def print_omp(self, indent_level=0):
        if Node.use_accelerator == False:
            if len(self._dimensions) == 1:
                return ' ' * indent_level * Node.spaces_per_tab + f"""delete[] {self._name.print_omp()};"""
            else:
                return f"""{self._loop.print_omp(indent_level)}
{' ' * indent_level * Node.spaces_per_tab}delete[] {self._name.print_omp()};"""
        else:
            if Node.use_memory_manager:
                return ' ' * indent_level * Node.spaces_per_tab + f"""tarch::accelerator::GPUMemoryManager::getInstance().free({self._name.print_omp()}, targetDevice);"""
            else:
                return ' ' * indent_level * Node.spaces_per_tab + f"""omp_target_free({self._name.print_omp()}, targetDevice);"""

    def print_sycl(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + f"""::sycl::free({self._name.print_sycl()}, queue);"""
        #else:
        #    return f"""{' ' * indent_level * Node.spaces_per_tab}::sycl::free({self._name.print_sycl()}, queue);"""
#            return f"""{self._loop.print_sycl(indent_level)}
#{' ' * indent_level * Node.spaces_per_tab}::sycl::free({self._name.print_sycl()}, queue);"""

    def print_mlir(self, indent_level=0):
        indent = ' ' * indent_level * Node.spaces_per_tab
        
        # Get the MLIR variable name
        name_mlir = self._name.print_mlir(indent_level)
        
        # Use the stored allocated type if available
        if name_mlir in MemoryAllocation.allocated_types:
            name_type_mlir = MemoryAllocation.allocated_types[name_mlir]
        elif self._allocated_type is not None:
            # Use the allocated type if it was set explicitly
            name_type_mlir = self._allocated_type
        else:
            # Fallback: try to reconstruct the type from the allocation dimensions
            # For nested memrefs (multi-dimensional arrays), we need memref<?xmemref<?xf64>>
            # For 1D arrays, we need memref<?xf64>
            name_element_type = self._name.get_type()
            element_type = name_element_type.print_mlir(0).strip()
            
            if len(self._dimensions) == 1:
                name_type_mlir = f"memref<?x{element_type}>"
            else:
                # For multidimensional, assume nested memref
                name_type_mlir = f"memref<?xmemref<?x{element_type}>>"
        
        return "\n" + indent + f"""memref.dealloc {name_mlir} : {name_type_mlir}\n"""

    def print_tree(self, indent_level=0):
        return ' ' * indent_level * Node.spaces_per_tab + f"""MemoryDeallocation:
{self._name.print_tree(indent_level + 1)}""" 
    

class Construction(Statement):
    def __init__(self, name: Name, expression: Expression):
        super().__init__()
        self._name = name
        self._name.on_left(True)
        self._expression = expression
        Name.variables.update({self._name.id: expression})
    
    def print_cpp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + f"{self._name.get_type().print_cpp()} {self._name.print_cpp()} = {self._expression.print_cpp()};"

    def print_omp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + f"{self._name.get_type().print_omp()} {self._name.print_omp()} = {self._expression.print_omp()};"

    def print_sycl(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + f"{self._name.get_type().print_sycl()} {self._name.print_sycl()} = {self._expression.print_sycl()};"

    def print_mlir(self, indent_level = 0):
        indent = ' ' * indent_level * Node.spaces_per_tab

        if self._expression._string is None:
            expr_mlir = self._expression.print_mlir(indent_level)
            
            # Get any operations that were added to the context
            mlir_str = ''.join(Node.context.get_block().pop_mlir())
            mlir_str += indent + f"{self._name.print_mlir(indent_level)} = {expr_mlir} : {self._name.get_type().print_mlir(indent_level).strip(' ')}\n"
        else:        
            # Get another variable id and load the pointer into that, then load the value into the actual variable (self)
            new_id = Node.get_mlir_id()
            mlir_str =  indent + f"%{new_id} = {self._expression.print_mlir(indent_level)}\n"

            # Use the expression's type, not the name's type
            expr_type = self._expression.get_type().print_mlir(indent_level).strip(' ')
            mlir_str += indent + f"{self._name.print_mlir(indent_level)} = llvm.load %{new_id} {{name = \"{self._name.print_mlir(indent_level).strip('%')}\"}} : !llvm.ptr -> {expr_type}\n"
            
        Node.context.get_block().mlir_append(mlir_str)            
        return ''

    def print_tree(self, indent_level=0):
        return f"""
Construction: 
{self._name.print_tree(indent_level + 1)}
{self._expression.print_tree(indent_level + 1)}"""


class DataBlockConstructionFromExisting(Statement):
    def __init__(self, name: Name, dataBlock: DataBlock):
        super().__init__()
        self._name = name
        self._dataBlock = dataBlock
        self._dataBlock.id = name.id
        Name.variables.update({self._name.id: self._dataBlock})
        FunctionDefinition._syclDataToCopy.append(self)

    def print_cpp(self, indent_level = 0):
        type_print = self._name.get_type().print_cpp()
        if len(self._dataBlock._iteration_range) > 1 and Node.use_accelerator == False and type(self._dataBlock) is not FaceDataBlock:
            type_print += "*"
        return ' ' * indent_level * Node.spaces_per_tab + f"{type_print} {self._name.print_cpp()} = {self._dataBlock._internal.print_cpp()};"
    
    def print_omp(self, indent_level = 0):
        type_print = self._name.get_type().print_omp()
        if len(self._dataBlock._iteration_range) > 1 and Node.use_accelerator == False and type(self._dataBlock) is not FaceDataBlock:
            type_print += "*"
        return ' ' * indent_level * Node.spaces_per_tab + f"{type_print} {self._name.print_omp()} = {self._dataBlock._internal.print_omp()};"
    
    def print_sycl(self, indent_level = 0):
        type_print = self._name.get_type().print_sycl()
        if len(self._dataBlock._iteration_range) > 1 and Node.use_accelerator == False and type(self._dataBlock) is not FaceDataBlock:
            type_print += "*"
        return ' ' * indent_level * Node.spaces_per_tab + f"{type_print} {self._name.print_sycl()} = {self._dataBlock._internal.print_sycl()};"
#            step_size = Integer(1)
#            for d in self._dataBlock._iteration_range[0:-1]:
#                step_size = step_size * (d[1] - d[0])
#            size = step_size * (self._dataBlock._iteration_range[-1][1] - self._dataBlock._iteration_range[-1][0])
#            return f"""{' ' * indent_level * Node.spaces_per_tab}auto {self._name.print_sycl()} = ::sycl::malloc_shared<double>({size.print_sycl()}, queue);
#{' ' * indent_level * Node.spaces_per_tab}for (int z = 0; z < {(self._dataBlock._iteration_range[-1][1]-self._dataBlock._iteration_range[-1][0]).print_sycl()}; z++){{
#""" + (f"""{' ' * (indent_level + 1) * Node.spaces_per_tab}queue.memcpy(&{self._name.print_sycl()}[{step_size.print_sycl()} * z], {self._dataBlock._internal.print_sycl()}[z], {step_size.print_sycl()} * sizeof({self._dataBlock._underlying_type.print_sycl()})).wait();""" if self._dataBlock._internal.print_sycl()[-4:] != "QOut" else """""") + f"""
#{' ' * indent_level * Node.spaces_per_tab}}}
#"""
#            for d in self._dataBlock._iteration_range[0:-1]:
#                size = size * (d[1] - d[0])
#            return f"""{' ' * indent_level * Node.spaces_per_tab}auto {self._name.print_sycl()} = ::sycl::malloc_shared<double*>({(self._dataBlock._iteration_range[-1][1]-self._dataBlock._iteration_range[-1][0]).print_sycl()}, queue);
#{' ' * indent_level * Node.spaces_per_tab}for (int z = 0; z < {(self._dataBlock._iteration_range[-1][1]-self._dataBlock._iteration_range[-1][0]).print_sycl()}; z++){{
#{' ' * (indent_level + 1) * Node.spaces_per_tab}{self._name.print_sycl()}[z] = ::sycl::malloc_shared<double>({size.print_sycl()}, queue);
#""" + (f"""{' ' * (indent_level + 1) * Node.spaces_per_tab}queue.memcpy({self._name.print_sycl()}[z], {self._dataBlock._internal.print_sycl()}[z], {size.print_sycl()} * sizeof({self._dataBlock._underlying_type.print_sycl()})).wait();""" if self._dataBlock._internal.print_sycl()[-4:] != "QOut" else """""") + f"""
#{' ' * indent_level * Node.spaces_per_tab}}}
#"""
        return ' ' * indent_level * Node.spaces_per_tab + f"{type_print} {self._name.print_sycl()} = {self._dataBlock._internal.print_sycl()};"
    
    def print_mlir(self, indent_level = 0):
        indent = ' ' * indent_level * Node.spaces_per_tab
        type_print = self._name.get_type().print_mlir(indent_level)
        
        dims = []
        for i in range(len(self._dataBlock._iteration_range) - 3, -1, -1):
                dims.append(self._dataBlock._iteration_range[i])
                
        if len(self._dataBlock._iteration_range) > 1:
            type_print = "!llvm.ptr " #+ f"{self._dataBlock._underlying_type} dims: {dims})"
    
        # NOTE: We encode structs as Strings, so check here if they are a struct and generate accordingly
        # Set the target variable name so String.print_mlir() can use it
        target_var_name = self._name.print_mlir(indent_level)
        Node.target_variable_name = target_var_name
        
        # Store the mapping from DSL name to MLIR variable name for later use in assignments
        # TODO: This is clumsy - it is assuming the exact flow here...
        dsl_name = self._dataBlock._internal._value
        Node.mlir_symbol_table[dsl_name] = target_var_name

        # Generate the struct field access operations with the target variable name
        struct_mlir = self._dataBlock._internal.print_mlir(indent_level).strip(' ')
        mlir_str = ''.join(Node.context.pop_block().get_mlir())
        
        # Clear the target variable name - we've finished generating the MLIR for this Node
        Node.target_variable_name = None
        
        return mlir_str        
        

    def print_tree(self, indent_level=0):
        return f"""
DataBlockConstructionFromExisting: 
{self._name.print_tree(indent_level + 1)}
{self._dataBlock._internal.print_tree(indent_level + 1)}"""
    

class DataBlockConstructionFromOperation:
    def __init__(self, name: Name, dataBlockOperation):
        super().__init__()
        self._name = name
        self._dataBlock = DataBlock(dataBlockOperation._iteration_range, None, False, self._name.id, underlying_type=dataBlockOperation.get_type().get_single())
        Name.variables.update({self._name.id: self._dataBlock})
        if type(dataBlockOperation) is DataBlockBinaryOperation or type(dataBlockOperation) is DataBlockUnaryOperation or type(dataBlockOperation) is DataBlockMax or type(dataBlockOperation) is DataBlockComparison or type(dataBlockOperation._internal) is String:
            self._operation = dataBlockOperation
        else:
            self._operation = dataBlockOperation._internal

        self.memoryAllocation = MemoryAllocation(self._name, self._dataBlock.get_type().get_single(), self._dataBlock._memory_range)
        self.assignment = DataBlockAssignment(self._name, self._operation)
        
    def print_cpp(self, indent_level = 0):
        return f"""{self.memoryAllocation.print_cpp(indent_level)}
{self.assignment.print_cpp(indent_level)}"""
    
    def print_omp(self, indent_level = 0):
        return f"""{self.memoryAllocation.print_omp(indent_level)}
{self.assignment.print_omp(indent_level)}"""

    def print_sycl(self, indent_level = 0):
        return f"""{self.memoryAllocation.print_sycl(indent_level)}
{self.assignment.print_sycl(indent_level)}"""
    
    def print_mlir(self, indent_level = 0):
        return f"""{self.memoryAllocation.print_mlir(indent_level)}
{self.assignment.print_mlir(indent_level)}"""
    
    def print_tree(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + f"""DataBlockConstructionFromOperation:
{self._name.print_tree(indent_level + 1)}
{self._dataBlock.print_tree(indent_level + 1)}"""


class DataBlockConstructionFromMatrixVectorProduct:
    def __init__(self, name: Name, matrix, vector):
        self._name = name
        if type(vector) is Name:
            self._vector = Name.variables[vector.id]
        else:
            self._vector = vector

        if type(matrix) is Name:
            self._matrix = Name.variables[matrix.id]
        else:
            self._matrix = matrix

        self._dataBlock = DataBlock(self._vector._iteration_range, None, False, self._name.id, underlying_type=vector.get_type().get_single())
        Name.variables.update({self._name.id: self._dataBlock})

        self.memoryAllocation = MemoryAllocation(self._name, self._dataBlock.get_type().get_single(), self._dataBlock._memory_range)

        element_width = self._vector._iteration_range[0][1] - self._vector._iteration_range[0][0]

        loops = []
        loops.append(For([Integer(0), self._dataBlock._iteration_range[-1][1] - self._dataBlock._iteration_range[-1][0]]))
        loops.append(For([Integer(0), self._matrix._dimensions[0]]))
        loops.append(For([Integer(0), self._dataBlock._iteration_range[0][1] - self._dataBlock._iteration_range[0][0]]))

        if Node.use_accelerator:
            size = Integer(1)
            for [a, b] in self._dataBlock._memory_range[0:-1]:
                size = size * (b - a)
            index = size * loops[0].get_iteration_variable() + (loops[1].get_iteration_variable() * element_width + loops[2].get_iteration_variable())
            loops[2].add_statement(Assignment(Subscript(self._dataBlock, index), Float(0.0)))
        else:
            loops[2].add_statement(Assignment(Subscript(Subscript(self._dataBlock, loops[0].get_iteration_variable()), loops[1].get_iteration_variable() * element_width + loops[2].get_iteration_variable()), Float(0.0)))
        
        loops.append(For([Integer(0), self._matrix._dimensions[1]]))

        if Node.use_accelerator:
            size = Integer(1)
            for [a, b] in self._dataBlock._memory_range[0:-1]:
                size = size * (b - a)
            index = size * loops[0].get_iteration_variable() + (loops[1].get_iteration_variable() * element_width + loops[2].get_iteration_variable())
            loops[3].add_statement(Assignment(Subscript(self._dataBlock, index), BinaryOperation("+", Subscript(self._dataBlock, index), BinaryOperation("*", matrix.index(loops[1].get_iteration_variable(), loops[3].get_iteration_variable()), Subscript(self._vector, loops[0].get_iteration_variable() * size + element_width * loops[3].get_iteration_variable() + loops[2].get_iteration_variable())))))
        else:
            loops[3].add_statement(Assignment(Subscript(Subscript(self._dataBlock, loops[0].get_iteration_variable()), loops[1].get_iteration_variable() * element_width + loops[2].get_iteration_variable()), BinaryOperation("+", Subscript(Subscript(self._dataBlock, loops[0].get_iteration_variable()), loops[1].get_iteration_variable() * element_width + loops[2].get_iteration_variable()), BinaryOperation("*", matrix.index(loops[1].get_iteration_variable(), loops[3].get_iteration_variable()), Subscript(Subscript(self._vector, loops[0].get_iteration_variable()), element_width * loops[3].get_iteration_variable() + loops[2].get_iteration_variable())))))
        

        
        loops[2].add_statement(loops[3])
        loops[1].add_statement(loops[2])
        loops[0].add_statement(loops[1])
        loops[3].close_scope()
        loops[2].close_scope()
        loops[1].close_scope()
        loops[0].close_scope()
        self.assignment = loops[0]
        
    def print_cpp(self, indent_level = 0):
        return f"""{self.memoryAllocation.print_cpp(indent_level)}
{self.assignment.print_cpp(indent_level)}"""
    
    def print_omp(self, indent_level = 0):
        return f"""{self.memoryAllocation.print_omp(indent_level)}
{self.assignment.print_omp(indent_level)}"""

    def print_sycl(self, indent_level = 0):
        return f"""{self.memoryAllocation.print_sycl(indent_level)}
{self.assignment.print_sycl(indent_level)}"""
    
    def print_mlir(self, indent_level = 0):
        return f"""{self.memoryAllocation.print_mlir(indent_level)}
{self.assignment.print_mlir(indent_level)}"""
    
    def print_tree(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + f"""DataBlockConstructionFromOperation:
{self._name.print_tree(indent_level + 1)}
{self._dataBlock.print_tree(indent_level + 1)}"""


class DataBlockConstructionFromDotProduct:
    def __init__(self, name: Name, matrix, vector):
        self._name = name
        if type(vector) is Name:
            self._vector = Name.variables[vector.id]
        else:
            self._vector = vector

        if type(matrix) is Name:
            self._matrix = Name.variables[matrix.id]
        else:
            self._matrix = matrix

        iteration_range = self._vector._iteration_range
        iteration_range[-1][1] = Integer(1)
        self._dataBlock = DataBlock(self._vector._iteration_range, None, False, self._name.id, underlying_type=vector.get_type().get_single())
        Name.variables.update({self._name.id: self._dataBlock})

        self.memoryAllocation = MemoryAllocation(self._name, self._dataBlock.get_type().get_single(), self._dataBlock._memory_range)

        loops = []
        index = []
        for [a, b] in self._dataBlock._iteration_range[-1::-1]:
            loops.append(For([Integer(0), b - a]))

        for loop in loops[::-1]:
            index.append(loop.get_iteration_variable())

        loops[-1].add_statement(Assignment(self._dataBlock.multidimensional_index(index), Float(0.0)))
        for i in range(len(loops) - 1):
            loops[i].add_statement(loops[i + 1])

        loops.append(For([Integer(0), self._matrix._width]))
        rhs_index = index.copy()
        rhs_index[-1] = loops[-1].get_iteration_variable()
        loops[-1].add_statement(Assignment(self._dataBlock.multidimensional_index(index), BinaryOperation("+", self._dataBlock.multidimensional_index(index), BinaryOperation("*", Subscript(self._matrix, rhs_index[-1]), self._vector.multidimensional_index(rhs_index)))))
        loops[-2].add_statement(loops[-1])

        for loop in loops:
            loop.close_scope()
        self.assignment = loops[0]
        
    def print_cpp(self, indent_level = 0):
        return f"""{self.memoryAllocation.print_cpp(indent_level)}
{self.assignment.print_cpp(indent_level)}"""
    
    def print_omp(self, indent_level = 0):
        return f"""{self.memoryAllocation.print_omp(indent_level)}
{self.assignment.print_omp(indent_level)}"""

    def print_sycl(self, indent_level = 0):
        return f"""{self.memoryAllocation.print_sycl(indent_level)}
{self.assignment.print_sycl(indent_level)}"""
    
    def print_mlir(self, indent_level = 0):
        return f"""{self.memoryAllocation.print_mlir(indent_level)}
{self.assignment.print_mlir(indent_level)}"""
    
    def print_tree(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + f"""DataBlockConstructionFromOperation:
{self._name.print_tree(indent_level + 1)}
{self._dataBlock.print_tree(indent_level + 1)}"""


class DataBlockConstructionFromFunction(Statement):
    def __init__(self, name: Name, dataBlock: DataBlock):
        super().__init__()
        self._name = name
        self._dataBlock = dataBlock
        self._dataBlock.id = name.id
        Name.variables.update({self._name.id: self._dataBlock})

        self.memoryAllocation = MemoryAllocation(self._name, self._dataBlock.get_type().get_single(), self._dataBlock._memory_range)
        self.initialisation = For(self._dataBlock._memory_range[-1])
        
        inner_loops = []
        for i in range(len(self._dataBlock._iteration_range) - 2, 0, -1):
            loop_bound = self._dataBlock._memory_range[i][1] - self._dataBlock._memory_range[i][0]
            inner_loop = For([Integer(0), loop_bound])
            inner_loops.append(inner_loop)
        if type(self._dataBlock._internal) is not FunctionCall:
            inner_loop = For(self._dataBlock._memory_range[0])
            inner_loops.append(inner_loop)

        index = []
        for i in range(len(inner_loops)):
            index.append(inner_loops[-(i + 1)].get_iteration_variable())
        index.append(self.initialisation.get_iteration_variable())
        
        output_offset = [Integer(0) for i in self._dataBlock._iteration_range[1:]]
        if type(self._dataBlock._internal._arguments[0]) is Index:
            input_offset = [0 for i in range(1, len(self._dataBlock._iteration_range))]
        else:
            input_offset = [self._dataBlock._iteration_range[i][0] - Name.variables[self._dataBlock._internal._arguments[0].id]._iteration_range[i][0] for i in range(1, len(self._dataBlock._iteration_range))]

        if type(self._dataBlock._internal) is FunctionCall:
            functionCall = FunctionCall(self._dataBlock._internal.id, [])
            functionCall._is_offloadable = self._dataBlock._internal._is_offloadable
            
            if type(self._dataBlock._internal._arguments[0]) is Index:
                functionCall.add_argument(Vector(index[0:-1], TInteger()))
            else:
                functionCall.add_argument(Reference(self._dataBlock._internal._arguments[0].multidimensional_index(index, 1)))

            for argument in self._dataBlock._internal._arguments[1:]:
                if self._dataBlock._internal.id == "volumetricSource" and getattr(argument, "id", None) == "x" and "h" in Name.variables:
                    # The source term is position-dependent, i.e. it has to see the centre of
                    # the volume it is evaluated for, not the centre of the patch. index holds
                    # the loop variables of the spatial axes (x first) followed by the one over
                    # the patches. The loops start from 0, while the iteration range starts at
                    # the first volume that the source term is evaluated for.
                    spatial_dimensions = len(self._dataBlock._iteration_range) - 2
                    # The kernels name the number of volumes per patch axis differently: NV (Finite Volumes) or NC (Finite Differences)
                    volumes_per_axis = "NV" if "NV" in Name.variables else "NC"
                    volume_index = ", ".join(
                        f"({index[d].print_cpp()} + {self._dataBlock._iteration_range[d + 1][0].print_cpp()})"
                        for d in range(spatial_dimensions)
                    )
                    functionCall.add_argument(String(
                        f"::exahype2::fv::getVolumeCentre({argument.multidimensional_index(index, 1).print_cpp()}, "
                        f"{Name.variables['h'].multidimensional_index(index, 1).print_cpp()}, {volumes_per_axis}, "
                        f"tarch::la::Vector<{spatial_dimensions},int>({{{volume_index}}}))"
                    ))
                elif type(argument.get_type()) is TDataBlock:
                    offset = [i[0] for i in Name.variables[argument.id]._iteration_range[1:]]
                    if len(Name.variables[argument.id]._iteration_range) == 1 or Name.variables[argument.id]._iteration_range[0][1] == Integer(1):
                        functionCall.add_argument(Name.variables[argument.id].multidimensional_index(index, 1))
                    else:
                        functionCall.add_argument(Reference(Name.variables[argument.id].multidimensional_index(index, 1)))
                else:
                    functionCall.add_argument(argument)
        
            functionCall.add_argument(Reference(self._dataBlock.multidimensional_index(index, 1)))
            if functionCall._is_offloadable:
                functionCall.add_argument(String("::exahype2::Solver::Offloadable::Yes"))
        else:
            functionCall = Assignment(self._dataBlock.multidimensional_index(index), self._dataBlock._internal)
        
        self.initialisation.add_statement(inner_loops[0])

        for i in range (0, len(inner_loops) - 1):
            inner_loops[i].add_statement(inner_loops[i + 1])
        inner_loops[-1].add_statement(functionCall)

        for i in range(len(inner_loops) - 1, -1, -1):
            inner_loops[i].close_scope()
        self.initialisation.close_scope()
        self._loops = inner_loops
        self._loops.insert(0, self.initialisation)

    def print_cpp(self, indent_level = 0):
        return f"""{self.memoryAllocation.print_cpp(indent_level)}
{self.initialisation.print_cpp(indent_level)}"""
    
    def print_omp(self, indent_level = 0):
        if Node.use_accelerator == True:
            omp_pragma = "#pragma omp target teams loop collapse(Dimensions + 1) device(targetDevice)"
        else:
            omp_pragma = "#pragma omp parallel for simd collapse(Dimensions + 1)"
        return f"""{self.memoryAllocation.print_omp(indent_level)}
{' ' * indent_level * Node.spaces_per_tab + omp_pragma}
{self.initialisation.print_omp(indent_level)}"""
    
    def print_sycl(self, indent_level = 0):
        ranges = [f"range{i} = {loop.get_interval_size().print_sycl()};" for i, loop in enumerate(self._loops[0:3])]
        indexing = [f"int {loop.get_iteration_variable().print_sycl()} = index[{i}];" for i, loop in enumerate(self._loops[0:3])]
        statement_prints = [statement.print_sycl(indent_level + 1) for statement in self._loops[2]._statements]
        return self.memoryAllocation.print_sycl(indent_level) + os.linesep.join(ranges) + "\n" + ' ' * indent_level * Node.spaces_per_tab + f"""queue.submit([&](::sycl::handler& handler) {{
{' ' * indent_level * Node.spaces_per_tab}handler.parallel_for(::sycl::range<3>{{range0, range1, range2}}, [=](::sycl::item<3> index) {{
{os.linesep.join(indexing)}
{os.linesep.join(statement_prints)}
{' ' * (indent_level + 1) * Node.spaces_per_tab}}});
{' ' * indent_level * Node.spaces_per_tab}}}).wait();"""


    def print_mlir(self, indent_level = 0):
        Node.context.push_block()
        mem_alloc = self.memoryAllocation.print_mlir(indent_level)
        init = self.initialisation.print_mlir(indent_level)
        alloc_block = Node.context.pop_block().get_mlir()
        for op in alloc_block:
            Node.context.get_block().mlir_append(op) 
            
        return f"""{mem_alloc}
{init}"""

    def print_tree(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + f"""DataBlockConstructionFromFunction:
{self._name.print_tree(indent_level + 1)}
{self._dataBlock.print_tree(indent_level + 1)}"""

class Assignment(Statement):
    def __init__(self, lhs: Expression, rhs: Expression):
        super().__init__()
        self._lhs = lhs
        self._lhs.on_left(True)
        self._rhs = rhs
        self._rhs.on_left(False) # Just to be sure...
    
    def print_cpp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + f"{self._lhs.print_cpp()} = {self._rhs.print_cpp()};"
    
    def print_omp(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + f"{self._lhs.print_omp()} = {self._rhs.print_omp()};"

    def print_sycl(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + f"{self._lhs.print_sycl()} = {self._rhs.print_sycl()};"

    def print_mlir(self, indent_level = 0):
        indent = ' ' * indent_level * Node.spaces_per_tab 
        mlir_str = ''
        Node.context.push_block()
        rhs = self._rhs.print_mlir(indent_level)
        
        rhs_indices = []
        rhs_dims = []
        rhs_casting_mlir = ''
                
        if isinstance(self._rhs, Subscript):
            mlir_list = Node.context.get_block().get_mlir()
            for i in range(self._rhs._dimensions):
                rhs_dims.append('?')
                if len(mlir_list) > 0:
                    rhs_indices.insert(0, self._rhs._new_index)
                    rhs_casting_mlir += mlir_list.pop()

            mlir_str += ''.join(Node.context.pop_block().get_mlir())
            mlir_str += rhs_casting_mlir 
        else:
            # Assume scalar here
            tmp = ''.join(Node.context.pop_block().get_mlir())        
            mlir_str += tmp

        Node.context.push_block()
        lhs = self._lhs.print_mlir(indent_level)
        
        lhs_indices = []
        lhs_dims = []
        lhs_casting_mlir = ''
        
        if isinstance(self._lhs, Subscript):
            mlir_list = Node.context.get_block().get_mlir()
            for i in range(self._lhs._dimensions):
                lhs_dims.append('?')
                if len(mlir_list) > 0:
                    popped = mlir_list.pop()
                    lhs_indices.insert(0, self._lhs._new_index)
                    lhs_casting_mlir += popped

            mlir_str += ''.join(Node.context.pop_block().get_mlir())
            mlir_str += lhs_casting_mlir
        else:
            tmp = ''.join(Node.context.pop_block().get_mlir())
            mlir_str += tmp
        
        # Check if LHS is a flat 2D memref (Subscript[Subscript] on String-based field)
        lhs_is_flat_2d = False
        lhs_memref_name = None
        all_lhs_indices = []  
        
        if isinstance(self._lhs, Subscript) and isinstance(self._lhs._value, Subscript):
            # Check if the base is a String-based field
            base_base = self._lhs._value._value
            if isinstance(base_base, DataBlock) and isinstance(base_base._internal, String):
                # This is a String-based field which is now a flat 2D memref
                lhs_is_flat_2d = True
                dsl_name = base_base._internal._value
                lhs_memref_name = Node.mlir_symbol_table[dsl_name]

                outer_sub = self._lhs
                inner_sub = self._lhs._value
                
                if hasattr(inner_sub, '_new_index'):
                    all_lhs_indices.append(inner_sub._new_index)
                if hasattr(outer_sub, '_new_index'):
                    all_lhs_indices.append(outer_sub._new_index)
        
        if lhs_is_flat_2d:
            rhs_value = rhs
            
            lhs_index_str = '[' + ','.join(all_lhs_indices) + ']'
            lhs_type = f"memref<?x?xf64>"
            mlir_str += indent + f"memref.store {rhs_value}, {lhs_memref_name}{lhs_index_str} {{name = \"{lhs_memref_name.strip('%')}\"}} : {lhs_type}\n"
        elif isinstance(self._lhs, Subscript) and isinstance(self._lhs._value, Subscript):
            base_base = self._lhs._value._value
            
            # If base is not a String field, it's a true nested memref (e.g. lambdaLeft[i][j])
            if not (isinstance(base_base, DataBlock) and isinstance(base_base._internal, String)):
                # This is a nested memref store (e.g., lambdaLeft[i][j] = value)
                rhs_value = rhs
                if isinstance(self._rhs, Float):
                    const_id = f"%{Node.get_mlir_id()}"
                    mlir_str += indent + f"{const_id} = arith.constant {self._rhs._value} : f64\n"
                    rhs_value = const_id
                elif isinstance(self._rhs, Integer):
                    const_id = f"%{Node.get_mlir_id()}"
                    mlir_str += indent + f"{const_id} = arith.constant {self._rhs._value} : i32\n"
                    rhs_value = const_id
                elif isinstance(self._rhs, Subscript):
                    rhs_id = f"%{Node.get_mlir_id()}"
                    rhs_index_str = '[' + ','.join(rhs_indices) + ']'
                    rhs_type = "memref<?xf64>" if not rhs_dims else f"memref<{ 'x'.join(rhs_dims) }xf64>"
                    mlir_str += indent + f"{rhs_id} = memref.load {rhs}{rhs_index_str} {{name = \"{rhs.strip('%')}\"}} : {rhs_type.strip(' ')}\n"
                    rhs_value = rhs_id
                
                # Extract indices from nested subscript chain
                # lhs._value is the inner subscript, lhs is the outer subscript
                outer_index = self._lhs._new_index           
                inner_memref_var = self._lhs._value._value.print_mlir(indent_level)  # Base memref name
                
                # Load the inner memref from the outer nested structure
                inner_memref_id = f"%{Node.get_mlir_id()}"
                # Get the actual outer memref type from type_map
                base_var = self._lhs._value._value
                type_entry = Node.type_map[base_var.id]

                if isinstance(type_entry, str):
                    outer_memref_type = type_entry
                else:
                    outer_memref_type = type_entry.print_mlir(0).strip()

                # Load inner memref with first index
                inner_sub_index = self._lhs._value._new_index
                mlir_str += indent + f"{inner_memref_id} = memref.load {inner_memref_var}[{inner_sub_index}] {{name = \"{inner_memref_var.strip('%')}\"}} : {outer_memref_type}\n"
                
                # Store directly into the inner memref with the second index
                inner_memref_type = "memref<?xf64>"  # Default inner arrays are 1D
                mlir_str += indent + f"memref.store {rhs_value}, {inner_memref_id}[{outer_index}] {{name = \"{inner_memref_var.strip('%')}\"}} : {inner_memref_type}\n"
        else:
            # Handle regular flat memref assignment (non-nested)
            rhs_id = f"%{Node.get_mlir_id()}"         

            lhs_index_str = '[' + ','.join(lhs_indices) + ']'
            # Ensure at least one dimension for memref type
            if lhs_dims:
                lhs_type = f"memref<{ 'x'.join(lhs_dims) }xf64>"
            else:
                lhs_type = "memref<?xf64>"
            rhs_index_str = '[' + ','.join(rhs_indices) + ']'            
            if rhs_dims:
                rhs_type = f"memref<{ 'x'.join(rhs_dims) }x f64>"
            else:
                rhs_type = "memref<?xf64>"
            
            if isinstance(self._rhs, Subscript):
                mlir_str += indent + f"{rhs_id} = memref.load {rhs}{rhs_index_str} {{name = \"{rhs.strip('%')}\"}} : {rhs_type.strip(' ')}\n"
                rhs_value = rhs_id
            else:
                rhs_value = rhs

            if isinstance(self._rhs, Float):
                const_id = f"%{Node.get_mlir_id()}"
                mlir_str += indent + f"{const_id} = arith.constant {self._rhs._value} : f64\n"
                rhs_value = const_id
            elif isinstance(self._rhs, Integer):
                const_id = f"%{Node.get_mlir_id()}"
                mlir_str += indent + f"{const_id} = arith.constant {self._rhs._value} : i32\n"
                rhs_value = const_id

            if isinstance(self._lhs, Subscript):
                mlir_str += indent + f"memref.store {rhs_value}, {lhs}{lhs_index_str} {{name = \"{lhs.strip('%')}\"}} : {lhs_type.strip(' ')}\n"
            else:
                lhs_type_obj = self._lhs.get_type()
                lhs_mlir_type = lhs_type_obj.print_mlir(indent_level).strip(' ')
                mlir_str += indent + f"memref.store {rhs_value}, {lhs} {{name = \"{lhs.strip('%')}\"}} : {lhs_mlir_type}\n"
                
        return mlir_str
    
    def print_tree(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + f"""Assignment:
{self._lhs.print_tree(indent_level + 1)}
{self._rhs.print_tree(indent_level + 1)}"""
    

class DataBlockAssignment(Statement):
    def __init__(self, lhs: Expression, rhs: Expression):
        super().__init__()
        if type(lhs) is Name:
            self._lhs = Name.variables[lhs.id]
            self._lhs._id = lhs.id
        else:
            self._lhs = lhs
        
        if type(rhs) is Name:
            self._rhs = Name.variables[rhs.id]
        else:
            self._rhs = rhs

        self._lhs.on_left(True)    
        self._rhs.on_left(False)	# Just to be sure...

        self._loops = []
        index = []

        self._loops.append(For([Integer(0), self._lhs._iteration_range[-1][1] - self._lhs._iteration_range[-1][0]]))
        for i in range(len(self._lhs._memory_range) - 2, -1, -1):
            self._loops.append(For([Integer(0), self._lhs._iteration_range[i][1] - self._lhs._iteration_range[i][0]]))
       
        for i in range(len(self._lhs._memory_range) - 1, -1, -1):
            index.append(self._loops[i].get_iteration_variable())              

        if type(self._rhs.get_type()) is TDataBlock:
            rhs_offset = []
            for i in range(len(self._lhs._iteration_range)):
                lhs_start = self._lhs._iteration_range[i][0]
                rhs_start = self._rhs._iteration_range[i][0]
                if type(rhs_start.get_type()) is TDataBlock:
                    rhs_start = Subscript(rhs_start, lhs_start)
                rhs_offset.append(lhs_start - rhs_start)
        else:
            rhs_offset = [Integer(0) for i in self._lhs._iteration_range]

        lhs_offset = [Integer(0) for i in self._lhs._iteration_range]

        self._loops[-1].add_statement(Assignment(self._lhs.multidimensional_index(index, dimensions=self._lhs._memory_range), self._rhs.multidimensional_index(index, dimensions=self._lhs._memory_range)))
        for i in range(0, len(self._loops) - 1):
            self._loops[i].add_statement(self._loops[i + 1])

        for loop in self._loops[::-1]:
            loop.close_scope()
        self._loop = self._loops[0]
        self._numDimensions = len(self._loops)
    
    def print_cpp(self, indent_level = 0):
        return self._loop.print_cpp(indent_level)
    
    def print_omp(self, indent_level = 0):
        if Node.use_accelerator == True:
            omp_pragma = "#pragma omp target teams loop collapse(Dimensions + 1) device(targetDevice)"
        else:
            omp_pragma = "#pragma omp parallel for simd collapse(Dimensions + 1)"
        return ' ' * indent_level * Node.spaces_per_tab + omp_pragma + f"""
{self._loop.print_omp(indent_level)}"""
    
    def print_sycl(self, indent_level = 0):
        #return self._loop.print_sycl(indent_level)
        ranges = [f"range{i} = {loop.get_interval_size().print_sycl()};" for i, loop in enumerate(self._loops[0:3])]
        indexing = [f"int {loop.get_iteration_variable().print_sycl()} = index[{i}];" for i, loop in enumerate(self._loops[0:3])]
        statement_prints = [statement.print_sycl(indent_level + 1) for statement in self._loops[2]._statements]
        return os.linesep.join(ranges) + "\n" + ' ' * indent_level * Node.spaces_per_tab + f"""queue.submit([&](::sycl::handler& handler) {{
{' ' * indent_level * Node.spaces_per_tab}handler.parallel_for(::sycl::range<3>{{range0, range1, range2}}, [=](::sycl::item<3> index) {{
{os.linesep.join(indexing)}
{os.linesep.join(statement_prints)}
{' ' * (indent_level + 1) * Node.spaces_per_tab}}});
{' ' * indent_level * Node.spaces_per_tab}}}).wait();"""


    def print_mlir(self, indent_level = 0):
        return self._loop.print_mlir(indent_level)
    
    def print_tree(self, indent_level = 0):
        return ' ' * indent_level * Node.spaces_per_tab + f"""DataBlockAssignment:
{self._lhs.print_tree(indent_level + 1)}
{self._rhs.print_tree(indent_level + 1)}"""
