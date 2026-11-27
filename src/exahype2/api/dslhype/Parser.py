import ast_comments as ast, inspect
from exahype2.api.dslhype import SyntaxTree
import copy

class Parser:
    def _parse_constant(self, constant: ast.Constant, type_name = None):
        if type(constant.value) is int:
            return SyntaxTree.Integer(constant.value)
        if type(constant.value) is str:
            if constant.value.startswith("SolverType::") and SyntaxTree.FunctionDefinition._stateless == False:
                return SyntaxTree.String(constant.value[len("SolverType::"):])
            return SyntaxTree.String(constant.value)
        if type(constant.value) is float:
            return SyntaxTree.Float(constant.value)
        if type(constant.value) is bool:
            return SyntaxTree.Boolean(constant.value)

    def _parse_call(self, call: ast.Call):
        function = call.func
        if type(function) is ast.Name:
            if function.id == "DataBlock":
                return self._parse_patch_constructor(call)
            if function.id == "FaceDataBlock":
                return self._parse_patch_constructor(call, isFace = True)
            elif function.id == "Vector":
                return SyntaxTree.Vector([], SyntaxTree.String(""), call.args[0].value)
            elif function.id == "DiagonalMatrix":
                return self._parse_diagonal_matrix_constructor(call)
            elif function.id == "Matrix":
                return self._parse_matrix_constructor(call)
            elif function.id == "kroneckerProduct":
                return self._parse_kronecker_product(call)
            elif function.id == "Integer":
                return SyntaxTree.Integer(0, call.args[0].value)
            elif function.id == "Float":
                return SyntaxTree.Float(0, call.args[0].value)
            elif function.id == "FloatReference":
                return SyntaxTree.Float(0.0, call.args[0].value, reference = True)
            elif function.id == "logDataBlock":
                return SyntaxTree.LogDataBlock(self._parse_expression(call.args[0]), self._parse_constant(call.args[1]))
            elif function.id == "matrixVectorProduct":
                return self._parse_matrix_vector_product(call)
            elif function.id == "dot":
                return self._parse_diagonal_matrix_vector_product(call)
            elif function.id == "max":
                if len(call.args) > 1:
                    return SyntaxTree.DataBlockMax(self._parse_expression(call.args[0]), self._parse_expression(call.args[1]))
                else:
                    return SyntaxTree.DataBlockUnaryMax(self._parse_expression(call.args[0]))
            else:
                return SyntaxTree.DataBlockIndex(SyntaxTree.Name(function.id), self._parse_expression(call.args[0]))

    def _parse_diagonal_matrix_vector_product(self, call: ast.Call):
        matrix = self._parse_expression(call.args[0])
        vector = self._parse_expression(call.args[1])
        return SyntaxTree.DataBlockConstructionFromDotProduct(SyntaxTree.Name(call.args[2].id, vector.get_type()), matrix, vector)

    def _parse_matrix_vector_product(self, call: ast.Call):
        matrix = self._parse_expression(call.args[0])
        vector = self._parse_expression(call.args[1])
        return SyntaxTree.DataBlockConstructionFromMatrixVectorProduct(SyntaxTree.Name(call.args[2].id, vector.get_type()), matrix, vector)

    #def _parse_vector_constructor(self, constructor: ast.Call):
    #    return SyntaxTree.VectorFromDataBlock(self._parse_expression(constructor.args[0]))

    def _parse_diagonal_matrix_constructor(self, constructor: ast.Call):
        width = self._parse_expression(constructor.args[0])
        array = self._parse_expression(constructor.args[1])
        return SyntaxTree.DiagonalMatrix(width, array, [width])

    def _parse_matrix_constructor(self, constructor: ast.Call):
        rows = self._parse_expression(constructor.args[0])
        cols = self._parse_expression(constructor.args[1])
        array = self._parse_expression(constructor.args[2])
        return SyntaxTree.Matrix([rows, cols], array, [rows])
    
    def _parse_kronecker_product(self, product: ast.Call):
        lhs_matrix = self._parse_expression(product.args[0])
        rhs_matrix = self._parse_expression(product.args[1])

        if type(lhs_matrix.get_type()) is SyntaxTree.TDiagonalMatrix and type(rhs_matrix.get_type()) is SyntaxTree.TDiagonalMatrix:
            if type(lhs_matrix) is SyntaxTree.Name:
                lhs_width = SyntaxTree.Name.variables[lhs_matrix.id]._width
            else:
                lhs_width = lhs_matrix._width
            
            if type(rhs_matrix) is SyntaxTree.Name:
                rhs_width = SyntaxTree.Name.variables[rhs_matrix.id]._width
            else:
                rhs_width = rhs_matrix._width

            width = lhs_width * rhs_width
            return SyntaxTree.DiagonalMatrix(width, SyntaxTree.FunctionCall("kroneckerProduct", [lhs_matrix, rhs_matrix]), [lhs_width, rhs_width])
        else:
            if type(lhs_matrix) is SyntaxTree.Name and type(lhs_matrix.get_type()) is SyntaxTree.TMatrix:
                lhs_dimensions = SyntaxTree.Name.variables[lhs_matrix.id]._dimensions
            elif type(lhs_matrix) is SyntaxTree.Name and type(lhs_matrix.get_type()) is SyntaxTree.TDiagonalMatrix:
                lhs_dimensions = [SyntaxTree.Name.variables[lhs_matrix.id]._width, SyntaxTree.Name.variables[lhs_matrix.id]._width]
            elif type(lhs_matrix) is SyntaxTree.TMatrix:
                lhs_dimensions = lhs_matrix._dimensions
            else:
                lhs_dimensions = [lhs_matrix._width, lhs_matrix._width]
            
            if type(rhs_matrix) is SyntaxTree.Name and type(rhs_matrix.get_type()) is SyntaxTree.TMatrix:
                rhs_dimensions = SyntaxTree.Name.variables[rhs_matrix.id]._dimensions
            elif type(rhs_matrix) is SyntaxTree.Name and type(rhs_matrix.get_type()) is SyntaxTree.TDiagonalMatrix:
                rhs_dimensions = [SyntaxTree.Name.variables[rhs_matrix.id]._width, SyntaxTree.Name.variables[rhs_matrix.id]._width]
            elif type(rhs_matrix) is SyntaxTree.TMatrix:
                rhs_dimensions = rhs_matrix._dimensions
            else:
                rhs_dimensions = [rhs_matrix._width, rhs_matrix._width]

            dimensions = [lhs_dimensions[0] * rhs_dimensions[0], lhs_dimensions[1] * rhs_dimensions[1]]
            return SyntaxTree.Matrix(dimensions, SyntaxTree.FunctionCall("kroneckerProduct", [lhs_matrix, rhs_matrix]), [lhs_dimensions[0], lhs_dimensions[1]])

    def _parse_patch_constructor(self, constructor: ast.Call, isFace = False):
        iteration_range = []
        for interval in constructor.args[0].elts:
            iteration_range.append([self._parse_expression(interval.elts[0]), self._parse_expression(interval.elts[1])])
        
        if type(constructor.args[1]) is ast.Call:
            internal = SyntaxTree.String(constructor.args[1].args[0].value)
            underlying_type = SyntaxTree.String("double")
            if len(constructor.args) > 2:
                underlying_type = SyntaxTree.String(constructor.args[2].value)
            function_arguments = []
            for arg in constructor.args[1].args[1:]:
                function_arguments.append(self._parse_expression(arg))
            if SyntaxTree.FunctionDefinition._stateless:
                internal = SyntaxTree.FunctionCall("SolverType::" + constructor.args[1].args[0].value, function_arguments, True)
            else:
                internal = SyntaxTree.FunctionCall(constructor.args[1].args[0].value, function_arguments)
            if not isFace:
                return SyntaxTree.DataBlock(iteration_range, internal, True, underlying_type=underlying_type) 
            else:
                return SyntaxTree.FaceDataBlock(iteration_range, internal, True, underlying_type=underlying_type) 
        elif type(constructor.args[1]) is ast.Constant:
            if type(constructor.args[1].value) is str:
                internal = SyntaxTree.String(constructor.args[1].value)
                if not isFace:
                    if len(constructor.args) > 2:
                        return SyntaxTree.DataBlock(iteration_range, internal, False, underlying_type=SyntaxTree.String(constructor.args[2].value))
                    else:
                        return SyntaxTree.DataBlock(iteration_range, internal, False)
                else:
                    if len(constructor.args) > 2:
                        return SyntaxTree.FaceDataBlock(iteration_range, internal, False, underlying_type=SyntaxTree.String(constructor.args[2].value))
                    else:
                        return SyntaxTree.FaceDataBlock(iteration_range, internal, False)
            else:
                internal = self._parse_constant(constructor.args[1])
                if not isFace:
                    return SyntaxTree.DataBlock(iteration_range, internal, True)
                else:
                    return SyntaxTree.FaceDataBlock(iteration_range, internal, True)

    def _parse_unary_operation(self, unary_operation: ast.UnaryOp):
        operand = self._parse_expression(unary_operation.operand)

        if type(unary_operation.op) is ast.USub:
            if type(operand.get_type()) is SyntaxTree.TDataBlock:
                return SyntaxTree.DataBlockUnaryOperation("-", operand)
            else:
                return SyntaxTree.UnaryOperation("-", self._parse_expression(unary_operation.operand))
                         

    def _parse_binary_operation(self, binary_operation: ast.BinOp):
        lhs = self._parse_expression(binary_operation.left)
        rhs = self._parse_expression(binary_operation.right)
        if type(binary_operation.op) is ast.Add:
            operation = "+"
        elif type(binary_operation.op) is ast.Mult:
            operation = "*"
        elif type(binary_operation.op) is ast.Sub:
            operation = "-"
        elif type(binary_operation.op) is ast.Div:
            operation = "/"

        if type(lhs.get_type()) is SyntaxTree.TDataBlock or type(rhs.get_type()) is SyntaxTree.TDataBlock:
            return SyntaxTree.DataBlockBinaryOperation(operation, lhs, rhs)
        #elif type(lhs.get_type()) is SyntaxTree.TMatrix or type(rhs.get_type()) is SyntaxTree.TMatrix:
        #    return SyntaxTree.DiagonalMatrixBinaryOperation(operation, lhs, rhs)
        else:
            return SyntaxTree.BinaryOperation(operation, lhs, rhs)

    def _parse_subscript(self, subscript: ast.Subscript):
        value = self._parse_expression(subscript.value)
        if type(value.get_type()) is SyntaxTree.TDataBlock:
            offset = []
            if type(subscript.slice) is ast.Tuple:
                for axis_offset in subscript.slice.elts:
                    if type(axis_offset) is ast.Slice:
                        offset.append([self._parse_expression(axis_offset.lower), self._parse_expression(axis_offset.upper)])
                    else:
                        offset.append([self._parse_expression(axis_offset), self._parse_expression(axis_offset) + SyntaxTree.Integer(1)])

            elif type(axis_offset) is ast.Slice:
                offset.append([self._parse_expression(axis_offset.lower), self._parse_expression(axis_offset.upper)])
            else:
                offset.append([self._parse_expression(axis_offset), self._parse_expression(axis_offset) + SyntaxTree.Integer(1)])

            return SyntaxTree.Name.variables[value.id].offset(offset)

    def _parse_comparison(self, comparison: ast.Compare):
        lhs = self._parse_expression(comparison.left)
        rhs = self._parse_expression(comparison.comparators[0])
        if type(comparison.ops[0]) is ast.Gt:
            return SyntaxTree.DataBlockComparison(">", lhs, rhs)
        if type(comparison.ops[0]) is ast.Lt:
            return SyntaxTree.DataBlockComparison(">", lhs, rhs)
        if type(comparison.ops[0]) is ast.Eq:
            return SyntaxTree.DataBlockComparison("==", lhs, rhs)

    def _parse_expression(self, expression):
        if type(expression) is ast.Constant:
            return self._parse_constant(expression)
        if type(expression) is ast.UnaryOp:
            return self._parse_unary_operation(expression)
        if type(expression) is ast.Compare:
            return self._parse_comparison(expression)
        if type(expression) is ast.BinOp:
            return self._parse_binary_operation(expression)
        if type(expression) is ast.Name:
            if expression.id == "index":
                return SyntaxTree.Index()
            return SyntaxTree.Name(expression.id)
        if type(expression) is ast.Call:
            return self._parse_call(expression)
        if type(expression) is ast.Subscript:
            return self._parse_subscript(expression)
        if expression is None:
            return None

    def _parse_assignment_with_declaration(self, assignment_with_declaration: ast.Assign):
        value = self._parse_expression(assignment_with_declaration.value)
        id = assignment_with_declaration.targets[0].id
        if type(value.get_type()) is SyntaxTree.TDataBlock:
            if type(value) is SyntaxTree.DataBlockUnaryMax:
                value.set_output_variable(SyntaxTree.Name(id, value.get_type()))
                return value
            if value.requires_memory_allocation == False:
                return SyntaxTree.DataBlockConstructionFromExisting(SyntaxTree.Name(id, value.get_type()), value)
            else:
                if type(value) is SyntaxTree.DataBlockBinaryOperation or type(value) is SyntaxTree.DataBlockMax or type(value) is SyntaxTree.DataBlockComparison or type(value._internal) is not SyntaxTree.FunctionCall:
                    return SyntaxTree.DataBlockConstructionFromOperation(SyntaxTree.Name(id, value.get_type()), value)
                else:
                    return SyntaxTree.DataBlockConstructionFromFunction(SyntaxTree.Name(id, value.get_type()), value)
#        elif type(value) is SyntaxTree.VectorFromDataBlock:
#            SyntaxTree.Name.variables.update({id: value})
#            return None
        elif type(value) is SyntaxTree.DiagonalMatrix:
            SyntaxTree.Name.variables.update({id: value})
            if type(value._internal) is SyntaxTree.FunctionCall and value._internal.id == "kroneckerProduct":
                value._id = SyntaxTree.String(id)
                return SyntaxTree.DiagonalMatrixConstructionFromKroneckerProduct(id, value)
            if type(value._internal) is SyntaxTree.String:
                return None
            return SyntaxTree.DiagonalMatrixConstructionFromValue(id, value)
        elif type(value) is SyntaxTree.Matrix:
            SyntaxTree.Name.variables.update({id: value})
            if type(value._internal) is SyntaxTree.FunctionCall and value._internal.id == "kroneckerProduct":
                value._id = SyntaxTree.String(id)
                return SyntaxTree.MatrixConstructionFromKroneckerProduct(id, value)
            return None
        else:
            return SyntaxTree.Construction(SyntaxTree.Name(id, value.get_type()), value)
    
    def _parse_assignment_without_declaration(self, assignment_with_declaration: ast.Assign):
        value = self._parse_expression(assignment_with_declaration.value)
        target = self._parse_expression(assignment_with_declaration.targets[0])

        if type(value.get_type()) is SyntaxTree.TDataBlock or type(target.get_type()) is SyntaxTree.TDataBlock:
            if type(value) is SyntaxTree.DataBlockUnaryMax:
                value.set_output_variable(target)
                return value
            return SyntaxTree.DataBlockAssignment(target, value)
        elif type(target.get_type()) is SyntaxTree.TVector:
            return SyntaxTree.VectorAssignment(target, value)
        else:
            return SyntaxTree.Assignment(target, value)

    def _parse_assignment(self, assignment: ast.Assign):
        target = assignment.targets[0]
        if type(target) is ast.Subscript:
            return self._parse_assignment_without_declaration(assignment)
        elif target.id not in SyntaxTree.Name.variables:
            return self._parse_assignment_with_declaration(assignment)
        else:
            return self._parse_assignment_without_declaration(assignment)

    def _parse_if(self, statement: ast.If):
        pde_term_is_in_use = False
        assert(isinstance(statement.test, ast.Name))
        test = statement.test.id
        for pde_term in self.inuse_pde_terms:
            if pde_term.NAME == test:
                pde_term_is_in_use = True

        if pde_term_is_in_use:
            statements = []
            for s in statement.body:
                statements.append(self._parse_statement(s))
            return statements
        else:
            return None
#        if_node = SyntaxTree.If(self._parse_expression(statement.test))
#        SyntaxTree.MemoryAllocation.memoryAllocated.append([])
#
#        for s in statement.body:
#            if_node.add_statement(self._parse_statement(s))
#        for allocation in SyntaxTree.MemoryAllocation.memoryAllocated[-1]:
#                if_node.add_statement(SyntaxTree.MemoryDeallocation(allocation))
#
#        SyntaxTree.MemoryAllocation.memoryAllocated.pop()
#        return if_node

    def _parse_statement(self, statement: ast.Expr):  
        if type(statement) is ast.Assign:
            return self._parse_assignment(statement)
        if type(statement) is ast.Comment:
            return SyntaxTree.Comment(statement.value)
        if type(statement) is ast.If:
            return self._parse_if(statement)
        if type(statement) is ast.Expr:
            return self._parse_expression(statement.value)
    
    def _parse_function(self, function: ast.FunctionDef, template, functor_arguments, namespaces, stateless, use_accelerator, top_level = False, function_name = None):
        id = function.name if function_name is None else function_name
        return_type = SyntaxTree.TCustom(function.returns.value) if function.returns is not None else SyntaxTree.TCustom(function.returns)
        parsed_function = SyntaxTree.FunctionDefinition(id, return_type, template, namespaces, stateless, top_level)

        if use_accelerator:
            parsed_function.add_argument(SyntaxTree.Argument("targetDevice", SyntaxTree.TInteger()))
            SyntaxTree.Node.use_accelerator = True
        for argument in function.args.args:
            parsed_function.add_argument(SyntaxTree.Argument(argument.arg, SyntaxTree.TCustom(argument.annotation.value) if argument.annotation is not None else SyntaxTree.TCustom(argument.annotation)))
            
        if stateless:
            parsed_function.id = parsed_function.id + "Stateless"
            parsed_function._template.insert(0, SyntaxTree.Argument("SolverType", SyntaxTree.TCustom("class")))
            for functor in functor_arguments:
                if type(functor._type) is not SyntaxTree.TCustom:
                    parsed_function.add_argument(functor)
        else:
            for functor in functor_arguments:
                parsed_function.add_argument(functor)
        
        SyntaxTree.MemoryAllocation.memoryAllocated.append([])
        for statement in function.body:
            next_statement = self._parse_statement(statement)
            if isinstance(next_statement, list):
                for nested_statement in next_statement:
                    parsed_function.add_statement(nested_statement)
            elif next_statement is not None:
                parsed_function.add_statement(next_statement)
        for allocation in SyntaxTree.MemoryAllocation.memoryAllocated[-1]:
            parsed_function.add_statement(SyntaxTree.MemoryDeallocation(allocation))
        SyntaxTree.MemoryAllocation.memoryAllocated.pop()

        return parsed_function

    def parse(self, function, template_arguments, functor_arguments, inuse_pde_terms, namespace = [], stateless = False, use_accelerator = False, mlir_omp = False, use_memory_manager = False, function_name = None):
        SyntaxTree.Name.variables.clear()
        SyntaxTree.For._inuse_iteration_variables.clear()
        SyntaxTree.Node.use_accelerator = False
        SyntaxTree.Node.use_mlir_omp = mlir_omp
        SyntaxTree.Node.use_memory_manager = use_memory_manager
        #SyntaxTree.FunctionDefinition._syclDataToCopy.clear()
        self.inuse_pde_terms = inuse_pde_terms
        # A generator may hand over the source text of a function it has
        # assembled, e.g. exahype2.api.rkfd4 for compact second derivatives
        source_code = function if isinstance(function, str) else inspect.getsource(function)
        
        module = ast.parse(source_code)
        function_definition = module.body[0]
        assert isinstance(function_definition, ast.FunctionDef)
        for argument in template_arguments:
            if type(argument._type) is SyntaxTree.TInteger:
                SyntaxTree.Name.variables.update({argument.id: SyntaxTree.Integer(99)})
            if type(argument._type) is SyntaxTree.TBoolean:
                SyntaxTree.Name.variables.update({argument.id: SyntaxTree.Boolean(False)})
            if type(argument._type) is SyntaxTree.TFloat:
                SyntaxTree.Name.variables.update({argument.id: SyntaxTree.Float(99.0)})
        for argument in functor_arguments:
            if type(argument._type) is SyntaxTree.TInteger:
                SyntaxTree.Name.variables.update({argument.id: SyntaxTree.Integer(99)})
            if type(argument._type) is SyntaxTree.TBoolean:
                SyntaxTree.Name.variables.update({argument.id: SyntaxTree.Boolean(False)})
            if type(argument._type) is SyntaxTree.TFloat:
                SyntaxTree.Name.variables.update({argument.id: SyntaxTree.Float(99.0)})

        template_arguments_copy = copy.deepcopy(template_arguments)
        root = self._parse_function(function_definition, template_arguments_copy, functor_arguments, namespace, stateless, use_accelerator, top_level = True, function_name = function_name)
        #root.top_level(True)
        return root
