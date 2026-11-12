import ast_comments as ast, inspect
from collections.abc import Sequence
import exahype2.api.dslhype2.AbstractSyntaxTree as AST
from exahype2.api.PDETerm import PDETerm

class Parser:
    def _parse_string(self, string: str) -> AST.External:
        if "." in string:
            sections = string.split(".")
            if len(sections) != 2:
                raise Exception("External format is not supported")
            struct_name = AST.Name(sections[0])
            if struct_name not in self.struct_map:
                raise Exception(f"Struct does not exist in struct_map: {struct_name.id}")
            struct_type = self.struct_map[struct_name]
            field_name = AST.Name(sections[1])
            return AST.ExternalStructField(struct_name, struct_type, field_name)
        else:
            return AST.External(string)
    
    def _parse_constant(self, constant: ast.Constant) -> AST.Element | AST.External:
        if type(constant.value) == int:
            return AST.Integer(constant.value)
        if type(constant.value) == float:
            return AST.Float(constant.value)
        if type(constant.value) == str:
            return self._parse_string(constant.value)
        raise Exception(f"Constant type is not supported: {type(constant.value)}")

    def _parse_variable(self, name: ast.Name) -> AST.Variable:
        return AST.Variable(AST.Name(name.id))

    def _parse_data_block_constructor(self, constructor: ast.Call) -> AST.DataBlock:
        if len(constructor.args) < 2 or len(constructor.args) > 3:
            raise Exception(f"DataBlock constructor expects 2 or 3 arguments, where the third is an optional type specifier: {len(constructor.args)} provided")
        
        # Argument 1 is the iteration range
        iteration_range: list[list[AST.Element | AST.Variable]] = []
        assert(type(constructor.args[0]) == ast.List)
        for interval in constructor.args[0].elts:
            assert(type(interval) == ast.List)
            if (len(interval.elts) != 2):
                raise Exception("DataBlock constructor intervals must be a list of 2 integers")
            interval_start = self._parse_expression(interval.elts[0])
            interval_end = self._parse_expression(interval.elts[1])
            if isinstance(interval_start.get_type(), AST.TInteger) and isinstance(interval_end.get_type(), AST.TInteger):
                assert(isinstance(interval_start, AST.Element) or isinstance(interval_start, AST.Variable))
                assert(isinstance(interval_end, AST.Element) or isinstance(interval_end, AST.Variable))
                interval_start = interval_start.convert_integers_to_indices()
                interval_end = interval_end.convert_integers_to_indices()
                iteration_range.append([interval_start, interval_end])
            else:
                raise Exception("DataBlock iteration ranges must be specified with integers")

        # Argument 2 is the source of the DataBlock's data
        source = self._parse_expression(constructor.args[1])
        if not (isinstance(source, AST.Element) or isinstance(source, AST.ExternalStructField) or isinstance(source, AST.FunctionCall)):
            raise Exception(f"DataBlock source is not a supported: {type(source)}")
        flatten_array_representation = False
        #if isinstance(source, AST.ExternalStructField):
        #    flatten_array_representation = True
        return AST.DataBlock(iteration_range, source, flatten=flatten_array_representation)

    def _parse_evaluation(self, call: ast.Call) -> AST.FunctionCall:
        function_name = self._parse_expression(call.args[0])
        assert(isinstance(function_name, AST.External))
        function_name = AST.Name(function_name.reference)
        if function_name not in self.function_map:
            raise Exception(f"Function name is not recognised: {function_name.id}")
        arguments = [self._parse_expression(argument) for argument in call.args[1:]]
        return AST.FunctionCall(function_name, arguments, self.function_map[function_name], stateless = self.stateless)

    def parse_call(self, call: ast.Call) -> AST.Expression | AST.FunctionCall:
        assert(type(call.func) == ast.Name)
        match call.func.id:
            case "DataBlock":
                return self._parse_data_block_constructor(call)
            case "Integer":
                assert(isinstance(call.args[0], ast.Constant))
                assert(isinstance(call.args[0].value, str))
                return AST.Integer(self._parse_string(call.args[0].value))
            case "eval":
                return self._parse_evaluation(call)
            case "max":
                if len(call.args) == 1:
                    operand = self._parse_expression(call.args[0])
                    assert(isinstance(operand, AST.Variable) and isinstance(operand.get_type(), AST.TDataBlock))
                    return AST.DataBlockUnaryOperation(AST.Operator.MAX, operand)
                if len(call.args) == 2:
                    lhs = self._parse_expression(call.args[0])
                    rhs = self._parse_expression(call.args[1])
                    if isinstance(lhs.get_type(), AST.TElement) and isinstance(rhs.get_type(), AST.TElement):
                        assert(isinstance(lhs, AST.Element) or isinstance(lhs, AST.Variable))
                        assert(isinstance(rhs, AST.Element) or isinstance(rhs, AST.Variable))
                        return AST.ElementBinaryOperation(AST.Operator.MAX, lhs, rhs)
                    else:
                        return AST.DataBlockBinaryOperation(AST.Operator.MAX, lhs, rhs)
                raise Exception("Unknown format for max: 1 or 2 arguments expected")

            case _:
                if AST.Name(call.func.id) in AST.Variable.symbol_map:
                    assert(isinstance(AST.Variable.symbol_map[AST.Name(call.func.id)].get_type(), AST.TDataBlock))
                    index = self._parse_expression(call.args[0])
                    assert(isinstance(index, AST.Integer))
                    index = index.convert_integers_to_indices()
                    return AST.DataBlockIndex(AST.Variable(AST.Name(call.func.id)), index)

        raise Exception(f"Function name is not supported: {call.func.id}")

    def parse_unary_operation(self, operation: ast.UnaryOp) -> AST.ElementUnaryOperation | AST.DataBlockUnaryOperation | AST.Element:
        if isinstance(operation.op, ast.USub):
            operand = self._parse_expression(operation.operand)
            if isinstance(operand.get_type(), AST.TElement):
                assert(isinstance(operand, AST.Element) or isinstance(operand, AST.Variable))
                if isinstance(operand, AST.Integer) and isinstance(operand.value, int):
                    return AST.Integer(-operand.value)
                return AST.ElementUnaryOperation(AST.Operator.SUBTRACTION, operand)
            else:
                return AST.DataBlockUnaryOperation(AST.Operator.SUBTRACTION, operand)
        raise Exception("Only minus is supported among unary operations")

    def parse_binary_operation(self, operation: ast.BinOp) -> AST.ElementBinaryOperation | AST.DataBlockBinaryOperation | AST.Element:
        lhs = self._parse_expression(operation.left)
        rhs = self._parse_expression(operation.right)

        if isinstance(lhs.get_type(), AST.TElement) and isinstance(rhs.get_type(), AST.TElement):
            assert(isinstance(lhs, AST.Element) or isinstance(lhs, AST.Variable))
            assert(isinstance(rhs, AST.Element) or isinstance(rhs, AST.Variable))

            # If both arguments are integers, the operation should be evaluated immediately
            if isinstance(lhs, AST.Integer) and isinstance(lhs.value, int) and \
               isinstance(rhs, AST.Integer) and isinstance(rhs.value, int):
                if isinstance(operation.op, ast.Add):
                    return AST.Integer(lhs.value + rhs.value)
                if isinstance(operation.op, ast.Sub):
                    return AST.Integer(lhs.value - rhs.value)
                if isinstance(operation.op, ast.Mult):
                    return AST.Integer(lhs.value * rhs.value)
            
            if isinstance(operation.op, ast.Add):
                return AST.ElementBinaryOperation(AST.Operator.ADDITION, lhs, rhs)
            if isinstance(operation.op, ast.Sub):
                return AST.ElementBinaryOperation(AST.Operator.SUBTRACTION, lhs, rhs)
            if isinstance(operation.op, ast.Mult):
                return AST.ElementBinaryOperation(AST.Operator.MULTIPLICATION, lhs, rhs)
            if isinstance(operation.op, ast.Div):
                return AST.ElementBinaryOperation(AST.Operator.DIVISION, lhs, rhs)
        else:
            if isinstance(operation.op, ast.Add):
                return AST.DataBlockBinaryOperation(AST.Operator.ADDITION, lhs, rhs)
            if isinstance(operation.op, ast.Sub):
                return AST.DataBlockBinaryOperation(AST.Operator.SUBTRACTION, lhs, rhs)
            if isinstance(operation.op, ast.Mult):
                return AST.DataBlockBinaryOperation(AST.Operator.MULTIPLICATION, lhs, rhs)
            if isinstance(operation.op, ast.Div):
                return AST.DataBlockBinaryOperation(AST.Operator.DIVISION, lhs, rhs)

    def parse_slice(self, slice: ast.Slice, original_iteration_range: Sequence[AST.Element | AST.Variable]) -> list[AST.Element | AST.Variable]:
        if slice.lower is None:
            slice_start = original_iteration_range[0]
        else:
            slice_start = self._parse_expression(slice.lower)
            assert(isinstance(slice_start, AST.Element) or isinstance(slice_start, AST.Variable))
            slice_start = slice_start.convert_integers_to_indices()

        if slice.upper is None:
            slice_end = original_iteration_range[1]
        else:
            slice_end = self._parse_expression(slice.upper)
            assert(isinstance(slice_end, AST.Element) or isinstance(slice_end, AST.Variable))
            slice_end = slice_end.convert_integers_to_indices()

        return [slice_start, slice_end]

    def parse_subscript(self, subscript: ast.Subscript) -> AST.DataBlockSlice:
        base = self._parse_expression(subscript.value)
        assert(isinstance(base, AST.Variable))
        base_value = base.get_value()
        assert(isinstance(base_value, AST.DataBlock))

        iteration_range: list[list[AST.Element | AST.Variable]] = []
        if isinstance(subscript.slice, ast.Tuple):
            for i, interval in enumerate(subscript.slice.elts):
                assert(isinstance(interval, ast.Slice))
                slice = self.parse_slice(interval, base_value.iteration_range[i])
                iteration_range.append(slice)
        else:
            assert(isinstance(subscript.slice, ast.Slice))
            slice = self.parse_slice(subscript.slice, base_value.iteration_range[0])
            iteration_range.append(slice)

        for i in range(len(iteration_range), len(base_value.iteration_range)):
            iteration_range.append([base_value.iteration_range[i][0], base_value.iteration_range[i][1]])
        return AST.DataBlockSlice(base, iteration_range)
    
    def _parse_assignable_expression(self, expression: ast.expr) -> AST.AssignableExpression:
        if isinstance(expression, ast.Name):
            return self._parse_variable(expression)
        if isinstance(expression, ast.Subscript):
            return self.parse_subscript(expression)
        raise Exception(f"Expression type is not supported or is not assignable: {type(expression)}")

    def _parse_expression(self, expression: ast.expr) -> AST.Expression:
        if isinstance(expression, ast.Constant):
            return self._parse_constant(expression)
        if isinstance(expression, ast.Name):
            for external_variable in self.external_variables.keys():
                if external_variable.name.id == expression.id:
                    if isinstance(external_variable.type, AST.TInteger):
                        value = self.external_variables[external_variable]
                        assert(isinstance(value, int))
                        return AST.Integer(value)
                    if isinstance(external_variable.type, AST.TFloat):
                        value = self.external_variables[external_variable]
                        assert(isinstance(value, float))
                        return AST.Float(value)
            return self._parse_variable(expression)
        if isinstance(expression, ast.Call):
            return self.parse_call(expression)
        if isinstance(expression, ast.BinOp):
            return self.parse_binary_operation(expression)
        if isinstance(expression, ast.UnaryOp):
            return self.parse_unary_operation(expression)
        if isinstance(expression, ast.Subscript):
            return self.parse_subscript(expression)
        raise Exception(f"Expression type is not supported: {type(expression)}")

    def parse_data_block_construction(self, lhs: AST.Variable, rhs: AST.DataBlock) -> list[AST.Statement]:
        if isinstance(rhs.value, AST.Element):
            return [AST.DataBlockConstructionFromElement(lhs, rhs)]
        if isinstance(rhs.value, AST.External):
            return [AST.DataBlockConstructionFromExternal(lhs, rhs)]
        if isinstance(rhs.value, AST.FunctionCall):
            return [AST.DataBlockConstructionFromFunctionCall(lhs, rhs)]
        if isinstance(rhs, AST.DataBlockBinaryOperation):
            return [AST.DataBlockConstructionFromOperation(lhs, rhs)]
        raise Exception(f"Source type for DataBlock is not supported: {rhs.value}")

    def _parse_construction(self, assignment: ast.Assign) -> list[AST.Statement]:
        lhs = self._parse_assignable_expression(assignment.targets[0])
        assert(type(lhs) == AST.Variable)

        rhs = self._parse_expression(assignment.value)
        if isinstance(rhs, AST.DataBlock):
            return self.parse_data_block_construction(lhs, rhs)
        return [AST.Construction(lhs, rhs)]
    
    def _parse_assignment_without_construction(self, assignment: ast.Assign) -> list[AST.Statement]:
        lhs = self._parse_assignable_expression(assignment.targets[0])
        rhs = self._parse_expression(assignment.value)

        if (isinstance(lhs.get_type(), AST.TDataBlock) or isinstance(rhs.get_type(), AST.TDataBlock)):
            # For a DataBlock assignment it is ok for the rhs to not be a DataBlock, but the lhs must be
            if not isinstance(lhs.get_type(), AST.TDataBlock):
                raise Exception("LHS of data block assignment must be a data block")
            if isinstance(rhs, AST.DataBlockUnaryOperation) and rhs.operator == AST.Operator.MAX:
                return [AST.DataBlockUnaryMax(lhs, rhs.operand)]
            return [AST.DataBlockAssignment(lhs, rhs)]
        
        return [AST.Assignment(lhs, rhs)]

    def _parse_assignment(self, assignment: ast.Assign) -> list[AST.Statement]:
        lhs = self._parse_expression(assignment.targets[0])
        if not isinstance(lhs, AST.AssignableExpression):
            raise Exception(f"LHS of assignment must be an assignable value: {type(lhs)} was given")

        if lhs.get_name() in AST.Variable.symbol_map:
            return self._parse_assignment_without_construction(assignment)
        else:
            return self._parse_construction(assignment)

    def _parse_if(self, if_statement: ast.If) -> list[AST.Statement]:
        pde_term_is_in_use = False
        assert(isinstance(if_statement.test, ast.Name))
        test = if_statement.test.id

        for pde_term in self.enabled_pde_terms:
            if pde_term.NAME == test or type(pde_term).__name__ == test:
                pde_term_is_in_use = True

        statements: list[AST.Statement] = []
        if pde_term_is_in_use:
            for s in if_statement.body:
                statements.extend(self._parse_statement(s))
        return statements
    
    def _parse_statement(self, statement: ast.stmt) -> list[AST.Statement]:
        try:
            if isinstance(statement, ast.Assign):
                return self._parse_assignment(statement)
            if isinstance(statement, ast.If):
                return self._parse_if(statement)
            if isinstance(statement, ast.Comment):
                return [AST.Comment(statement.value[1:])]
            raise Exception(f"Statement type is not supported: {type(statement)}")
        
        except Exception as exception:
            if len(exception.args) > 0:
                raise Exception(f"Error on line {self.line_number}: {exception.args[0]}")
            raise Exception(f"Error on line {self.line_number}")
    
    def _parse_function(self, function: ast.FunctionDef, function_id: str|None, function_namespace: list[str], function_arguments: list[AST.Argument], stateless: bool) -> AST.Function:
        if function_id is None:
            function_name = AST.Name(function.name)
        else:
            function_name = AST.Name(function_id)

        parsed_function = AST.Function(function_name, function_namespace, stateless)
        parsed_function.function_map = self.function_map
        parsed_function.add_arguments(function_arguments)

        for statement in function.body:
            parsed_function.add_statements(self._parse_statement(statement))
            self.line_number = self.line_number + 1
        
        array_allocations = AST.ArrayAllocation.array_allocations.copy()
        for array_allocation in array_allocations:
            parsed_function.add_statements([AST.ArrayDeallocation(array_allocation)])

        return parsed_function
    
    def parse(self, function, function_id: str|None = None, function_namespace: list[str] = [], function_arguments: list[AST.Argument] = [], function_map: dict[AST.Name, AST.TFunction] = {}, enabled_pde_terms: list[PDETerm] = [], external_variables: dict[AST.Argument,int|float] = {}, stateless: bool = False) -> AST.Function:
        self.line_number = 1
        self.enabled_pde_terms = enabled_pde_terms
        self.external_variables = external_variables
        self.function_map = function_map
        self.stateless = stateless
        AST.Variable.symbol_map = {}
        AST.ArrayAllocation.array_allocations = []
        
        self.struct_map: dict[AST.Name, AST.TStruct] = {}
        for argument in function_arguments:
            if isinstance(argument.type, AST.TStruct):
                self.struct_map.update({argument.name: argument.type})

        source_code = inspect.getsource(function)
        module = ast.parse(source_code)
        assert (isinstance(module, ast.Module))
        function_definition = module.body[0]
        assert isinstance(function_definition, ast.FunctionDef)

        function = self._parse_function(function_definition, function_id, function_namespace, function_arguments, stateless)
        return function