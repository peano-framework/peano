# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import os
import re
import jinja2


from .Helper import write_file

from .Overwrite import Overwrite

from .Jinja2TemplatedHeaderFile import Jinja2TemplatedHeaderFile

from ..Logger import Logger


class Observer(object):
    """!

    C++ observer of one mesh traversal

    ## Task groups

    The observer invokes its action sets one by one: Per event, such as
    touchVertexFirstTime() or createCell(), it runs through the action sets
    and invokes the event on each of them. The action sets can be split
    into task groups, i.e. consecutive action sets which may run in parallel
    (@ref peano4.api.solversteps.SynchroniseActionSets.get_task_groups()).
    Within enterCell() and leaveCell(), the observer embeds the code which
    invokes the events of a group's action sets into one task per action
    set, spawns these tasks and waits for them
    (tarch::multicore::spawnAndWait()) before it continues with the next
    group. This holds for all events, i.e. the create, touch and destroy
    events of vertices, faces and cells. The tasks capture everything by
    reference, which is safe, as the observer waits for them. A group with
    one action set yields exactly the same code as an observer without task
    groups.

    """

    default_overwrite = True

    def __init__(
        self,
        classname,
        namespace,
        subdirectory,
        included_actions,
        vertices,
        faces,
        cells,
        task_groups=None,
    ):
        """!

        @param included_actions: [str]
          Full qualified names of the action sets in invocation order.

        @param task_groups: [[int]] or None
          Groups of indices into included_actions. The groups are invoked
          one after the other, the action sets within a group in parallel.
          Concatenated, the groups have to yield 0, 1, 2, ..., i.e. every
          action set exactly once and in invocation order. If None, every
          action set forms a group of its own.

        """
        self.classname = classname
        self.namespace = namespace
        self.subdirectory = subdirectory

        self.included_actions = included_actions
        if task_groups is None:
            task_groups = [[index] for index in range(len(included_actions))]
        if [index for group in task_groups for index in group] != list(range(len(included_actions))):
            raise Exception(
                f"Task groups {task_groups} of observer {classname} do not enumerate its {len(included_actions)} action sets in order"
            )
        self.task_groups = task_groups
        self.vertices = vertices
        self.faces = faces
        self.cells = cells

        self.d = {}
        self.d["OPEN_NAMESPACE"] = ""
        self.d["CLOSE_NAMESPACE"] = ""
        self.d["FULL_QUALIFIED_CLASSNAME"] = ""
        for i in namespace:
            self.d["OPEN_NAMESPACE"] += "namespace " + i + "{\n"
            self.d["CLOSE_NAMESPACE"] += "}\n"
            self.d["FULL_QUALIFIED_CLASSNAME"] += i + "::"
        self.d["CLASSNAME"] = classname
        self.d["FULL_QUALIFIED_CLASSNAME"] += classname
        self.d["INCLUDE_GUARD"] = (
            "_" + self.d["FULL_QUALIFIED_CLASSNAME"].replace("::", "_").upper() + "_H_"
        )
        self.d["INCLUDES"] = ""
        self.d["ATTRIBUTES"] = ""

        for action in self.included_actions:
            self.d["INCLUDES"] += '#include "'
            self.d["INCLUDES"] += action.replace("::", "/")
            self.d["INCLUDES"] += '.h"\n'
            self.d["ATTRIBUTES"] += "    "
            self.d["ATTRIBUTES"] += action
            self.d["ATTRIBUTES"] += "   _actionSet"
            self.d["ATTRIBUTES"] += str(self.included_actions.index(action))
            self.d["ATTRIBUTES"] += "; \n"

        subdirectories = set()

        self.d["MAPPING_SIGNATURE_FINE_GRID_CELL_ARGUMENTS"] = ""
        self.d["MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS"] = ""
        self.d["MAPPING_SIGNATURE_FINE_GRID_CELL_ARGUMENTS_PICK_ENTRY"] = ""
        self.d["MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS_PICK_ENTRY"] = ""
        for cell in cells:
            if cells.index(cell) != 0:
                self.d["MAPPING_SIGNATURE_FINE_GRID_CELL_ARGUMENTS"] += ","
                self.d["MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS"] += ","
                self.d["MAPPING_SIGNATURE_FINE_GRID_CELL_ARGUMENTS_PICK_ENTRY"] += ","
                self.d["MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS_PICK_ENTRY"] += ","
            self.d["MAPPING_SIGNATURE_FINE_GRID_CELL_ARGUMENTS"] += (
                cell.subnamespace +
                "repositories::DataRepository::_"
                + cell.get_logical_type_name()
                + "Stack.getForPop( " +
                cell.subnamespace +
                "repositories::DataRepository::DataKey(_spacetreeId,peano4::grid::PeanoCurve::CallStack))->top(0)"
            )
            self.d["MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS"] += (
                cell.subnamespace +
                "repositories::DataRepository::_"
                + cell.get_logical_type_name()
                + "Stack.getForPop( " +
                cell.subnamespace +
                "repositories::DataRepository::DataKey(_spacetreeId,peano4::grid::PeanoCurve::CallStack))->top(1)"
            )
            self.d["MAPPING_SIGNATURE_FINE_GRID_CELL_ARGUMENTS_PICK_ENTRY"] += (
                cell.subnamespace +
                "repositories::DataRepository::_"
                + cell.get_logical_type_name()
                + "Stack.getForPop( " +
                cell.subnamespace +
                "repositories::DataRepository::DataKey(_spacetreeId,peano4::grid::PeanoCurve::CallStack))->top(0)"
            )
            self.d["MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS_PICK_ENTRY"] += (
                cell.subnamespace +
                "repositories::DataRepository::_"
                + cell.get_logical_type_name()
                + "Stack.getForPop( " +
                cell.subnamespace +
                "repositories::DataRepository::DataKey(_spacetreeId,peano4::grid::PeanoCurve::CallStack))->top(1)"
            )
            if cell.subdirectory:
              subdirectories.add(cell.subdirectory)

        self.d["MAPPING_SIGNATURE_FINE_GRID_CELL_ARGUMENTS_CELL_EVENT"] = self.d[
            "MAPPING_SIGNATURE_FINE_GRID_CELL_ARGUMENTS"
        ]
        self.d["MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS_CELL_EVENT"] = self.d[
            "MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS"
        ]

        self.d["MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS"] = ""
        self.d["MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS"] = ""
        self.d["MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS_PICK_ENTRY"] = ""
        self.d["MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_PICK_ENTRY"] = ""
        for face in faces:
            if faces.index(face) != 0:
                self.d["MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS"] += ","
                self.d["MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS"] += ","
                self.d["MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS_PICK_ENTRY"] += ","
                self.d[
                    "MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_PICK_ENTRY"
                ] += ","
            self.d["MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS"] += (
                "peano4::datamanagement::FaceEnumerator<"
                + face.get_full_qualified_type()
                + ">( &" +
                face.subnamespace +
                "repositories::DataRepository::_"
                + face.get_logical_type_name()
                + "Stack.getForPop( " +
                face.subnamespace +
                "repositories::DataRepository::DataKey(_spacetreeId,peano4::grid::PeanoCurve::CallStack))->top(TwoTimesD-1) )"
            )
            self.d["MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS"] += (
                "peano4::datamanagement::FaceEnumerator<"
                + face.get_full_qualified_type()
                + ">( &" +
                face.subnamespace +
                "repositories::DataRepository::_"
                + face.get_logical_type_name()
                + "Stack.getForPop( " +
                face.subnamespace +
                "repositories::DataRepository::DataKey(_spacetreeId,peano4::grid::PeanoCurve::CallStack))->top(TwoTimesD*2-1) )"
            )
            self.d["MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS_PICK_ENTRY"] += (
                "peano4::datamanagement::FaceEnumerator<"
                + face.get_full_qualified_type()
                + ">( &" +
                face.subnamespace +
                "repositories::DataRepository::_"
                + face.get_logical_type_name()
                + "Stack.getForPop( " +
                face.subnamespace +
                "repositories::DataRepository::DataKey(_spacetreeId,peano4::grid::PeanoCurve::CallStack))->top(TwoTimesD-1) )(pick)"
            )
            self.d["MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_PICK_ENTRY"] += (
                "peano4::datamanagement::FaceEnumerator<"
                + face.get_full_qualified_type()
                + ">( &" +
                face.subnamespace +
                "repositories::DataRepository::_"
                + face.get_logical_type_name()
                + "Stack.getForPop( " +
                face.subnamespace +
                "repositories::DataRepository::DataKey(_spacetreeId,peano4::grid::PeanoCurve::CallStack))->top(TwoTimesD*2-1) )(pick)"
            )
            if face.subdirectory:
              subdirectories.add(face.subdirectory)

        self.d["MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS"] = ""
        self.d["MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS"] = ""
        self.d["MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS_PICK_ENTRY"] = ""
        self.d["MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_PICK_ENTRY"] = ""
        for vertex in vertices:
            if vertices.index(vertex) != 0:
                self.d["MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS"] += ","
                self.d["MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS"] += ","
                self.d[
                    "MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS_PICK_ENTRY"
                ] += ","
                self.d[
                    "MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_PICK_ENTRY"
                ] += ","
            self.d["MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS"] += (
                "peano4::datamanagement::VertexEnumerator<"
                + vertex.get_full_qualified_type()
                + ">( &" +
                vertex.subnamespace +
                "repositories::DataRepository::_"
                + vertex.get_logical_type_name()
                + "Stack.getForPop( " +
                vertex.subnamespace +
                "repositories::DataRepository::DataKey(_spacetreeId,peano4::grid::PeanoCurve::CallStack))->top(TwoPowerD-1) )"
            )
            self.d["MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS"] += (
                "peano4::datamanagement::VertexEnumerator<"
                + vertex.get_full_qualified_type()
                + ">( &" +
                vertex.subnamespace +
                "repositories::DataRepository::_"
                + vertex.get_logical_type_name()
                + "Stack.getForPop( " +
                vertex.subnamespace +
                "repositories::DataRepository::DataKey(_spacetreeId,peano4::grid::PeanoCurve::CallStack))->top(TwoPowerD*2-1) )"
            )
            self.d["MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS_PICK_ENTRY"] += (
                "peano4::datamanagement::VertexEnumerator<"
                + vertex.get_full_qualified_type()
                + ">( &" +
                vertex.subnamespace +
                "repositories::DataRepository::_"
                + vertex.get_logical_type_name()
                + "Stack.getForPop( " +
                vertex.subnamespace +
                "repositories::DataRepository::DataKey(_spacetreeId,peano4::grid::PeanoCurve::CallStack))->top(TwoPowerD-1) )(pick)"
            )
            self.d["MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_PICK_ENTRY"] += (
                "peano4::datamanagement::VertexEnumerator<"
                + vertex.get_full_qualified_type()
                + ">( &" +
                vertex.subnamespace +
                "repositories::DataRepository::_"
                + vertex.get_logical_type_name()
                + "Stack.getForPop( " +
                vertex.subnamespace +
                "repositories::DataRepository::DataKey(_spacetreeId,peano4::grid::PeanoCurve::CallStack))->top(TwoPowerD*2-1) )(pick)"
            )
            if vertex.subdirectory:
              subdirectories.add(vertex.subdirectory)

        self.d["DATAREPOSITORY_INCLUDES"] = ""
        if subdirectories:
          for subdir in subdirectories:
            self.d["INCLUDES"] = "#include \"../" + subdir + "repositories/DataRepository.h\"\n" + self.d["INCLUDES"]
            self.d["DATAREPOSITORY_INCLUDES"] += "#include \"../../" + subdir + "repositories/DataRepository.h\"\n"
        else:
          self.d["DATAREPOSITORY_INCLUDES"] = "#include \"../repositories/DataRepository.h\"\n"

    def __generate_header(self, overwrite, directory):
        headerfile_template = (
            os.path.realpath(__file__)
            .replace(".pyc", ".template.h")
            .replace(".py", ".template.h")
        )
        md = self.mkSubDict(["ATTRIBUTES", "INCLUDES"])
        header = Jinja2TemplatedHeaderFile(
            headerfile_template,
            self.classname,
            self.namespace,
            self.subdirectory,
            md,
            self.default_overwrite,
            announce=False,
        )
        header.generate(overwrite, directory)
        del header

    TemplateConstructor = """

{{FULL_QUALIFIED_CLASSNAME}}::{{CLASSNAME}}(int spacetreeId):
  _spacetreeId( spacetreeId ) {{MAPPING_INITIALISATION_LIST}}
{}


  """

    def __generate_constructor(self, output_file):
        #
        # Constructor
        #
        self.d["MAPPING_INITIALISATION_LIST"] = ""
        for actions in self.included_actions:
            self.d["MAPPING_INITIALISATION_LIST"] += ", _actionSet"
            self.d["MAPPING_INITIALISATION_LIST"] += str(
                self.included_actions.index(actions)
            )
            self.d["MAPPING_INITIALISATION_LIST"] += "(spacetreeId)"
        output_file.write(jinja2.Template(self.TemplateConstructor).render(**self.d))

    TemplateClone = """

peano4::grid::TraversalObserver* {{FULL_QUALIFIED_CLASSNAME}}::clone(int spacetreeId) {
  return new {{CLASSNAME}}(spacetreeId);
}

  """

    TemplateBeginTraversal = """

void {{FULL_QUALIFIED_CLASSNAME}}::beginTraversal( const tarch::la::Vector<Dimensions,double>&  x, const tarch::la::Vector<Dimensions,double>&  h ) {
  logTraceInWith2Arguments( "beginTraversal(...)", x, h );
  //
  // Invoke beginTraversal() on the actions
  //
{{MAPPING_BEGIN_TRAVERSAL_CALLS}}

  //
  // Fill call stacks with dummies which represent level 0 such that we can
  // call standard action routines on level 1 with parents. Without these
  // statements, a top(1) call would raise an assertion.
  //
{{INITIAL_PUSH_TO_OUTPUT_STREAMS}}
  logTraceOutWith2Arguments( "beginTraversal(...)", x, h );
}

  """

    def __generate_beginTraversal(self, output_file):
        self.d["MAPPING_BEGIN_TRAVERSAL_CALLS"] = ""
        for action in self.included_actions:
            self.d["MAPPING_BEGIN_TRAVERSAL_CALLS"] += "  _actionSet"
            self.d["MAPPING_BEGIN_TRAVERSAL_CALLS"] += str(
                self.included_actions.index(action)
            )
            self.d["MAPPING_BEGIN_TRAVERSAL_CALLS"] += ".beginTraversal();\n"

        self.d["INITIAL_PUSH_TO_OUTPUT_STREAMS"] = ""
        for cell in self.cells:
            self.d["INITIAL_PUSH_TO_OUTPUT_STREAMS"] += (
                "  " +
                cell.subnamespace +
                "repositories::DataRepository::_"
                + cell.get_logical_type_name()
                + "Stack.getForPush( " +
                cell.subnamespace +
                "repositories::DataRepository::DataKey(_spacetreeId,peano4::grid::PeanoCurve::CallStack))->push( "
                + cell.get_full_qualified_type()
                + "() );\n"
            )
            pass

        for face in self.faces:
            self.d[
                "INITIAL_PUSH_TO_OUTPUT_STREAMS"
            ] += "  for (int i=0; i<TwoTimesD; i++) {\n"
            self.d["INITIAL_PUSH_TO_OUTPUT_STREAMS"] += (
                "  " +
                face.subnamespace +
                "repositories::DataRepository::_"
                + face.get_logical_type_name()
                + "Stack.getForPush( " +
                face.subnamespace +
                "repositories::DataRepository::DataKey(_spacetreeId,peano4::grid::PeanoCurve::CallStack))->push( "
                + face.get_full_qualified_type()
                + "() );\n"
            )
            self.d["INITIAL_PUSH_TO_OUTPUT_STREAMS"] += "  }\n"
            pass

        for vertex in self.vertices:
            self.d[
                "INITIAL_PUSH_TO_OUTPUT_STREAMS"
            ] += "  for (int i=0; i<TwoPowerD; i++) {\n"
            self.d["INITIAL_PUSH_TO_OUTPUT_STREAMS"] += (
                "    " +
                vertex.subnamespace +
                "repositories::DataRepository::_"
                + vertex.get_logical_type_name()
                + "Stack.getForPush( " +
                vertex.subnamespace +
                "repositories::DataRepository::DataKey(_spacetreeId,peano4::grid::PeanoCurve::CallStack))->push( "
                + vertex.get_full_qualified_type()
                + "() );\n"
            )
            self.d["INITIAL_PUSH_TO_OUTPUT_STREAMS"] += "  }\n"
            pass

        output_file.write(jinja2.Template(self.TemplateBeginTraversal).render(**self.d))

    TemplateEndTraversal = """

void {{FULL_QUALIFIED_CLASSNAME}}::endTraversal( const tarch::la::Vector<Dimensions,double>&  x, const tarch::la::Vector<Dimensions,double>&  h ) {
  logTraceInWith2Arguments( "endTraversal(...)", x, h );
  {{MAPPING_END_TRAVERSAL_CALLS}}
  {{FINAL_POP_FROM_INPUT_STREAMS}}
  logTraceOutWith2Arguments( "endTraversal(...)", x, h );
}

  """

    def __generate_endTraversal(self, output_file):
        self.d["MAPPING_END_TRAVERSAL_CALLS"] = ""
        for action in self.included_actions:
            self.d["MAPPING_END_TRAVERSAL_CALLS"] += "  _actionSet"
            self.d["MAPPING_END_TRAVERSAL_CALLS"] += str(
                self.included_actions.index(action)
            )
            self.d["MAPPING_END_TRAVERSAL_CALLS"] += ".endTraversal();\n"

        self.d["FINAL_POP_FROM_INPUT_STREAMS"] = ""
        for cell in self.cells:
            self.d["FINAL_POP_FROM_INPUT_STREAMS"] += (
                "  " +
                cell.subnamespace +
                "repositories::DataRepository::_"
                + cell.get_logical_type_name()
                + "Stack.getForPop( " +
                cell.subnamespace +
                "repositories::DataRepository::DataKey(_spacetreeId,peano4::grid::PeanoCurve::CallStack))->pop();\n"
            )
            pass

        for face in self.faces:
            self.d[
                "FINAL_POP_FROM_INPUT_STREAMS"
            ] += "  for (int i=0; i<TwoTimesD; i++) {\n"
            self.d["FINAL_POP_FROM_INPUT_STREAMS"] += (
                "    " +
                face.subnamespace +
                "repositories::DataRepository::_"
                + face.get_logical_type_name()
                + "Stack.getForPop( " +
                face.subnamespace +
                "repositories::DataRepository::DataKey(_spacetreeId,peano4::grid::PeanoCurve::CallStack))->pop();\n"
            )
            self.d["FINAL_POP_FROM_INPUT_STREAMS"] += "  }\n"
            pass

        for vertex in self.vertices:
            self.d[
                "FINAL_POP_FROM_INPUT_STREAMS"
            ] += "  for (int i=0; i<TwoPowerD; i++) {\n"
            self.d["FINAL_POP_FROM_INPUT_STREAMS"] += (
                "    " +
                vertex.subnamespace +
                "repositories::DataRepository::_"
                + vertex.get_logical_type_name()
                + "Stack.getForPop( " +
                vertex.subnamespace +
                "repositories::DataRepository::DataKey(_spacetreeId,peano4::grid::PeanoCurve::CallStack))->pop();\n"
            )
            self.d["FINAL_POP_FROM_INPUT_STREAMS"] += "  }\n"
            pass

        output_file.write(jinja2.Template(self.TemplateEndTraversal).render(**self.d))

    TemplatePrepareTraversal = """

void {{FULL_QUALIFIED_CLASSNAME}}::prepareTraversal() {
  logTraceIn( "prepareTraversal(...)" );
  {{MAPPING_PREPARE_TRAVERSAL_CALLS}}
  logTraceOut( "prepareTraversal(...)" );
}

  """

    def __generate_prepareTraversal(self, output_file):
        self.d["MAPPING_PREPARE_TRAVERSAL_CALLS"] = ""
        for action in self.included_actions:
            self.d["MAPPING_PREPARE_TRAVERSAL_CALLS"] += "  "
            self.d["MAPPING_PREPARE_TRAVERSAL_CALLS"] += action
            self.d["MAPPING_PREPARE_TRAVERSAL_CALLS"] += "::prepareTraversal();\n"
        output_file.write(
            jinja2.Template(self.TemplatePrepareTraversal).render(**self.d)
        )

    TemplateUnprepareTraversal = """

void {{FULL_QUALIFIED_CLASSNAME}}::unprepareTraversal() {
  logTraceIn( "unprepareTraversal(...)" );
  {{MAPPING_UNPREPARE_TRAVERSAL_CALLS}}
  logTraceOut( "unprepareTraversal(...)" );
}

  """

    def __generate_unprepareTraversal(self, output_file):
        self.d["MAPPING_UNPREPARE_TRAVERSAL_CALLS"] = ""
        for action in self.included_actions:
            self.d["MAPPING_UNPREPARE_TRAVERSAL_CALLS"] += "  "
            self.d["MAPPING_UNPREPARE_TRAVERSAL_CALLS"] += action
            self.d["MAPPING_UNPREPARE_TRAVERSAL_CALLS"] += "::unprepareTraversal();\n"
        output_file.write(
            jinja2.Template(self.TemplateUnprepareTraversal).render(**self.d)
        )

    def __format_template_per_action(
        self, output_file, template, reverse_order=False, manual_dict=None
    ):
        """

        Takes the specified template file, iterates over actions and pastes
        the template into the output file once per action. Per action, the dictionary's
        entries are updated. Otherwise, the dictionary remains unchanged.

        @todo It is not clear what manual_dict really means, and I think it is never none. Documentation might have to be updated, too.

        output_file:
          Handle on output file

        """
        local_actions = [x for x in self.included_actions]
        if reverse_order:
            local_actions.reverse()

        for action in local_actions:
            if manual_dict is None:
                self.d["ACTIVE_ACTION_SET"] = "_actionSet" + str(
                    self.included_actions.index(action)
                )
                self.d["ACTIVE_ACTION_SET_FULL_QUALIFIED_NAME"] = action
                if output_file is not None:
                    output_file.write(jinja2.Template(template).render(**self.d))
                else:
                    return jinja2.Template(template).render(**self.d)
            else:
                manual_dict["ACTIVE_ACTION_SET"] = "_actionSet" + str(
                    self.included_actions.index(action)
                )
                manual_dict["ACTIVE_ACTION_SET_FULL_QUALIFIED_NAME"] = action
                if output_file is not None:
                    output_file.write(jinja2.Template(template).render(**manual_dict))
                else:
                    return jinja2.Template(template).render(**manual_dict)

    def __render_template_for_action(self, template, action_index, manual_dict):
        """!

        Render template for the action set with the given index, i.e. the
        C++ code with which the observer invokes this action set's events

        """
        dictionary = self.d if manual_dict is None else manual_dict
        dictionary["ACTIVE_ACTION_SET"] = "_actionSet" + str(action_index)
        dictionary["ACTIVE_ACTION_SET_FULL_QUALIFIED_NAME"] = self.included_actions[action_index]
        return jinja2.Template(template).render(**dictionary)

    def __format_template_per_task_group(
        self, output_file, template, reverse_order=False, manual_dict=None
    ):
        """!

        Counterpart of __format_template_per_action() which takes the task
        groups into account

        The groups are handled one after the other. A group with one action
        set yields exactly the code of __format_template_per_action(). For a
        group with several action sets, the code of each action set is
        embedded into a lambda which captures everything by reference. Each
        lambda is spawned as a task, and the code waits for all of them
        before it continues. See the class documentation.

        @param reverse_order: bool
          Invoke the groups, and the action sets within each group, in
          reversed order, as we do for all events of leaveCell().

        """
        groups = [list(group) for group in self.task_groups]
        if reverse_order:
            groups = [list(reversed(group)) for group in reversed(groups)]

        for group in groups:
            if len(group) == 1:
                output_file.write(self.__render_template_for_action(template, group[0], manual_dict))
                continue

            code = "\n  {\n"
            code += "    // Action sets " + ", ".join(str(index) for index in group) + " run in parallel\n"
            code += "    std::vector< tarch::multicore::Task* > parallelActionSets;\n"
            for index in group:
                code += f"    std::function<void()> invokeActionSet{index} = [&]() -> void {{\n"
                code += self.__render_template_for_action(template, index, manual_dict)
                code += "\n    };\n"
                code += "    parallelActionSets.push_back( new tarch::multicore::TaskWithoutCopyOfFunctor(\n"
                code += "      tarch::multicore::Task::DontFuse,\n"
                code += "      tarch::multicore::Task::DefaultPriority,\n"
                code += f"      invokeActionSet{index}\n"
                code += "    ));\n"
            code += "    tarch::multicore::spawnAndWait( parallelActionSets );\n"
            code += "  }\n"
            output_file.write(code)

    TemplateGetGridControlEvents_Prologue = """

std::vector< peano4::grid::GridControlEvent > {{FULL_QUALIFIED_CLASSNAME}}::createGridControlEventsForNextTraversal() const {
  std::vector< peano4::grid::GridControlEvent > result;
"""

    TemplateGetGridControlEvents_MappingCall = """
  {
    const std::vector< peano4::grid::GridControlEvent > actionResult = {{ACTIVE_ACTION_SET}}.createGridControlEventsForNextTraversal();
    result.insert(result.begin(),actionResult.begin(),actionResult.end());
  }
"""

    TemplateGetGridControlEvents_Epilogue = """
  return result;
}


"""

    def __generate_createGridControlEventsForNextTraversal(self, output_file):
        output_file.write(
            jinja2.Template(self.TemplateGetGridControlEvents_Prologue).render(
                **{"FULL_QUALIFIED_CLASSNAME": self.d["FULL_QUALIFIED_CLASSNAME"]}
            )
        )
        self.__format_template_per_action(
            output_file, self.TemplateGetGridControlEvents_MappingCall, manual_dict={}
        )
        output_file.write(
            jinja2.Template(self.TemplateGetGridControlEvents_Epilogue).render(**{})
        )

    def __generate_clone(self, output_file):
        output_file.write(jinja2.Template(self.TemplateClone).render(**self.d))

    TemplateEnterCell_Prologue = """
void {{FULL_QUALIFIED_CLASSNAME}}::enterCell( const peano4::grid::GridTraversalEvent&  event ) {
  logTraceInWith2Arguments( "enterCell(...)", _spacetreeId, event.toString() );
  
  [[maybe_unused]] const peano4::datamanagement::CellMarker    cellMarker( event );
  [[maybe_unused]] const peano4::datamanagement::FaceMarker    faceMarker( event );
  [[maybe_unused]] const peano4::datamanagement::VertexMarker  vertexMarker(event);
"""

    TemplateLoadCell_VertexLoad = r"""
  // Load vertex {{logical_type_name}}
  std::function<void ()> loadVertex{{logical_type_name}} =
    [&]()->void {
    ::peano4::datamanagement::loadVertexData<{{full_qualified_type}}>(
      {{subnamespace}}repositories::DataRepository::_{{logical_type_name}}Stack,
      _spacetreeId,
      event,
      vertexMarker,
      "{{name}}",
      [&](const peano4::datamanagement::VertexMarker& marker) -> ::peano4::grid::LoadStoreComputeFlag {
        return {{full_qualified_type}}::loadStoreComputeFlag(marker{% for arg in ADDITIONAL_LOAD_STORE_ARGUMENTS %}, {{arg[0]}}{% endfor %});
      }
    );
  };

  tasks.push_back( new tarch::multicore::TaskWithoutCopyOfFunctor(
    tarch::multicore::Task::DontFuse,
    tarch::multicore::Task::DefaultPriority,
    loadVertex{{logical_type_name}}
  ));
"""

    TemplateEnterCell_VertexMappingCall = r"""
  // Handle vertex {{logical_type_name}}
  {
    peano4::datamanagement::VertexMarker  marker = vertexMarker;

    for (int i=0; i<TwoPowerD; i++) {
      int inVertexStack  = event.getVertexDataFrom(i);
      int pick          = event.getVertexDataTo(i);   // the vertex position

      marker.select(pick);

      logDebug( "enterCell(...)", inVertexStack << "->" << pick << " (is-local=" << marker.isLocal() << ")" );

      if (
        inVertexStack==peano4::grid::TraversalObserver::CreateOrDestroyPersistentGridEntity
        and
        marker.isLocal()
      ) {
        // Take care about the coarse grid accesses: Faces and cells are not yet loaded.
        // Therefore we don't use the usual shift of 2*2d or 2*2**d (don't use LaTeX here
        // as this is within a Python document and the Python interpreter doesn't fancy 
        // escape chars here, even though these escape chars are actually within a TeX
        // formula) but only half of it.
        {{ACTIVE_ACTION_SET}}.createPersistentVertex(
           marker
          ,{{MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS_PICK_ENTRY}}
          {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS}}
        );
        {{ACTIVE_ACTION_SET}}.touchVertexFirstTime(
           marker
          ,{{MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS_PICK_ENTRY}}
          {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS}}
        );
      }
      else if (
        inVertexStack==peano4::grid::TraversalObserver::CreateOrDestroyHangingGridEntity
        and
        marker.isLocal()
      ) {
        {{ACTIVE_ACTION_SET}}.createHangingVertex(
           marker
          ,{{MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS_PICK_ENTRY}}
          {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS}}
        );
      }
      else if (
        peano4::grid::PeanoCurve::isInOutStack(inVertexStack)
        and
        marker.isLocal()
      ) {
        {{ACTIVE_ACTION_SET}}.touchVertexFirstTime(
           marker
          ,{{MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS_PICK_ENTRY}}
          {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS}}
        );
      }
    }
  }
"""

    TemplateLoadCell_FaceLoad = r"""
  // Load face {{logical_type_name}}
  std::function<void ()> loadFace{{logical_type_name}} =
    [&]()->void {
    ::peano4::datamanagement::loadFaceData<{{full_qualified_type}}>(
      {{subnamespace}}repositories::DataRepository::_{{logical_type_name}}Stack,
      _spacetreeId,
      event,
      faceMarker,
      "{{name}}",
      [&](const peano4::datamanagement::FaceMarker& marker) -> ::peano4::grid::LoadStoreComputeFlag {
        return {{full_qualified_type}}::loadStoreComputeFlag(marker{% for arg in ADDITIONAL_LOAD_STORE_ARGUMENTS %}, {{arg[0]}}{% endfor %});
      }
    );
  };

  tasks.push_back( new tarch::multicore::TaskWithoutCopyOfFunctor(
    tarch::multicore::Task::DontFuse,
    tarch::multicore::Task::DefaultPriority,
    loadFace{{logical_type_name}}
  ));
"""

    TemplateEnterCell_FaceMappingCall = r"""
  // Handle face {{logical_type_name}}
  {
    peano4::datamanagement::FaceMarker marker = faceMarker;
    
    for (int i=0; i<TwoTimesD; i++) {
      int inFaceStack = event.getFaceDataFrom(i);
      int pick        = event.getFaceDataTo(i);

      marker.select(pick);

      assertion4( 
        marker.isLocal() or not event.getIsCellLocal(), 
        marker.toString(), 
        event.toString(), 
        i, 
        _spacetreeId 
      );

      if (
        inFaceStack==peano4::grid::TraversalObserver::CreateOrDestroyPersistentGridEntity
        and
        marker.isLocal()
      ) {
        {{ACTIVE_ACTION_SET}}.createPersistentFace(
           marker
          {{_COMMA_MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS}}
          ,{{MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS_PICK_ENTRY}}
          {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS}}
        );
        {{ACTIVE_ACTION_SET}}.touchFaceFirstTime(
           marker
          {{_COMMA_MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS}}
          ,{{MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS_PICK_ENTRY}}
          {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS}}
        );
      }
      else if (
        inFaceStack==peano4::grid::TraversalObserver::CreateOrDestroyHangingGridEntity
        and
        marker.isLocal()
      ) {
        {{ACTIVE_ACTION_SET}}.createHangingFace(
           marker
          {{_COMMA_MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS}}
          ,{{MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS_PICK_ENTRY}}
          {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS}}
        );
      }
      else if (
        peano4::grid::PeanoCurve::isInOutStack(inFaceStack)
        and
        marker.isLocal()
      ) {
        {{ACTIVE_ACTION_SET}}.touchFaceFirstTime(
           marker
          {{_COMMA_MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS}}
          ,{{MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS_PICK_ENTRY}}
          {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS}}
        );
      }
    }
  }
"""

    TemplateLoadCell_CellLoad = r"""
  // Load cell {{logical_type_name}}
  std::function<void ()> loadCell{{logical_type_name}} =
    [&]()->void {
    ::peano4::datamanagement::loadCellData<{{full_qualified_type}}>(
      {{subnamespace}}repositories::DataRepository::_{{logical_type_name}}Stack,
      _spacetreeId,
      event,
      cellMarker,
      "{{name}}",
      [&](const peano4::datamanagement::CellMarker& marker) -> ::peano4::grid::LoadStoreComputeFlag {
        return {{full_qualified_type}}::loadStoreComputeFlag(marker{% for arg in ADDITIONAL_LOAD_STORE_ARGUMENTS %}, {{arg[0]}}{% endfor %});
      }
    );
  };

  tasks.push_back( new tarch::multicore::TaskWithoutCopyOfFunctor(
    tarch::multicore::Task::DontFuse,
    tarch::multicore::Task::DefaultPriority,
    loadCell{{logical_type_name}}
  ));
"""

    TemplateEnterCell_CellMappingCall = r"""
  // Invoke creational events on cell {{logical_type_name}}
  {
    peano4::datamanagement::CellMarker marker = cellMarker;
    
    if (
      event.getCellData()==peano4::grid::TraversalObserver::CreateOrDestroyPersistentGridEntity
      and
      marker.isLocal()
    ) {
      {{ACTIVE_ACTION_SET}}.createCell(
        marker,
        {{MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS_COMMA_}}
        {{MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS_COMMA_}}
        {{MAPPING_SIGNATURE_FINE_GRID_CELL_ARGUMENTS_PICK_ENTRY}}
        {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS}}
        {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS}}
        {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS}}
      );
    }
  }
"""

    TemplateEnterCell_MappingCall = r"""
  {
    peano4::datamanagement::CellMarker marker( event );
    if (
      marker.isLocal()
    ) {
      {{ACTIVE_ACTION_SET}}.touchCellFirstTime(
         marker
        {{_COMMA_MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_FINE_GRID_CELL_ARGUMENTS_CELL_EVENT}}
        {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS_CELL_EVENT}}
      );
    }
  }
"""

    TemplateEnterCell_Epilogue = """
  logTraceOut( "enterCell(...)" );
}


"""

    def generateDictEntry(self, key):
        """
        Some logic to produce new dictionary entries from existing ones.
        This deals mainly with having to avoid C++ code with consequtive commata
        as some dict entries can be an empty string.
        """

        fields = key.split(",")
        nonempty = [self.d[f] for f in fields if not f == "" and not self.d[f] == ""]
        if len(nonempty) > 0:
            s = ",".join(nonempty)
            if key.startswith(",") and not s.startswith(","):
                s = "," + s
            if key.endswith(",") and not s.endswith(","):
                s += ","
        else:
            s = ""

        return s

    def mkSubDict(self, keys):
        """

        Create the particular subdictionary that's used for an expression.

        """
        temp = {}
        for k in keys:
            if not k in self.d:
                temp[k] = self.generateDictEntry(k)
            else:
                temp[k] = self.d[k]
        result = {}
        for k in temp:
            result[k.replace(",", "_COMMA_").strip()] = temp[k]
        return result

    def __generate_loadCell(self, output_file):
        output_file.write(
            jinja2.Template(self.TemplateLoadCell_Prologue).render(**self.d)
        )

        for vertex in self.vertices:
            temp = {
                "name": vertex.name,
                "enumeration_type": vertex.get_enumeration_type(),
                "logical_type_name": vertex.get_logical_type_name(),
                "full_qualified_type": vertex.get_full_qualified_type(),
                "ADDITIONAL_LOAD_STORE_ARGUMENTS": vertex.additional_load_and_store_arguments,
                "subnamespace": vertex.subnamespace,
            }
            self.d["name"] = vertex.name
            output_file.write(
                jinja2.Template(self.TemplateLoadCell_VertexLoad).render(**temp)
            )

        for face in self.faces:
            temp = {
                "name": face.name,
                "enumeration_type": face.get_enumeration_type(),
                "logical_type_name": face.get_logical_type_name(),
                "full_qualified_type": face.get_full_qualified_type(),
                "ADDITIONAL_LOAD_STORE_ARGUMENTS": face.additional_load_and_store_arguments,
                "subnamespace": face.subnamespace,
            }
            self.d["name"] = face.name
            output_file.write(
                jinja2.Template(self.TemplateLoadCell_FaceLoad).render(**temp)
            )

        for cell in self.cells:
            temp = {
                "name": cell.name,
                "logical_type_name": cell.get_logical_type_name(),
                "full_qualified_type": cell.get_full_qualified_type(),
                "ADDITIONAL_LOAD_STORE_ARGUMENTS": cell.additional_load_and_store_arguments,
                "subnamespace": cell.subnamespace,
            }
            self.d["name"] = cell.name
            output_file.write(
                jinja2.Template(self.TemplateLoadCell_CellLoad).render(**temp)
            )

        output_file.write(jinja2.Template(self.TemplateLoadCell_Epilogue).render({}))

    def __generate_enterCell(self, output_file):
        """
        Generates enter cell
        """
        output_file.write(
            jinja2.Template(self.TemplateEnterCell_Prologue).render(**self.d)
        )

        if len(self.vertices) > 0:
            md = self.mkSubDict(
                [
                    "name",
                    "MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS_PICK_ENTRY",
                    ",MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS,MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS,MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS",
                ]
            )
            self.__format_template_per_task_group(
                output_file,
                self.TemplateEnterCell_VertexMappingCall,
                False,
                manual_dict=md,
            )

        if len(self.faces) > 0:
            md = self.mkSubDict(
                [
                    "name",
                    "MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS",
                    "MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS_PICK_ENTRY",
                    "MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS",
                    "MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS",
                    "MAPPING_SIGNATURE_FINE_GRID_CELL_ARGUMENTS",
                    ",MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS,MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS,MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS",
                    ",MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS",
                ]
            )
            self.__format_template_per_task_group(
                output_file,
                self.TemplateEnterCell_FaceMappingCall,
                False,
                manual_dict=md,
            )

        if len(self.cells) > 0:
            md = self.mkSubDict(
                [
                    "name",
                    "MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS,",
                    "MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS,",
                    "MAPPING_SIGNATURE_FINE_GRID_CELL_ARGUMENTS_PICK_ENTRY",
                    ",MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS",
                    ",MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS",
                    ",MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS",
                ]
            )
            self.__format_template_per_task_group(
                output_file,
                self.TemplateEnterCell_CellMappingCall,
                False,
                manual_dict=md,
            )

        md = self.mkSubDict(
            [
                ",MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS,MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS,MAPPING_SIGNATURE_FINE_GRID_CELL_ARGUMENTS_CELL_EVENT",
                ",MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS,MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS,MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS_CELL_EVENT",
            ]
        )

        self.__format_template_per_task_group(
            output_file, self.TemplateEnterCell_MappingCall, False, manual_dict=md
        )
        output_file.write(jinja2.Template(self.TemplateEnterCell_Epilogue).render({}))

    TemplateLoadCell_Prologue = """
void {{FULL_QUALIFIED_CLASSNAME}}::loadCell( const peano4::grid::GridTraversalEvent&  event ) {
  logTraceInWith2Arguments( "loadCell(...)", _spacetreeId, event.toString() );
  
  [[maybe_unused]] const peano4::datamanagement::CellMarker    cellMarker( event );
  [[maybe_unused]] const peano4::datamanagement::FaceMarker    faceMarker( event );
  [[maybe_unused]] const peano4::datamanagement::VertexMarker  vertexMarker(event);
  
  std::vector< tarch::multicore::Task* >  tasks;
"""

    TemplateStoreCell_Prologue = """
void {{FULL_QUALIFIED_CLASSNAME}}::storeCell( const peano4::grid::GridTraversalEvent&  event ) {
  logTraceInWith2Arguments( "storeCell(...)", _spacetreeId, event.toString() );
  
  [[maybe_unused]] const peano4::datamanagement::CellMarker    cellMarker( event );
  [[maybe_unused]] const peano4::datamanagement::FaceMarker    faceMarker( event );
  [[maybe_unused]] const peano4::datamanagement::VertexMarker  vertexMarker(event);

  std::vector< tarch::multicore::Task* >  tasks;
"""

    TemplateLeaveCell_Prologue = """
void {{FULL_QUALIFIED_CLASSNAME}}::leaveCell( const peano4::grid::GridTraversalEvent&  event ) {
  logTraceInWith2Arguments( "leaveCell(...)", _spacetreeId, event.toString() );
  
  [[maybe_unused]] const peano4::datamanagement::CellMarker    cellMarker( event );
  [[maybe_unused]] const peano4::datamanagement::FaceMarker    faceMarker( event );
  [[maybe_unused]] const peano4::datamanagement::VertexMarker  vertexMarker(event);
"""

    TemplateLeaveCell_MappingCall = """
  {
    peano4::datamanagement::CellMarker marker = cellMarker;
    
    if (
      marker.isLocal()
    ) {
      {{ACTIVE_ACTION_SET}}.touchCellLastTime(
         marker
        {{_COMMA_MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_FINE_GRID_CELL_ARGUMENTS_CELL_EVENT}}
        {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS_CELL_EVENT}}
      );
    }
  }
"""

    TemplateLeaveCell_CellStore_MappingCall = """
  {
    peano4::datamanagement::CellMarker marker = cellMarker;
    
    if (
      event.getCellData()==peano4::grid::TraversalObserver::CreateOrDestroyPersistentGridEntity
      and
      marker.isLocal()
    ) {
    {{ACTIVE_ACTION_SET}}.destroyCell(
      marker
        {{_COMMA_MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS}}
        ,{{MAPPING_SIGNATURE_FINE_GRID_CELL_ARGUMENTS_PICK_ENTRY}}
        {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS}}
      );
    }
  }
"""

    TemplateStoreCell_CellStore = """
  // Store cell {{logical_type_name}}
  std::function<void ()> storeCell{{logical_type_name}} =
    [&]()->void {
    ::peano4::datamanagement::storeCellData<{{full_qualified_type}}>(
      {{subnamespace}}repositories::DataRepository::_{{logical_type_name}}Stack,
      _spacetreeId,
      event,
      cellMarker,
      "{{name}}",
      [&](const peano4::datamanagement::CellMarker& marker) -> ::peano4::grid::LoadStoreComputeFlag {
        return {{full_qualified_type}}::loadStoreComputeFlag(marker{% for arg in ADDITIONAL_LOAD_STORE_ARGUMENTS %}, {{arg[0]}}{% endfor %});
      }
    );
  };

  tasks.push_back( new tarch::multicore::TaskWithoutCopyOfFunctor(
    tarch::multicore::Task::DontFuse,
    tarch::multicore::Task::DefaultPriority,
    storeCell{{logical_type_name}}
  ));
"""

    TemplateLeaveCell_FaceStore_MappingCall = """
  // Handle face {{logical_type_name}}
  {
    peano4::datamanagement::FaceMarker  marker = faceMarker;
    
    for (int i=0; i<TwoTimesD; i++) {
      int outFaceStack      = event.getFaceDataTo(i);
      int pick              = event.getFaceDataFrom(i);

      marker.select(pick);

      if (
        outFaceStack==peano4::grid::TraversalObserver::CreateOrDestroyPersistentGridEntity
        and
        marker.isLocal()
      ) {
        {{ACTIVE_ACTION_SET}}.touchFaceLastTime(
           marker
          {{_COMMA_MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS}}
          ,{{MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS_PICK_ENTRY}}
          {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS}}
        );
        {{ACTIVE_ACTION_SET}}.destroyPersistentFace(
           marker
          {{_COMMA_MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS}}
          ,{{MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS_PICK_ENTRY}}
          {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS}}
        );
      }
      else if (
        outFaceStack==peano4::grid::TraversalObserver::CreateOrDestroyHangingGridEntity
        and
        marker.isLocal()
      ) {
        {{ACTIVE_ACTION_SET}}.destroyHangingFace(
           marker
          {{_COMMA_MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS}}
          ,{{MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS_PICK_ENTRY}}
          {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS}}
        );
      }
      else if (
        peano4::grid::PeanoCurve::isInOutStack(outFaceStack)
        and
        marker.isLocal()
      ) {
        {{ACTIVE_ACTION_SET}}.touchFaceLastTime(
           marker
          {{_COMMA_MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS}}
          ,{{MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS_PICK_ENTRY}}
          {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS}}
        );
      }
    }
  }
"""

    TemplateStoreCell_FaceStore = """
  // Store face {{logical_type_name}}
  std::function<void ()> storeFace{{logical_type_name}} =
    [&]()->void {
    ::peano4::datamanagement::storeFaceData<{{full_qualified_type}}>(
      {{subnamespace}}repositories::DataRepository::_{{logical_type_name}}Stack,
      _spacetreeId,
      event,
      faceMarker,
      "{{name}}",
      [&](const peano4::datamanagement::FaceMarker& marker) -> ::peano4::grid::LoadStoreComputeFlag {
        return {{full_qualified_type}}::loadStoreComputeFlag(marker{% for arg in ADDITIONAL_LOAD_STORE_ARGUMENTS %}, {{arg[0]}}{% endfor %});
      }
    );
  };

  tasks.push_back( new tarch::multicore::TaskWithoutCopyOfFunctor(
    tarch::multicore::Task::DontFuse,
    tarch::multicore::Task::DefaultPriority,
    storeFace{{logical_type_name}}
  ));
"""

    TemplateLeaveCell_VertexStore_MappingCall = """
  // Handle vertex {{logical_type_name}}
  {
    peano4::datamanagement::VertexMarker  marker = vertexMarker;

    for (int i=0; i<TwoPowerD; i++) {
      int outVertexStack        = event.getVertexDataTo(i);
      int pick                  = event.getVertexDataFrom(i);

      marker.select(pick);

      if (
        outVertexStack==peano4::grid::TraversalObserver::CreateOrDestroyPersistentGridEntity
        and
        marker.isLocal()
      ) {
        {{ACTIVE_ACTION_SET}}.touchVertexLastTime(
           marker
          ,{{MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS_PICK_ENTRY}}
          {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS}}
        );
        {{ACTIVE_ACTION_SET}}.destroyPersistentVertex(
           marker
          ,{{MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS_PICK_ENTRY}}
          {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS}}
        );
      }
      else if (
        outVertexStack==peano4::grid::TraversalObserver::CreateOrDestroyHangingGridEntity
        and
        marker.isLocal()
      ) {
        {{ACTIVE_ACTION_SET}}.destroyHangingVertex(
           marker
          ,{{MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS_PICK_ENTRY}}
          {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS}}
        );
      }
      else if (
        peano4::grid::PeanoCurve::isInOutStack(outVertexStack)
        and
        marker.isLocal()
      ) {
        {{ACTIVE_ACTION_SET}}.touchVertexLastTime(
           marker
          ,{{MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS_PICK_ENTRY}}
          {{_COMMA_MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS_COMMA_MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS}}
        );
      }
    }
  }
"""

    TemplateStoreCell_VertexStore = """
  // Store vertex {{logical_type_name}}
  std::function<void ()> storeVertex{{logical_type_name}} =
    [&]()->void {
    ::peano4::datamanagement::storeVertexData<{{full_qualified_type}}>(
      {{subnamespace}}repositories::DataRepository::_{{logical_type_name}}Stack,
      _spacetreeId,
      event,
      vertexMarker,
      "{{name}}",
      [&](const peano4::datamanagement::VertexMarker& marker) -> ::peano4::grid::LoadStoreComputeFlag {
        return {{full_qualified_type}}::loadStoreComputeFlag(marker{% for arg in ADDITIONAL_LOAD_STORE_ARGUMENTS %}, {{arg[0]}}{% endfor %});
      }
    );
  };

  tasks.push_back( new tarch::multicore::TaskWithoutCopyOfFunctor(
    tarch::multicore::Task::DontFuse,
    tarch::multicore::Task::DefaultPriority,
    storeVertex{{logical_type_name}}
  ));
"""

    TemplateLeaveCell_Epilogue = """
  logTraceOutWith1Argument( "leaveCell(...)", _spacetreeId );
}


"""

    TemplateLoadCell_Epilogue = """
  tarch::multicore::spawnAndWait(tasks);
  logTraceOutWith1Argument( "loadCell(...)", _spacetreeId );
}


"""

    TemplateStoreCell_Epilogue = """
  tarch::multicore::spawnAndWait( tasks );
  logTraceOutWith1Argument( "storeCell(...)", _spacetreeId );
}


"""

    def __generate_leaveCell(self, output_file):
        """
        Generates enter cell
        """
        output_file.write(
            jinja2.Template(self.TemplateLeaveCell_Prologue).render(
                **self.mkSubDict(["FULL_QUALIFIED_CLASSNAME"])
            )
        )

        md = self.mkSubDict(
            [
                ",MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS,MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS,MAPPING_SIGNATURE_FINE_GRID_CELL_ARGUMENTS_CELL_EVENT",
                ",MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS,MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS,MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS_CELL_EVENT",
            ]
        )
        self.__format_template_per_task_group(
            output_file, self.TemplateLeaveCell_MappingCall, True, manual_dict=md
        )

        if len(self.cells) > 0:
            md = self.mkSubDict(
                [
                    ",MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS,MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS",
                    "MAPPING_SIGNATURE_FINE_GRID_CELL_ARGUMENTS_PICK_ENTRY",
                    ",MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS,MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS,MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS",
                ]
            )
            self.__format_template_per_task_group(
                output_file,
                self.TemplateLeaveCell_CellStore_MappingCall,
                True,
                manual_dict=md,
            )

        if len(self.faces) > 0:
            md = self.mkSubDict(
                [
                    "name",
                    ",MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS",
                    "MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS_PICK_ENTRY",
                    ",MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS,MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS,MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS",
                    ",MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS",
                    "MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS_PICK_ENTRY",
                    ",MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS,MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS,MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS",
                ]
            )
            self.__format_template_per_task_group(
                output_file,
                self.TemplateLeaveCell_FaceStore_MappingCall,
                True,
                manual_dict=md,
            )

        if len(self.vertices) > 0:
            md = self.mkSubDict(
                [
                    "name",
                    "MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS_PICK_ENTRY",
                    ",MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS,MAPPING_SIGNATURE_FINE_GRID_FACES_ARGUMENTS,MAPPING_SIGNATURE_FINE_GRID_CELL_ARGUMENTS",
                    ",MAPPING_SIGNATURE_COARSE_GRID_VERTICES_ARGUMENTS,MAPPING_SIGNATURE_COARSE_GRID_FACES_ARGUMENTS,MAPPING_SIGNATURE_COARSE_GRID_CELL_ARGUMENTS",
                    "MAPPING_SIGNATURE_FINE_GRID_VERTICES_ARGUMENTS_PICK_ENTRY",
                ]
            )
            self.__format_template_per_task_group(
                output_file,
                self.TemplateLeaveCell_VertexStore_MappingCall,
                True,
                manual_dict=md,
            )

        output_file.write(jinja2.Template(self.TemplateLeaveCell_Epilogue).render({}))

    def __generate_storeCell(self, output_file):
        output_file.write(
            jinja2.Template(self.TemplateStoreCell_Prologue).render(**self.d)
        )

        cells_inverted = [x for x in self.cells]
        faces_inverted = [x for x in self.faces]
        vertices_inverted = [x for x in self.vertices]

        cells_inverted.reverse()
        faces_inverted.reverse()
        vertices_inverted.reverse()

        for cell in cells_inverted:
            temp = {
                "name": cell.name,
                "logical_type_name": cell.get_logical_type_name(),
                "full_qualified_type": cell.get_full_qualified_type(),
                "ADDITIONAL_LOAD_STORE_ARGUMENTS": cell.additional_load_and_store_arguments,
                "subnamespace": cell.subnamespace,
            }
            self.d["name"] = cell.name
            output_file.write(
                jinja2.Template(self.TemplateStoreCell_CellStore).render(**temp)
            )

        for face in faces_inverted:
            temp = {
                "name": face.name,
                "enumeration_type": face.get_enumeration_type(),
                "logical_type_name": face.get_logical_type_name(),
                "full_qualified_type": face.get_full_qualified_type(),
                "ADDITIONAL_LOAD_STORE_ARGUMENTS": face.additional_load_and_store_arguments,
                "subnamespace": face.subnamespace,
            }
            self.d["name"] = face.name
            output_file.write(
                jinja2.Template(self.TemplateStoreCell_FaceStore).render(**temp)
            )

        for vertex in vertices_inverted:
            temp = {
                "name": vertex.name,
                "enumeration_type": vertex.get_enumeration_type(),
                "logical_type_name": vertex.get_logical_type_name(),
                "full_qualified_type": vertex.get_full_qualified_type(),
                "ADDITIONAL_LOAD_STORE_ARGUMENTS": vertex.additional_load_and_store_arguments,
                "subnamespace": vertex.subnamespace,
            }
            self.d["name"] = vertex.name
            output_file.write(
                jinja2.Template(self.TemplateStoreCell_VertexStore).render(**temp)
            )

        output_file.write(jinja2.Template(self.TemplateStoreCell_Epilogue).render({}))

    TemplateExchangeRoutines_exchangeAllVerticalDataExchangeStacks_Prologue = """
void {{FULL_QUALIFIED_CLASSNAME}}::exchangeAllVerticalDataExchangeStacks( int masterId ) {
  logTraceInWith2Arguments( "exchangeAllVerticalDataExchangeStacks(...)", masterId, _spacetreeId  );
"""

    TemplateExchangeRoutines_exchangeAllVerticalDataExchangeStacks_Exchange = """
  peano4::parallel::SpacetreeSet::exchangeAllVerticalDataExchangeStacks(
    {{DATASET}},
    _spacetreeId,
    masterId
  );
"""

    TemplateExchangeRoutines_exchangeAllVerticalDataExchangeStacks_Epilogue = """
  logTraceOut( "exchangeAllVerticalDataExchangeStacks(...)" );
}


"""

    TemplateExchangeRoutines_exchangeAllHorizontalDataExchangeStacks_Prologue = """
void {{FULL_QUALIFIED_CLASSNAME}}::exchangeAllHorizontalDataExchangeStacks( bool symmetricDataCardinality ) {
  logTraceInWith2Arguments( "exchangeAllHorizontalDataExchangeStacks(...)", symmetricDataCardinality, _spacetreeId  );
"""

    TemplateExchangeRoutines_exchangeAllHorizontalDataExchangeStacks_Exchange = """
  peano4::parallel::SpacetreeSet::exchangeAllHorizontalDataExchangeStacks(
    {{DATASET}},
    _spacetreeId,
    symmetricDataCardinality
  );
"""

    TemplateExchangeRoutines_exchangeAllHorizontalDataExchangeStacks_Epilogue = """
  logTraceOut( "exchangeAllHorizontalDataExchangeStacks(...)" );
}


"""

    TemplateExchangeRoutines_exchangeAllPeriodicBoundaryDataStacks_Prologue = """
void {{FULL_QUALIFIED_CLASSNAME}}::exchangeAllPeriodicBoundaryDataStacks() {
  logTraceInWith1Argument( "exchangeAllPeriodicBoundaryDataStacks()", _spacetreeId  );
"""

    TemplateExchangeRoutines_exchangeAllPeriodicBoundaryDataStacks_Exchange = """
  peano4::parallel::SpacetreeSet::exchangeAllPeriodicBoundaryDataStacks(
    {{DATASET}},
    _spacetreeId
  );
"""

    TemplateExchangeRoutines_exchangeAllPeriodicBoundaryDataStacks_Epilogue = """
  logTraceOut( "exchangeAllPeriodicBoundaryDataStacks()" );
}


"""

    TemplateExchangeRoutines_streamDataFromSplittingTreeToNewTree_Prologue = """
void {{FULL_QUALIFIED_CLASSNAME}}::streamDataFromSplittingTreeToNewTree(int newWorker) {
  logTraceInWith2Arguments( "streamDataFromSplittingTreeToNewTree(int)", _spacetreeId, newWorker );
"""

    TemplateExchangeRoutines_streamDataFromSplittingTreeToNewTree_Exchange = """
  peano4::parallel::SpacetreeSet::streamDataFromSplittingTreeToNewTree(
    {{DATASET}},
    _spacetreeId,
    newWorker
  );
"""

    TemplateExchangeRoutines_streamDataFromSplittingTreeToNewTree_Epilogue = """
  logTraceOut( "streamDataFromSplittingTreeToNewTree(int)");
}


"""

    TemplateExchangeRoutines_streamDataFromJoiningTreeToMasterTree_Prologue = """
void {{FULL_QUALIFIED_CLASSNAME}}::streamDataFromJoiningTreeToMasterTree(int master) {
  logTraceInWith2Arguments( "streamDataFromJoiningTreeToMasterTree(int)", _spacetreeId, master );
"""

    TemplateExchangeRoutines_streamDataFromJoiningTreeToMasterTree_Exchange = """
  peano4::parallel::SpacetreeSet::streamDataFromJoiningTreeToMasterTree(
    {{DATASET}},
    _spacetreeId,
    master
  );
"""

    TemplateExchangeRoutines_streamDataFromJoiningTreeToMasterTree_Epilogue = """
  logTraceOut( "streamDataFromJoiningTreeToMasterTree(int)");
}


"""

    TemplateExchangeRoutines_finishAllOutstandingSendsAndReceives_Prologue = """
void {{FULL_QUALIFIED_CLASSNAME}}::finishAllOutstandingSendsAndReceives() {
  logTraceInWith1Argument( "finishAllOutstandingSendsAndReceives()", _spacetreeId );
"""

    TemplateExchangeRoutines_finishAllOutstandingSendsAndReceives_Exchange = """
  peano4::parallel::SpacetreeSet::finishAllOutstandingSendsAndReceives(
    {{DATASET}},
    _spacetreeId
  );
"""

    TemplateExchangeRoutines_finishAllOutstandingSendsAndReceives_Epilogue = """
  logTraceOut( "finishAllOutstandingSendsAndReceives()");
}


"""

    TemplateSendVertex_Prologue = """
void {{FULL_QUALIFIED_CLASSNAME}}::sendVertex(int position, int toStack, ::peano4::grid::TraversalObserver::SendReceiveContext context, const peano4::grid::GridTraversalEvent&  event) {
  logTraceInWith4Arguments( "sendVertex(int,int,int)", position, toStack, event.toString(), _spacetreeId );

  [[maybe_unused]] const peano4::datamanagement::VertexMarker  vertexMarker(event);
"""

    TemplateSendVertex_SendCall = """
  ::peano4::datamanagement::sendVertexData<{{full_qualified_type}}>(
    {{subnamespace}}repositories::DataRepository::_{{logical_type_name}}Stack,
    _spacetreeId,
    position,
    toStack,
    context,
    vertexMarker,
    [&](const peano4::datamanagement::VertexMarker& marker) -> ::peano4::grid::LoadStoreComputeFlag {
        return {{full_qualified_type}}::loadStoreComputeFlag(marker{% for arg in ADDITIONAL_LOAD_STORE_ARGUMENTS %}, {{arg[0]}}{% endfor %});
      },
    [&](const {{full_qualified_type}}& data, const peano4::datamanagement::VertexMarker& marker) -> bool {
        return data.send(marker);
      }
  );
"""

    TemplateSendVertex_Epilogue = """
  logTraceOut( "sendVertex(int,int,int)");
}


"""

    TemplateSendFace_Prologue = """
void {{FULL_QUALIFIED_CLASSNAME}}::sendFace(int position, int toStack, ::peano4::grid::TraversalObserver::SendReceiveContext context, const peano4::grid::GridTraversalEvent&  event) {
  logTraceInWith4Arguments( "sendFace(int,int,int)", position, toStack, event.toString(), _spacetreeId );

  [[maybe_unused]] const peano4::datamanagement::FaceMarker    faceMarker( event );
"""

    TemplateSendFace_SendCall = """
  ::peano4::datamanagement::sendFaceData<{{full_qualified_type}}>(
    {{subnamespace}}repositories::DataRepository::_{{logical_type_name}}Stack,
    _spacetreeId,
    position,
    toStack,
    context,
    faceMarker,
    [&](const peano4::datamanagement::FaceMarker& marker) -> ::peano4::grid::LoadStoreComputeFlag {
        return {{full_qualified_type}}::loadStoreComputeFlag(marker{% for arg in ADDITIONAL_LOAD_STORE_ARGUMENTS %}, {{arg[0]}}{% endfor %});
      },
    [&](const {{full_qualified_type}}& data, const peano4::datamanagement::FaceMarker& marker) -> bool {
        return data.send(marker{% for arg in ADDITIONAL_LOAD_STORE_ARGUMENTS %}, {{arg[0]}}{% endfor %});
      }
  );
"""

    TemplateSendFace_Epilogue = """
  logTraceOut( "sendFace(int,int,int)");
}


"""

    TemplateSendCell_Prologue = """
void {{FULL_QUALIFIED_CLASSNAME}}::sendCell(int toStack, ::peano4::grid::TraversalObserver::SendReceiveContext context, const peano4::grid::GridTraversalEvent&  event) {
  logTraceInWith3Arguments( "sendCell(int,int,int)", toStack, event.toString(), _spacetreeId );

  [[maybe_unused]] const peano4::datamanagement::CellMarker    cellMarker( event );
"""

    TemplateSendCell_SendCall = """
  ::peano4::datamanagement::sendCellData<{{full_qualified_type}}>(
    {{subnamespace}}repositories::DataRepository::_{{logical_type_name}}Stack,
    _spacetreeId,
    toStack,
    context,
    cellMarker,
    [&](const peano4::datamanagement::CellMarker& marker) -> ::peano4::grid::LoadStoreComputeFlag {
        return {{full_qualified_type}}::loadStoreComputeFlag(marker{% for arg in ADDITIONAL_LOAD_STORE_ARGUMENTS %}, {{arg[0]}}{% endfor %});
      },
    [&](const {{full_qualified_type}}& data, const peano4::datamanagement::CellMarker& marker) -> bool {
        return data.send(marker{% for arg in ADDITIONAL_LOAD_STORE_ARGUMENTS %}, {{arg[0]}}{% endfor %});
      }
  );
"""

    TemplateSendCell_Epilogue = """
  logTraceOut( "sendCell(int,int,int)");
}


"""

    TemplateReceiveAndMergeCell_Prologue = """
void {{FULL_QUALIFIED_CLASSNAME}}::receiveAndMergeCell(int fromStack, peano4::grid::TraversalObserver::SendReceiveContext context, const peano4::grid::GridTraversalEvent& event ) {
  logTraceInWith3Arguments( "receiveAndMergeCell(...)", fromStack, event.toString(), _spacetreeId );
  
  [[maybe_unused]] const peano4::datamanagement::CellMarker    cellMarker( event );
"""

    TemplateReceiveAndMergeCell_Epilogue = """
  logTraceOut( "receiveAndMergeCell(...)");
}


"""

    TemplateReceiveAndMergeVertex_Prologue = """
void {{FULL_QUALIFIED_CLASSNAME}}::receiveAndMergeVertex(int position, int fromStack, peano4::grid::TraversalObserver::SendReceiveContext context, const peano4::grid::GridTraversalEvent& event) {
  logTraceInWith4Arguments( "receiveAndMergeVertex(...)", position, fromStack, event.toString(), _spacetreeId );

  [[maybe_unused]] const peano4::datamanagement::VertexMarker  vertexMarker(event);
"""

    TemplateReceiveAndMergeVertex_Epilogue = """
  logTraceOut( "receiveAndMergeVertex(...)");
}


"""

    TemplateReceiveAndMergeFace_Epilogue = """
  logTraceOut( "receiveAndMergeFace(...)");
}


"""

    TemplateReceiveAndMergeFace_Prologue = """
void {{FULL_QUALIFIED_CLASSNAME}}::receiveAndMergeFace(int position, int fromStack, peano4::grid::TraversalObserver::SendReceiveContext context, const peano4::grid::GridTraversalEvent& event) {
  logTraceInWith4Arguments( "receiveAndMergeFace(...)", position, fromStack, event.toString(), _spacetreeId );
  
  [[maybe_unused]] const peano4::datamanagement::FaceMarker    faceMarker( event );
"""

    TemplateReceiveAndMergeVertex_ReceiveAndMergeCalls = """
  ::peano4::datamanagement::receiveAndMergeVertexData<{{full_qualified_type}}>(
    {{subnamespace}}repositories::DataRepository::_{{logical_type_name}}Stack,
    _spacetreeId,
    position,
    fromStack,
    context,
    event,
    vertexMarker,
    [&](const peano4::datamanagement::VertexMarker& marker) -> ::peano4::grid::LoadStoreComputeFlag {
        return {{full_qualified_type}}::loadStoreComputeFlag(marker{% for arg in ADDITIONAL_LOAD_STORE_ARGUMENTS %}, {{arg[0]}}{% endfor %});
      },
    [&]({{full_qualified_type}}& data, const peano4::datamanagement::VertexMarker& marker) -> bool {
        return data.receiveAndMerge(marker{% for arg in ADDITIONAL_LOAD_STORE_ARGUMENTS %}, {{arg[0]}}{% endfor %});
      }
  );
"""

    TemplateReceiveAndMergeFace_ReceiveAndMergeCalls = """
  ::peano4::datamanagement::receiveAndMergeFaceData<{{full_qualified_type}}>(
    {{subnamespace}}repositories::DataRepository::_{{logical_type_name}}Stack,
    _spacetreeId,
    position,
    fromStack,
    context,
    event,
    faceMarker,
    [&](const peano4::datamanagement::FaceMarker& marker) -> ::peano4::grid::LoadStoreComputeFlag {
        return {{full_qualified_type}}::loadStoreComputeFlag(marker{% for arg in ADDITIONAL_LOAD_STORE_ARGUMENTS %}, {{arg[0]}}{% endfor %});
      },
    [&]({{full_qualified_type}}& data, const peano4::datamanagement::FaceMarker& marker) -> bool {
        return data.receiveAndMerge(marker{% for arg in ADDITIONAL_LOAD_STORE_ARGUMENTS %}, {{arg[0]}}{% endfor %});
      }
  );
"""

    TemplateReceiveAndMergeCell_ReceiveAndMergeCalls = """
  ::peano4::datamanagement::receiveAndMergeCellData<{{full_qualified_type}}>(
    {{subnamespace}}repositories::DataRepository::_{{logical_type_name}}Stack,
    _spacetreeId,
    fromStack,
    context,
    event,
    cellMarker,
    [&](const peano4::datamanagement::CellMarker& marker) -> ::peano4::grid::LoadStoreComputeFlag {
        return {{full_qualified_type}}::loadStoreComputeFlag(marker{% for arg in ADDITIONAL_LOAD_STORE_ARGUMENTS %}, {{arg[0]}}{% endfor %});
      },
    [&]({{full_qualified_type}}& data, const peano4::datamanagement::CellMarker& marker) -> bool {
        return data.receiveAndMerge(marker{% for arg in ADDITIONAL_LOAD_STORE_ARGUMENTS %}, {{arg[0]}}{% endfor %});
      }
  );
"""

    TemplateExchangeRoutines_deleteAllStacks_Prologue = """
void {{FULL_QUALIFIED_CLASSNAME}}::deleteAllStacks() {
  logTraceInWith1Argument( "deleteAllStacks()", _spacetreeId );
"""

    TemplateExchangeRoutines_deleteAllStacks_Exchange = """
  peano4::parallel::SpacetreeSet::deleteAllStacks(
    {{DATASET}},
    _spacetreeId
  );
"""

    TemplateExchangeRoutines_deleteAllStacks_Epilogue = """
  logTraceOut( "deleteAllStacks()");
}


"""

    def __generate_exchange_routines(self, output_file):
        s = ""

        generic_dict_for_prologue_and_epilogue = {
            "FULL_QUALIFIED_CLASSNAME": self.d["FULL_QUALIFIED_CLASSNAME"]
        }

        s += jinja2.Template(
            self.TemplateExchangeRoutines_exchangeAllVerticalDataExchangeStacks_Prologue
        ).render(**generic_dict_for_prologue_and_epilogue)
        for cell in self.cells:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_exchangeAllVerticalDataExchangeStacks_Exchange
            ).render(
                **{
                    "DATASET": cell.subnamespace + "repositories::DataRepository::_"
                    + cell.get_logical_type_name()
                    + "Stack"
                }
            )
        for face in self.faces:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_exchangeAllVerticalDataExchangeStacks_Exchange
            ).render(
                **{
                    "DATASET": face.subnamespace + "repositories::DataRepository::_"
                    + face.get_logical_type_name()
                    + "Stack"
                }
            )
        for vertex in self.vertices:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_exchangeAllVerticalDataExchangeStacks_Exchange
            ).render(
                **{
                    "DATASET": vertex.subnamespace + "repositories::DataRepository::_"
                    + vertex.get_logical_type_name()
                    + "Stack"
                }
            )
        s += jinja2.Template(
            self.TemplateExchangeRoutines_exchangeAllVerticalDataExchangeStacks_Epilogue
        ).render({})

        s += jinja2.Template(
            self.TemplateExchangeRoutines_exchangeAllHorizontalDataExchangeStacks_Prologue
        ).render(**generic_dict_for_prologue_and_epilogue)
        for cell in self.cells:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_exchangeAllHorizontalDataExchangeStacks_Exchange
            ).render(
                **{
                    "DATASET": cell.subnamespace + "repositories::DataRepository::_"
                    + cell.get_logical_type_name()
                    + "Stack"
                }
            )
        for face in self.faces:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_exchangeAllHorizontalDataExchangeStacks_Exchange
            ).render(
                **{
                    "DATASET": face.subnamespace + "repositories::DataRepository::_"
                    + face.get_logical_type_name()
                    + "Stack"
                }
            )
        for vertex in self.vertices:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_exchangeAllHorizontalDataExchangeStacks_Exchange
            ).render(
                **{
                    "DATASET": vertex.subnamespace + "repositories::DataRepository::_"
                    + vertex.get_logical_type_name()
                    + "Stack"
                }
            )
        s += jinja2.Template(
            self.TemplateExchangeRoutines_exchangeAllHorizontalDataExchangeStacks_Epilogue
        ).render({})

        s += jinja2.Template(
            self.TemplateExchangeRoutines_exchangeAllPeriodicBoundaryDataStacks_Prologue
        ).render(**generic_dict_for_prologue_and_epilogue)
        for cell in self.cells:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_exchangeAllPeriodicBoundaryDataStacks_Exchange
            ).render(
                **{
                    "DATASET": cell.subnamespace + "repositories::DataRepository::_"
                    + cell.get_logical_type_name()
                    + "Stack"
                }
            )
        for face in self.faces:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_exchangeAllPeriodicBoundaryDataStacks_Exchange
            ).render(
                **{
                    "DATASET": face.subnamespace + "repositories::DataRepository::_"
                    + face.get_logical_type_name()
                    + "Stack"
                }
            )
        for vertex in self.vertices:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_exchangeAllPeriodicBoundaryDataStacks_Exchange
            ).render(
                **{
                    "DATASET": vertex.subnamespace + "repositories::DataRepository::_"
                    + vertex.get_logical_type_name()
                    + "Stack"
                }
            )
        s += jinja2.Template(
            self.TemplateExchangeRoutines_exchangeAllPeriodicBoundaryDataStacks_Epilogue
        ).render({})

        s += jinja2.Template(
            self.TemplateExchangeRoutines_streamDataFromSplittingTreeToNewTree_Prologue
        ).render(**generic_dict_for_prologue_and_epilogue)
        for cell in self.cells:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_streamDataFromSplittingTreeToNewTree_Exchange
            ).render(
                **{
                    "DATASET": cell.subnamespace + "repositories::DataRepository::_"
                    + cell.get_logical_type_name()
                    + "Stack"
                }
            )
        for face in self.faces:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_streamDataFromSplittingTreeToNewTree_Exchange
            ).render(
                **{
                    "DATASET": face.subnamespace + "repositories::DataRepository::_"
                    + face.get_logical_type_name()
                    + "Stack"
                }
            )
        for vertex in self.vertices:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_streamDataFromSplittingTreeToNewTree_Exchange
            ).render(
                **{
                    "DATASET": vertex.subnamespace + "repositories::DataRepository::_"
                    + vertex.get_logical_type_name()
                    + "Stack"
                }
            )
        s += jinja2.Template(
            self.TemplateExchangeRoutines_streamDataFromSplittingTreeToNewTree_Epilogue
        ).render({})

        s += jinja2.Template(
            self.TemplateExchangeRoutines_streamDataFromJoiningTreeToMasterTree_Prologue
        ).render(**generic_dict_for_prologue_and_epilogue)
        for cell in self.cells:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_streamDataFromJoiningTreeToMasterTree_Exchange
            ).render(
                **{
                    "DATASET": cell.subnamespace + "repositories::DataRepository::_"
                    + cell.get_logical_type_name()
                    + "Stack"
                }
            )
        for face in self.faces:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_streamDataFromJoiningTreeToMasterTree_Exchange
            ).render(
                **{
                    "DATASET": face.subnamespace + "repositories::DataRepository::_"
                    + face.get_logical_type_name()
                    + "Stack"
                }
            )
        for vertex in self.vertices:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_streamDataFromJoiningTreeToMasterTree_Exchange
            ).render(
                **{
                    "DATASET": vertex.subnamespace + "repositories::DataRepository::_"
                    + vertex.get_logical_type_name()
                    + "Stack"
                }
            )
        s += jinja2.Template(
            self.TemplateExchangeRoutines_streamDataFromJoiningTreeToMasterTree_Epilogue
        ).render({})

        s += jinja2.Template(
            self.TemplateExchangeRoutines_finishAllOutstandingSendsAndReceives_Prologue
        ).render(**generic_dict_for_prologue_and_epilogue)
        for cell in self.cells:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_finishAllOutstandingSendsAndReceives_Exchange
            ).render(
                **{
                    "DATASET": cell.subnamespace + "repositories::DataRepository::_"
                    + cell.get_logical_type_name()
                    + "Stack"
                }
            )
        for face in self.faces:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_finishAllOutstandingSendsAndReceives_Exchange
            ).render(
                **{
                    "DATASET": face.subnamespace + "repositories::DataRepository::_"
                    + face.get_logical_type_name()
                    + "Stack"
                }
            )
        for vertex in self.vertices:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_finishAllOutstandingSendsAndReceives_Exchange
            ).render(
                **{
                    "DATASET": vertex.subnamespace + "repositories::DataRepository::_"
                    + vertex.get_logical_type_name()
                    + "Stack"
                }
            )
        s += jinja2.Template(
            self.TemplateExchangeRoutines_finishAllOutstandingSendsAndReceives_Epilogue
        ).render({})

        s += jinja2.Template(self.TemplateSendVertex_Prologue).render(
            **generic_dict_for_prologue_and_epilogue
        )
        for vertex in self.vertices:
            temp = {
                "name": vertex.name,
                "enumeration_type": vertex.get_enumeration_type(),
                "logical_type_name": vertex.get_logical_type_name(),
                "full_qualified_type": vertex.get_full_qualified_type(),
                "FULL_QUALIFIED_CLASSNAME": self.d["FULL_QUALIFIED_CLASSNAME"],
                "ADDITIONAL_LOAD_STORE_ARGUMENTS": vertex.additional_load_and_store_arguments,
                "subnamespace": vertex.subnamespace,
            }
            s += jinja2.Template(self.TemplateSendVertex_SendCall).render(**temp)
        s += jinja2.Template(self.TemplateSendVertex_Epilogue).render(
            **generic_dict_for_prologue_and_epilogue
        )

        s += jinja2.Template(self.TemplateSendFace_Prologue).render(
            **generic_dict_for_prologue_and_epilogue
        )
        for face in self.faces:
            temp = {
                "name": face.name,
                "enumeration_type": face.get_enumeration_type(),
                "logical_type_name": face.get_logical_type_name(),
                "full_qualified_type": face.get_full_qualified_type(),
                "FULL_QUALIFIED_CLASSNAME": self.d["FULL_QUALIFIED_CLASSNAME"],
                "ADDITIONAL_LOAD_STORE_ARGUMENTS": face.additional_load_and_store_arguments,
                "subnamespace": face.subnamespace,
            }
            s += jinja2.Template(self.TemplateSendFace_SendCall).render(**temp)
        s += jinja2.Template(self.TemplateSendFace_Epilogue).render(
            **generic_dict_for_prologue_and_epilogue
        )

        s += jinja2.Template(self.TemplateSendCell_Prologue).render(
            **generic_dict_for_prologue_and_epilogue
        )
        for cell in self.cells:
            temp = {
                "name": cell.name,
                "logical_type_name": cell.get_logical_type_name(),
                "full_qualified_type": cell.get_full_qualified_type(),
                "FULL_QUALIFIED_CLASSNAME": self.d["FULL_QUALIFIED_CLASSNAME"],
                "ADDITIONAL_LOAD_STORE_ARGUMENTS": cell.additional_load_and_store_arguments,
                "subnamespace": cell.subnamespace,
            }
            s += jinja2.Template(self.TemplateSendCell_SendCall).render(**temp)
        s += jinja2.Template(self.TemplateSendCell_Epilogue).render(
            **generic_dict_for_prologue_and_epilogue
        )

        s += jinja2.Template(self.TemplateReceiveAndMergeCell_Prologue).render(
            **generic_dict_for_prologue_and_epilogue
        )
        for cell in self.cells:
            temp = {
                "name": cell.name,
                "logical_type_name": cell.get_logical_type_name(),
                "full_qualified_type": cell.get_full_qualified_type(),
                "FULL_QUALIFIED_CLASSNAME": self.d["FULL_QUALIFIED_CLASSNAME"],
                "ADDITIONAL_LOAD_STORE_ARGUMENTS": cell.additional_load_and_store_arguments,
                "subnamespace": cell.subnamespace,
            }
            s += jinja2.Template(
                self.TemplateReceiveAndMergeCell_ReceiveAndMergeCalls
            ).render(**temp)
        s += jinja2.Template(self.TemplateReceiveAndMergeCell_Epilogue).render(
            **generic_dict_for_prologue_and_epilogue
        )

        s += jinja2.Template(self.TemplateReceiveAndMergeFace_Prologue).render(
            **generic_dict_for_prologue_and_epilogue
        )
        for face in self.faces:
            temp = {
                "name": face.name,
                "enumeration_type": face.get_enumeration_type(),
                "logical_type_name": face.get_logical_type_name(),
                "full_qualified_type": face.get_full_qualified_type(),
                "FULL_QUALIFIED_CLASSNAME": self.d["FULL_QUALIFIED_CLASSNAME"],
                "ADDITIONAL_LOAD_STORE_ARGUMENTS": face.additional_load_and_store_arguments,
                "subnamespace": face.subnamespace,
            }
            s += jinja2.Template(
                self.TemplateReceiveAndMergeFace_ReceiveAndMergeCalls
            ).render(**temp)
        s += jinja2.Template(self.TemplateReceiveAndMergeFace_Epilogue).render(
            **generic_dict_for_prologue_and_epilogue
        )

        s += jinja2.Template(self.TemplateReceiveAndMergeVertex_Prologue).render(
            **generic_dict_for_prologue_and_epilogue
        )
        for vertex in self.vertices:
            temp = {
                "name": vertex.name,
                "enumeration_type": vertex.get_enumeration_type(),
                "logical_type_name": vertex.get_logical_type_name(),
                "full_qualified_type": vertex.get_full_qualified_type(),
                "FULL_QUALIFIED_CLASSNAME": self.d["FULL_QUALIFIED_CLASSNAME"],
                "ADDITIONAL_LOAD_STORE_ARGUMENTS": vertex.additional_load_and_store_arguments,
                "subnamespace": vertex.subnamespace,
            }
            s += jinja2.Template(
                self.TemplateReceiveAndMergeVertex_ReceiveAndMergeCalls
            ).render(**temp)
        s += jinja2.Template(self.TemplateReceiveAndMergeVertex_Epilogue).render(
            **generic_dict_for_prologue_and_epilogue
        )

        s += jinja2.Template(
            self.TemplateExchangeRoutines_deleteAllStacks_Prologue
        ).render(**generic_dict_for_prologue_and_epilogue)
        for cell in self.cells:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_deleteAllStacks_Exchange
            ).render(
                **{
                    "DATASET": cell.subnamespace + "repositories::DataRepository::_"
                    + cell.get_logical_type_name()
                    + "Stack"
                }
            )
        for face in self.faces:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_deleteAllStacks_Exchange
            ).render(
                **{
                    "DATASET": face.subnamespace + "repositories::DataRepository::_"
                    + face.get_logical_type_name()
                    + "Stack"
                }
            )
        for vertex in self.vertices:
            s += jinja2.Template(
                self.TemplateExchangeRoutines_deleteAllStacks_Exchange
            ).render(
                **{
                    "DATASET": vertex.subnamespace + "repositories::DataRepository::_"
                    + vertex.get_logical_type_name()
                    + "Stack"
                }
            )
        s += jinja2.Template(
            self.TemplateExchangeRoutines_deleteAllStacks_Epilogue
        ).render(**generic_dict_for_prologue_and_epilogue)

        output_file.write(s)

    TemplateImplementationFilePrologue = """
#include "{{CLASSNAME}}.h"
{{DATAREPOSITORY_INCLUDES}}

#include "peano4/grid/PeanoCurve.h"

#include "peano4/datamanagement/VertexEnumerator.h"
#include "peano4/datamanagement/VertexMarker.h"
#include "peano4/datamanagement/FaceEnumerator.h"
#include "peano4/datamanagement/FaceMarker.h"
#include "peano4/datamanagement/CellMarker.h"
#include "peano4/datamanagement/VertexDataMovement.h"
#include "peano4/datamanagement/FaceDataMovement.h"
#include "peano4/datamanagement/CellDataMovement.h"

#include "peano4/parallel/SpacetreeSet.h"


tarch::logging::Log {{FULL_QUALIFIED_CLASSNAME}}::_log( "{{FULL_QUALIFIED_CLASSNAME}}" );

"""

    def __generate_implementation(self, overwrite, full_qualified_filename):
        if write_file(overwrite, self.default_overwrite, full_qualified_filename):
            output_file = open(full_qualified_filename, "w")
            output_file.write(
                jinja2.Template(self.TemplateImplementationFilePrologue).render(
                    **self.d
                )
            )

            self.__generate_constructor(output_file)
            self.__generate_clone(output_file)
            self.__generate_createGridControlEventsForNextTraversal(output_file)
            self.__generate_beginTraversal(output_file)
            self.__generate_endTraversal(output_file)
            self.__generate_prepareTraversal(output_file)
            self.__generate_unprepareTraversal(output_file)
            self.__generate_loadCell(output_file)
            self.__generate_enterCell(output_file)
            self.__generate_leaveCell(output_file)
            self.__generate_storeCell(output_file)
            self.__generate_exchange_routines(output_file)

    #
    # @todo Sollte man mit Jinja 2 nicht mehr brauchen
    #
    def get_cpp_file_name(self):
        return self.subdirectory + "/" + self.classname + ".cpp"

    def generate(self, overwrite, directory):
        if not os.path.exists(os.path.join(directory, self.subdirectory)):
            os.mkdir(os.path.join(directory, self.subdirectory))

        header_filename = os.path.join(directory, self.subdirectory, self.classname + ".h")
        cpp_filename = os.path.join(directory, self.get_cpp_file_name())

        header_written = write_file(overwrite, self.default_overwrite, header_filename)
        cpp_written = write_file(overwrite, self.default_overwrite, cpp_filename)

        if header_written and cpp_written:
            Logger().log("write " + header_filename + " and " + cpp_filename)
        elif header_written:
            Logger().log("write " + header_filename)
        elif cpp_written:
            Logger().log("write " + cpp_filename)

        self.__generate_header(overwrite, directory)
        self.__generate_implementation(overwrite, cpp_filename)
