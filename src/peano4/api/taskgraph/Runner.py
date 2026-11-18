# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import peano4.api.output
import os

TEMPLATE_RUNNER_HEADER_FILE = """
// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "peano4/RunMultipleSteps.h"

{% for item in NAMESPACE -%}
namespace {{ item }} {
{%- endfor %}
  class {{CLASSNAME}};
{% for item in NAMESPACE -%}
}
{%- endfor %}


class {{NAMESPACE | join("::")}}::{{CLASSNAME}}: public peano4::RunMultipleSteps {
  public:
    {{CLASSNAME}}();

    virtual void run() override;
  private:
    static tarch::logging::Log _log;
};
"""

TEMPLATE_RUNNER_CPP_FILE = """
#include "{{CLASSNAME}}.h"

#include "{{SUBDIRECTORY}}repositories/StepRepository.h"
#include "tarch/Assertions.h"
#include "tarch/logging/LogFilter.h"
#include "peano4/parallel/Node.h"
#include "peano4/parallel/SpacetreeSet.h"

{% for observer in OBSERVERS -%}
#include "{{SUBDIRECTORY}}observers/{{observer}}.h"
{% endfor %}

tarch::logging::Log {{NAMESPACE | join("::")}}::{{CLASSNAME}}::_log( "{{NAMESPACE | join("::")}}::{{CLASSNAME}}" );

{{NAMESPACE | join("::")}}::{{CLASSNAME}}::{{CLASSNAME}}():
  peano4::RunMultipleSteps(
    "{{CLASSNAME}}",
    {{PROJECT_NAMESPACE | join("::")}}::{{SUBNAMESPACE}}repositories::StepRepository::toProgramStep(
      {{PROJECT_NAMESPACE | join("::")}}::{{SUBNAMESPACE}}repositories::StepRepository::Steps::{{FIRST_STEP}}
    )
  ) {
}


void {{NAMESPACE | join("::")}}::{{CLASSNAME}}::run() {
  _watch.start();
  {% for step in STEPS %}
  {% if loop.index0==0 %}
  // The caller already had to set Node's program step to this, our very
  // first step, before it could even decide to invoke run() on us in the
  // first place (@see getFirstSubstep()) - so this is redundant, and we
  // only assert on it instead of setting it again.
  assertionEquals(
    peano4::parallel::Node::getInstance().getCurrentProgramStep(),
    {{PROJECT_NAMESPACE | join("::")}}::{{SUBNAMESPACE}}repositories::StepRepository::toProgramStep(
      {{PROJECT_NAMESPACE | join("::")}}::{{SUBNAMESPACE}}repositories::StepRepository::Steps::{{step}}
    )
  );
  {% else %}
  peano4::parallel::Node::getInstance().setNextProgramStep(
    {{PROJECT_NAMESPACE | join("::")}}::{{SUBNAMESPACE}}repositories::StepRepository::toProgramStep(
      {{PROJECT_NAMESPACE | join("::")}}::{{SUBNAMESPACE}}repositories::StepRepository::Steps::{{step}}
    )
  );
  {% endif %}
  do {
    tarch::logging::LogFilter::getInstance().switchProgramPhase("{{step}}");
    {{PROJECT_NAMESPACE | join("::")}}::{{SUBNAMESPACE}}observers::{{OBSERVERS[loop.index0]}}::prepareTraversal();
    {{PROJECT_NAMESPACE | join("::")}}::{{SUBNAMESPACE}}observers::{{OBSERVERS[loop.index0]}} observer;
    peano4::parallel::SpacetreeSet::getInstance().traverse(observer);
    {{PROJECT_NAMESPACE | join("::")}}::{{SUBNAMESPACE}}observers::{{OBSERVERS[loop.index0]}}::unprepareTraversal();
    logDebug( "run()", "will rerun this step: {{REPEAT_CONDITIONS[loop.index0] or "false"}}=" << {{REPEAT_CONDITIONS[loop.index0] or "false"}} );
  }
  while ( {{REPEAT_CONDITIONS[loop.index0] or "false"}} );
  {% endfor %}
  _watch.stop();
  _runtimeMeasurement.setValue(_watch.getCalendarTime());
}
"""


def create_runner(peano_project, identifier, steps):
    """!

    Create the header/implementation file pair that represents one of a
    project's algorithmic phases represented by a sequence of steps.
    It yields one class of the classname identifier which is held in
    the project's runners subdirectory/namespace. This class holds
    exactly one method run() which toggles through the steps one by one.

    Even though the runner seems to accept only a sequence of steps
    (which are members of @ref peano4.api.solversteps), the runner 
    belongs into the task graph repository, as it assumes that each 
    step has a field has_to_repeat. This field is not defined within
    @ref peano4.api.solversteps.Step. It is something graph compilers
    add to the data model. Consult the compiler description in 
    @ref peano4.api.taskgraph.compiler.

    peano_project: peano4.api.Project
      The Peano 4 project to add the generated files to.

    identifier: str
      Same identifier used for the phase's Observers, e.g.
      "SetupSimulation", "TimeStep", "Plot".

    steps: [peano4.api.solversteps.Step]
      The sequence of Observers realising this algorithmic phase, in the
      order compiled from the phase's task graph (@see
      peano4.api.taskgraph.compiler.Sequential.translate). Its first
      element's name is the repositories::StepRepository::Steps enumerator
      the generated runner hands to peano4::RunMultipleSteps's constructor
      to kick the phase off. run() instantiates each step's Observer in
      turn and traverses the spacetree set with it, switching the log
      filter's program phase beforehand, bracketing the traversal itself
      with that Observer's own static prepareTraversal()/
      unprepareTraversal() (@see peano4.api.output.Observer - the spacetree
      set never calls these itself, so the runner is exactly the "up to
      the user" call site its own doc comment asks for), and timing the
      traversal via the inherited _watch/_runtimeMeasurement. "Undef" (and
      no traversals at all) if the phase has no steps, e.g. no solver
      contributed to it, so the generated code still compiles.

      Before every step but the first, run() also calls
      peano4::parallel::Node::getInstance().setNextProgramStep(...) with
      that step's own program step - whoever calls run() in the first
      place already had to set Node's program step to this phase's first
      step beforehand (@see getFirstSubstep(), used exactly that way in
      exahype2.api.generators.DefaultSimulationMain), so run() only
      asserts on it there rather than setting it again; every subsequent
      step within the same phase has nothing else to set it, so run()
      must do so itself. This is what makes
      peano4::parallel::Node::getInstance().getCurrentProgramStep()
      reliable for every step, not just a phase's first - in particular,
      it is what the guard expressions
      peano4.api.taskgraph.compiler.DeriveLoadStoreComputeFlag builds
      actually depend on at runtime.

    """
    step_names = [step.name for step in steps]
    observer_names = [step.get_observer_class_name() for step in steps]
    first_step = step_names[0] if step_names else "Undef"
    repeat_conditions = [step.has_to_repeat for step in steps]

    generated_files = peano4.api.output.Jinja2TemplateStringHeaderImplementationFilePair(
        headerfile_template = TEMPLATE_RUNNER_HEADER_FILE,
        cppfile_template    = TEMPLATE_RUNNER_CPP_FILE,
        classname           = identifier,
        namespace           = peano_project.namespace + ["runners"],
        subdirectory        = os.path.join(peano_project.temporary_subdirectory, "runners"),
        dictionary          = {
            "PROJECT_NAMESPACE": peano_project.namespace,
            "SUBNAMESPACE":      peano_project._subdirectory + "::" if peano_project._subdirectory else "",
            "SUBDIRECTORY":      peano_project.subdirectory,
            "STEPS":             step_names,
            "OBSERVERS":         observer_names,
            "FIRST_STEP":        first_step,
            "REPEAT_CONDITIONS": repeat_conditions,
        },
        default_overwrite   = True,
        apply_iteratively   = False,
    )
    peano_project.output.add(generated_files)
    peano_project.output.makefile.add_cpp_file(
        peano_project.subdirectory + "runners/" + identifier + ".cpp", generated=True
    )
