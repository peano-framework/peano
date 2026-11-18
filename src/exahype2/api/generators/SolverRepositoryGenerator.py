# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import peano4.api
import peano4.api.output.Jinja2TemplateStringHeaderImplementationFilePair
import os

from ..Project import Project


class SolverRepositoryGenerator(Project.Generator):
    """!

    Generates repositories/SolverRepository.h/.cpp: one extern instance per
    solver registered with the project, plus getMinTimeStamp()/
    getMaxTimeStamp() aggregating over all of them.

    """

    TEMPLATE_HEADER_FILE = """
// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

{% for item in SOLVERS -%}
#include "{{ item[0] }}.h"
{# Comment to ensure that jinja2 adds a new line#}
{% endfor %}


{% for item in NAMESPACE -%}
namespace {{ item }} {
{%- endfor %}

  {% for item in SOLVERS -%}
  extern {{ item[0] }}  {{ item[1] }};
  {# Comment to ensure that jinja2 adds a new line#}
  {%- endfor %}

  /**
   * Return the global min time stamp over all solvers.
   */
  double getMinTimeStamp();

  /**
   * Return the global max time stamp over all solvers.
   */
  double getMaxTimeStamp();

  /**
   * Return the global min time step size over all solvers.
   */
  double getMinTimeStepSize();

  /**
   * Return the global max time step size over all solvers.
   */
  double getMaxTimeStepSize();

  /**
   * Does any solver want to plot? Only solvers with a plotter are asked.
   * Each plotter then decides itself within the plot phase whether it
   * plots (@see exahype2.api.Solver.Plotter).
   */
  bool wantsToPlot();

{% for item in NAMESPACE -%}
}
{%- endfor %}
"""

    TEMPLATE_CPP_FILE = """
// This file is part of the ExaHyPE2 project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#include "SolverRepository.h"

#include <algorithm>
#include <limits>

{% for item in NAMESPACE -%}
namespace {{ item }} {
{%- endfor %}

  {% for item in SOLVERS -%}
  {{ item[0] }}  {{ item[1] }};
  {# Comment to ensure that jinja2 adds a new line#}
  {%- endfor %}


  double getMinTimeStamp() {
    double result = std::numeric_limits<double>::max();
    {% for item in SOLVERS -%}
    result = std::min( result, {{ item[1] }}.getMinTimeStamp() );
    {# Comment to ensure that jinja2 adds a new line#}
    {%- endfor %}
    return result;
  }


  double getMaxTimeStamp() {
    double result = 0.0;
    {% for item in SOLVERS -%}
    result = std::max( result, {{ item[1] }}.getMaxTimeStamp() );
    {# Comment to ensure that jinja2 adds a new line#}
    {%- endfor %}
    return result;
  }


  double getMinTimeStepSize() {
    double result = std::numeric_limits<double>::max();
    {% for item in SOLVERS -%}
    result = std::min( result, {{ item[1] }}.getMinTimeStepSize() );
    {# Comment to ensure that jinja2 adds a new line#}
    {%- endfor %}
    return result;
  }


  double getMaxTimeStepSize() {
    double result = 0.0;
    {% for item in SOLVERS -%}
    result = std::max( result, {{ item[1] }}.getMaxTimeStepSize() );
    {# Comment to ensure that jinja2 adds a new line#}
    {%- endfor %}
    return result;
  }


  bool wantsToPlot() {
    bool result = false;
    {% for item in PLOTTING_SOLVERS -%}
    result |= {{ item }}.wantsToPlot();
    {# Comment to ensure that jinja2 adds a new line#}
    {%- endfor %}
    return result;
  }

{% for item in NAMESPACE -%}
}
{%- endfor %}
"""

    def __init__(self, project: Project):
        super().__init__(project)

    def __set_solver_repository_dict(self):
        self.solverRepositoryDictionary = {
            "SOLVERS":                  [],
            "PLOTTING_SOLVERS":         [],
            "LOAD_BALANCER":            self.project._load_balancer_name,
            "LOAD_BALANCER_ARGUMENTS":  self.project._load_balancer_arguments,
            "LOG_FILTER_FILE":          self.project._log_filter_file,
            "NUMBER_OF_THREADS":        self.project._number_of_threads,
            "MULTICORE_ORCHESTRATION":  self.project._multicore_orchestration,
            "TIMEOUT":                  self.project._timeout,
            "GPUS":                     self.project._gpus,
            "ADDITIONAL_INCLUDES":      self.project._additional_includes,
        }

    def construct_output(self, peano_project):
        self.__set_solver_repository_dict()

        for solver in self.project._solvers:
            self.solverRepositoryDictionary["SOLVERS"].append(
                (solver.get_name_of_solver_class(), solver.get_name_of_global_instance())
            )
            if solver.plotters:
                self.solverRepositoryDictionary["PLOTTING_SOLVERS"].append(
                    solver.get_name_of_global_instance()
                )

        generated_solver_files = (
            peano4.api.output.Jinja2TemplateStringHeaderImplementationFilePair(
                self.TEMPLATE_HEADER_FILE,
                self.TEMPLATE_CPP_FILE,
                "SolverRepository",
                peano_project.namespace + ["repositories"],
                os.path.join(peano_project.temporary_subdirectory, "repositories"),
                self.solverRepositoryDictionary,
                True,
            )
        )

        peano_project.output.add(generated_solver_files)
        peano_project.output.makefile.add_cpp_file(
            peano_project.subdirectory + "repositories/SolverRepository.cpp", generated=True
        )
