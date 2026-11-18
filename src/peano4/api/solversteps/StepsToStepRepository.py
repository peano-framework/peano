# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org


import peano4.api.output.TemplatedHeaderImplementationFilePair
import os


class StepsToStepRepository(object):
  def __init__(self,steps):
    self.steps = steps
    self.d     = {}
    self.d["FORWARD_DECLARATIONS"] = ""
    self.d["ENUM_VARIANTS"]        = ""
    self.d["STRING_RETURN_CASE"]   = ""
    self.d["INT_RETURN_CASE"]   = ""
    self.d["STEP_RETURN_CASE"]   = ""

  def __get_full_namespace(self, step):
    return step.namespace + [ "repositories" ] # self.steps._namespace

  def __get_class_name(self):
    return "StepRepository"

  def construct_output(self,output):
    # print("steps._project.namespaces = ", self.steps._project.namespaces)
    for i in range(len(self.steps._project.namespaces)):
      namespace = self.steps._project.namespaces[i]
      subdirectory = self.steps._project.subdirectories[i]
      temporary_subdirectory = self.steps._project.temporary_subdirectories[i]
      self.d["FORWARD_DECLARATIONS"] = ""
      self.d["ENUM_VARIANTS"]        = ""
      self.d["STRING_RETURN_CASE"]   = ""
      self.d["INT_RETURN_CASE"]   = ""
      self.d["STEP_RETURN_CASE"]   = ""
      for step in self.steps._steps:
        if step.namespace == namespace:
          self.d["FORWARD_DECLARATIONS"] += "  class " + step.name + ";\n"
          self.d["ENUM_VARIANTS"]        += ", " + step.name 
          self.d["STRING_RETURN_CASE"]   += "    case Steps::" + step.name + ": return \"" + step.name + "\";\n"
          self.d["INT_RETURN_CASE"]      += "    case Steps::" + step.name + ": return " + str(self.steps._steps.index(step)+1) + ";\n"
          self.d["STEP_RETURN_CASE"]     += "    case " + str(self.steps._steps.index(step)+1) + ": return Steps::" + step.name + ";\n"

    #     + " = peano4::parallel::Node::UndefProgramStep + " + 
      # print("steps.subdirectory = ", subdirectory, ", steps.namespace = ", namespace)

      class_name = self.__get_class_name()
      output.makefile.add_cpp_file( os.path.join(subdirectory, "repositories", class_name + ".cpp"), generated=True )
      templatefile_prefix = os.path.realpath(__file__).replace( ".pyc", "" ).replace( ".py", "" )
      local_d = self.d.copy()
      generated_files = peano4.api.output.TemplatedHeaderImplementationFilePair(
        templatefile_prefix+".h.template",
        templatefile_prefix+".cpp.template",
        class_name, 
        namespace + ["repositories"],
        os.path.join(temporary_subdirectory, "repositories"),
        local_d,
        True)
      output.add(generated_files)
    pass
