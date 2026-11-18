# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import os
import re
import jinja2

from copy import deepcopy

from .Helper import write_file, WARNING_BANNER

from .Overwrite import Overwrite

class Jinja2TemplatedHeaderImplementationFilePair(object):
  """!
Represents a pair of C++ header and implementation files

The content of these files is represented by jinja2 templates. While
sorting out these templates, the class defines a couple of default
dictionary entries besides the ones manually handed over by the user.
  """

  def __init__(self,
               headfile_template,
               cppfile_template,
               classname,
               namespace,
               subdirectory,
               dictionary,
               default_overwrite=True,
               apply_iteratively=False):
    """
      The template files should be fully qualified
      classname is a string
      namespace is a (possibly empty) list of strings

      cppfile_template can be None
    """
    self.headerfile_template  = headfile_template
    self.cppfile_template     = cppfile_template
    self.classname            = classname
    self.namespace            = namespace
    self.subdirectory         = subdirectory
    self.default_overwrite    = default_overwrite
    self.apply_iteratively    = apply_iteratively

    self.d = deepcopy(dictionary)
    self.d[ "CLASSNAME" ] = classname
    self.d[ "NAMESPACE" ] = []
    for i in namespace:
      self.d["NAMESPACE"].append(i)
    # Eliminate the prepending "::" that can lead to compatibility issues across the solvers
    self.d["FULL_QUALIFIED_NAMESPACE"] = "::".join(namespace)

  def __generate_file(self,overwrite,full_qualified_filename,template_file):
    if template_file!=None and write_file(overwrite,self.default_overwrite,full_qualified_filename):
      import inspect
      print( "{} written by {} (from template {})".format(full_qualified_filename, os.path.basename(inspect.getfile(self.__class__)),  template_file))

      template_loader = jinja2.FileSystemLoader(searchpath=os.path.split(template_file)[0])
      templateEnv = jinja2.Environment(loader=template_loader, undefined=jinja2.DebugUndefined)
      template = templateEnv.get_template( os.path.split(template_file)[1] )

      if self.apply_iteratively:
        old_size                = 0
        rendered_text           = template.render(self.d)
        while "{{" in rendered_text and len(rendered_text)!=old_size:
          old_size      = len(rendered_text)
          template      = jinja2.Template(rendered_text, undefined=jinja2.DebugUndefined)
          rendered_text = template.render( self.d )

      rendered_text = template.render(self.d)
      if self.default_overwrite:
        rendered_text = WARNING_BANNER + rendered_text

      with open( full_qualified_filename, "w" ) as output:
        output.write( rendered_text )

  def generate(self,overwrite,directory):
    subdirectory_path = os.path.normpath( self.subdirectory )
    print (directory)
    print (self.subdirectory)
    if not os.path.exists( subdirectory_path ):
      os.mkdir(subdirectory_path)

    header_filename = os.path.normpath( subdirectory_path + "/" + self.classname + ".h" )
    cpp_filename    = os.path.normpath( subdirectory_path + "/" + self.classname + ".cpp" )

    self.__generate_file(overwrite,header_filename,self.headerfile_template)
    self.__generate_file(overwrite,cpp_filename   ,self.cppfile_template)
