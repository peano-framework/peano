# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import os
import re

from .Helper import write_file, WARNING_BANNER

from .Overwrite import Overwrite

from ..Logger import Logger

class TemplatedHeaderImplementationFilePair(object):
  def __init__(self,headfile_template,cppfile_template,classname,namespace,subdirectory,dictionary,default_overwrite=True):
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

    self.d = dictionary
    self.d[ "OPEN_NAMESPACE" ]           = ""
    self.d[ "CLOSE_NAMESPACE" ]          = ""
    self.d[ "FULL_NAMESPACE" ]           = ""
    self.d[ "FULL_QUALIFIED_CLASSNAME" ] = ""
    for i in namespace:
      self.d[ "OPEN_NAMESPACE" ]           += "namespace " + i + "{\n"
      self.d[ "CLOSE_NAMESPACE" ]          += "}\n"
      self.d[ "FULL_QUALIFIED_CLASSNAME" ] += i + "::"
      if self.d[ "FULL_NAMESPACE" ] != "":
        self.d[ "FULL_NAMESPACE" ]         += "::"
      self.d[ "FULL_NAMESPACE" ]           += i
    self.d[ "CLASSNAME" ]                 = classname
    self.d[ "FULL_QUALIFIED_CLASSNAME" ] += classname
    self.d[ "INCLUDE_GUARD" ]             = self.d[ "FULL_QUALIFIED_CLASSNAME" ].replace( "::", "_" ).replace( "-", "_" ).upper()

  def generate_file(self,overwrite,full_qualified_filename,template_file):
    if template_file!=None and write_file(overwrite,self.default_overwrite,full_qualified_filename):
      Logger().log( "write " + full_qualified_filename )
      with open( os.path.realpath(template_file), "r" ) as input:
        template = input.read()
        rendered_text = template.format(**self.d)
        if self.default_overwrite:
          rendered_text = WARNING_BANNER + rendered_text
        with open( full_qualified_filename, "w" ) as output:
          output.write( rendered_text )

  def generate(self,overwrite,directory):
    subdirectory_path = os.path.normpath( os.path.join(directory, self.subdirectory) )
    if not os.path.exists( subdirectory_path ):
      os.mkdir(subdirectory_path)

    header_filename = os.path.normpath( os.path.join(subdirectory_path, self.classname + ".h") )
    cpp_filename    = os.path.normpath( os.path.join(subdirectory_path, self.classname + ".cpp") )

    self.generate_file(overwrite,header_filename,self.headerfile_template)
    self.generate_file(overwrite,cpp_filename   ,self.cppfile_template)

