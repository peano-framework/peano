# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import os
import re
import jinja2

from copy import deepcopy

from .Helper import write_file, WARNING_BANNER

from .Overwrite import Overwrite

class Jinja2TemplatedHeaderFile(object):
  def __init__(self,headerfile_template,classname,namespace,subdirectory,dictionary,default_overwrite=True,apply_iteratively=False,announce=True):
    """
      The template files should be fully qualified

      classname is a string
      namespace is a (possibly empty) list of strings
      dictionary a series of mappings (cmp. Jinja2 convention)
      announce controls whether generate() prints a "write <file>" line.
        Set it to False if the caller reports the write itself (e.g. as
        part of a combined header/implementation message).
    """
    self.headerfile_template  = headerfile_template
    self.classname            = classname
    self.namespace            = namespace
    self.subdirectory         = subdirectory
    self.default_overwrite    = default_overwrite
    self.apply_iteratively    = apply_iteratively
    self.announce             = announce

    self.d = deepcopy(dictionary)
    self.d[ "CLASSNAME" ] = classname
    self.d[ "NAMESPACE" ] = []
    for i in namespace:
      self.d["NAMESPACE"].append(i)
    # Eliminate the prepending "::" that can lead to compatibility issues across the solvers
    self.d["FULL_QUALIFIED_NAMESPACE"] = "::".join(namespace)

  def __generate_file(self,overwrite,full_qualified_filename,template_file):
    """
      template_file: string
         Usually, this is an absolute file name.
    """
    if template_file!=None and write_file(overwrite,self.default_overwrite,full_qualified_filename):
      if self.announce:
        print( "write " + full_qualified_filename )

      template_loader = jinja2.FileSystemLoader(searchpath=os.path.split(template_file)[0])
      templateEnv = jinja2.Environment(loader=template_loader, undefined=jinja2.DebugUndefined)
      template = templateEnv.get_template( os.path.split(template_file)[1] )

      if self.apply_iteratively:
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
    if not os.path.exists( os.path.join(directory, self.subdirectory) ):
      os.mkdir(os.path.join(directory, self.subdirectory))

    header_filename = os.path.join(directory, self.subdirectory, self.classname + ".h")

    self.__generate_file(overwrite,header_filename,self.headerfile_template)
