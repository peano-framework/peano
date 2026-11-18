# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
from peano4.api.datamodel.ModelToDataRepository import ModelToDataRepository
from peano4.api.datamodel.DoF import DoFAssociation


class Model(object):
    def __init__(self, namespace, peano_project, subdirectory=""):
        self.cell_data = []
        self.face_data = []
        self.vertex_data = []
        self.global_data = []
        self.namespaces = []
        self.subdirectories = []
        self.temporary_subdirectories = peano_project.temporary_subdirectories
        self.temporary_subdirectory = peano_project.temporary_subdirectory
        self.namespace = namespace
        if namespace:
            if not namespace in self.namespaces:
                self.namespaces.append(namespace)
                self._subdirectory = subdirectory
                self.subdirectories.append(self.subdirectory)
        self.generator = ModelToDataRepository(self)

    def __str__(self):
        return (
            "(#cells="
            + str(len(self.cell_data))
            + ",#faces="
            + str(len(self.face_data))
            + ",#vertices="
            + str(len(self.vertex_data))
            + ",#global-objects="
            + str(len(self.global_data))
            + ")"
        )

    def add_cell(self, submodel):
        submodel.configure(self.namespace, DoFAssociation.Cell, self.subdirectory, self.temporary_subdirectory)
        self.cell_data.append(submodel)

    def add_face(self, submodel):
        submodel.configure(self.namespace, DoFAssociation.Face, self.subdirectory, self.temporary_subdirectory)
        self.face_data.append(submodel)

    def add_vertex(self, submodel):
        submodel.configure(self.namespace, DoFAssociation.Vertex, self.subdirectory, self.temporary_subdirectory)
        self.vertex_data.append(submodel)

    def add_global_object(self, submodel):
        submodel.configure(self.namespace, DoFAssociation.Global, self.subdirectory, self.temporary_subdirectory)
        self.global_data.append(submodel)

    def construct_output(self, output):
        for i in self.cell_data:
            i.generator.construct_output(output)
            output.readme.add_entry(i.readme_descriptor)
            output.readme.add_package_description(i.readme_package_descriptor)
        for i in self.face_data:
            i.generator.construct_output(output)
            output.readme.add_entry(i.readme_descriptor)
            output.readme.add_package_description(i.readme_package_descriptor)
        for i in self.vertex_data:
            i.generator.construct_output(output)
            output.readme.add_entry(i.readme_descriptor)
            output.readme.add_package_description(i.readme_package_descriptor)
        for i in self.global_data:
            i.generator.construct_output(output)
            output.readme.add_entry(i.readme_descriptor)
            output.readme.add_package_description(i.readme_package_descriptor)

        self.generator.construct_output(output)
    
    def clear(self):
        self.cell_data = []
        self.face_data = []
        self.vertex_data = []
        self.global_data = []

    @property
    def subdirectory(self):
        subdirectory = self._subdirectory
        if subdirectory:
            subdirectory += "/"
        return subdirectory
