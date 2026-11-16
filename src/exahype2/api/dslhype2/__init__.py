# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org

#from .Parser import Parser
from .AbstractSyntaxTree import *
from .Expressions import *
from .Types import *
from .Parser import Parser
from .printers.CPPPrinter import CPPPrinter
from .printers.OMPTargetPrinter import OMPTargetPrinter
from .kernels import translate_dslhype_function, generate_namespace, generate_mlir_function_call_bridges


def __getattr__(name):
    """!
    Lazy import of the MLIR printer, so xdsl is only required if it is used.
    """
    if name == "MLIRPrinter":
        from .printers.MLIRPrinter import MLIRPrinter

        return MLIRPrinter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
