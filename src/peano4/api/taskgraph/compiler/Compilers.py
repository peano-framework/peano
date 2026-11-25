# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
"""!

Registry of all task graph compilers, so benchmarks and tests can pick one by name

Every benchmark offers the names as choices of its command line argument
--compiler, i.e. a new task graph compiler has to be added here only, and
every benchmark offers it.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
parser.add_argument("--compiler", choices=list(peano4.api.taskgraph.compiler.Compilers.COMPILERS), default="fusion")
...
project.compiler = peano4.api.taskgraph.compiler.Compilers.COMPILERS[args.compiler]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from . import CellTaskProducerConsumer, FuseSteps, Sequential, TaskGraphAssembly

COMPILERS = {
    "sequential": Sequential.translate,
    "fusion": FuseSteps.translate,
    "producer-consumer": CellTaskProducerConsumer.translate_with_sequential,
    "producer-consumer-fusion": CellTaskProducerConsumer.translate_with_fusion,
    "producer-consumer-chains": CellTaskProducerConsumer.translate_with_chains_and_fusion,
    "producer-consumer-chains-sequential": CellTaskProducerConsumer.translate_with_chains_and_sequential,
    "assembly": TaskGraphAssembly.translate_with_sequential,
    "assembly-fusion": TaskGraphAssembly.translate_with_fusion,
    "assembly-concatenation": TaskGraphAssembly.translate_with_concatenation_and_sequential,
    "assembly-concatenation-fusion": TaskGraphAssembly.translate_with_concatenation_and_fusion,
}
"""! Name of a task graph compiler to the compiler function """
