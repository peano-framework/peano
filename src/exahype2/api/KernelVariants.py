from enum import Enum

"""!

Target specifies the final architecture that will be used.
TODO: This file will later include variants to control which MLIR passes are used, and which backends

"""
class Target(Enum):
    Serial = 0
    Multicore = 1
    Accelerator = 2

class OutputLanguage(Enum):
    CPP = 0
    MLIR = 1

class MemoryLayout(Enum):
    AoS_AllNested = 0
    AoS_CellDataFlattened_TemporariesNested = 1
    AoS_CellDataNested_TemporariesFlattened = 2
    AoS_AllFlattened = 3