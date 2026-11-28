from enum import Enum
from dataclasses import dataclass

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

@dataclass(frozen=True)
class IncludeMLIRFunctions:
    input_file: str

    def __str__(self):
        return f"include-mlir-functions{{mlir-file=\"{self.input_file.split(".")[0]}_clean.mlir\" filename-being-modified=\"$<\"}}"

@dataclass(frozen=True)
class InlineFunctions:
    def __str__(self):
        return "inline"

@dataclass(frozen=True)
class MoveMemRefAllocationsToStart:
    def __str__(self):
        return "move-memref-allocations-to-start"

# This pass is never enabled directly by the user, it exists to facilitate
# other passes
@dataclass(frozen=True)
class LowerAffineToScf:
    def __str__(self):
        return "lower-affine"

MLIRPass = IncludeMLIRFunctions \
         | InlineFunctions \
         | MoveMemRefAllocationsToStart \
         | LowerAffineToScf

# A configuration specified, in order, the type of target device, whether the kernel should
# use the stateless variant if possible, the output language, the memory layout, and a list
# of passes that should be enabled. This final option is ignored if the output language is not
# MLIR.
KernelConfiguration = tuple[Target, bool, OutputLanguage, MemoryLayout, list[MLIRPass]]