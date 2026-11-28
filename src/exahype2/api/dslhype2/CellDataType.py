from exahype2.api import Solver, Target, OutputLanguage, MemoryLayout, PDETerm
import exahype2.api.dslhype2 as dslhype

def construct_cell_data_type(dimensions: int, target: Target, output_language: OutputLanguage, memory_layout: MemoryLayout):
    if output_language == OutputLanguage.MLIR:
        if memory_layout == MemoryLayout.AoS_CellDataFlattened_TemporariesNested or memory_layout == MemoryLayout.AoS_AllFlattened:
            cell_data_type = dslhype.TStruct("::exahype2::MLIRCellDataFlattened&")
        else:
            cell_data_type = dslhype.TStruct("::exahype2::MLIRCellDataNested&")
    elif target == Target.Accelerator:
        cell_data_type = dslhype.TStruct("::exahype2::CopyCellDataGPU<double, double>&")
    elif memory_layout == MemoryLayout.AoS_CellDataFlattened_TemporariesNested or memory_layout == MemoryLayout.AoS_AllFlattened:
        cell_data_type = dslhype.TStruct("::exahype2::CellDataFlattened<double, double>&")
    else:
        cell_data_type = dslhype.TStruct("::exahype2::CellData<double, double>&")

    if memory_layout == MemoryLayout.AoS_AllNested or memory_layout == MemoryLayout.AoS_CellDataNested_TemporariesFlattened:
        cell_data_type.add_field("QIn", dslhype.TDataBlock(2, dslhype.TFloat(), flatten_array=False))
        cell_data_type.add_field("QOut", dslhype.TDataBlock(2, dslhype.TFloat(), flatten_array=False))
    else:
        cell_data_type.add_field("QIn", dslhype.TDataBlock(2, dslhype.TFloat(), flatten_array=True))
        cell_data_type.add_field("QOut", dslhype.TDataBlock(2, dslhype.TFloat(), flatten_array=True))

    cell_data_type.add_field("cellCentre", dslhype.TDataBlock(1, dslhype.TVector(dslhype.TFloat(), dimensions), flatten_array=False))
    cell_data_type.add_field("cellSize", dslhype.TDataBlock(1, dslhype.TVector(dslhype.TFloat(), dimensions), flatten_array=False))
    cell_data_type.add_field("t", dslhype.TArray(dslhype.TFloat(), 1))
    cell_data_type.add_field("dt", dslhype.TArray(dslhype.TFloat(), 1))
    cell_data_type.add_field("maxEigenvalue", dslhype.TArray(dslhype.TFloat(), 1))
    cell_data_type.add_field("numberOfCells", dslhype.TInteger())

    return cell_data_type