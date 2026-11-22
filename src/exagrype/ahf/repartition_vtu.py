#!/usr/bin/env python3
"""Repartition a single-piece .vtu into a K-piece .pvtu for low-memory streaming.

The CCZ4 sample snapshots are written as one big `<Piece>`, which VTK's XML
reader can only load all-at-once. This one-time, offline tool splits such a file
into K disjoint pieces referenced by a `.pvtu` summary. The apparent-horizon
finder can then stream that `.pvtu` piece by piece, so its peak memory is bounded
by one piece rather than the whole grid.

The split itself must load the whole grid once (that is the cost this tool pays
so the finder never has to). It is deliberately separate from the reader: it runs
occasionally, has the opposite memory profile, and is unnecessary for genuinely
multi-piece / domain-decomposed output that Peano already produces.

Usage:
    python3 repartition_vtu.py input.vtu output.pvtu [-k 8] [--no-compress]

The K piece files are written next to output.pvtu as <output-stem>_<p>.vtu.
"""

import argparse
import os
import sys
from xml.sax.saxutils import quoteattr

import vtk

# VTK scalar type name -> VTK XML DataArray type name.
_VTK_TO_XML_TYPE = {
    "double": "Float64",
    "float": "Float32",
    "int": "Int32",
    "long": "Int64",
    "long long": "Int64",
    "idtype": "Int64",
    "short": "Int16",
    "unsigned char": "UInt8",
    "unsigned short": "UInt16",
    "unsigned int": "UInt32",
    "unsigned long": "UInt64",
}


def _xml_type(data_array):
    return _VTK_TO_XML_TYPE.get(data_array.GetDataTypeAsString(), "Float64")


def _component_names(array):
    """ComponentName attributes, as VTK's writer emits them, for the named components."""
    names = ""
    for component in range(array.GetNumberOfComponents()):
        name = array.GetComponentName(component)
        if name:
            names += f' ComponentName{component}={quoteattr(name)}'
    return names


def _parallel_arrays(field_data, indent="      "):
    # The parallel reader builds its arrays from these declarations, so the
    # component names have to be here as well as in the pieces: the finder
    # selects its fields by them.
    lines = []
    for i in range(field_data.GetNumberOfArrays()):
        array = field_data.GetArray(i)
        if array is None:
            continue
        lines.append(
            f'{indent}<PDataArray type="{_xml_type(array)}" '
            f'Name="{array.GetName()}" '
            f'NumberOfComponents="{array.GetNumberOfComponents()}"'
            f'{_component_names(array)}/>'
        )
    return "\n".join(lines)


def _write_summary(pvtu_path, grid, piece_sources):
    """Write the .pvtu summary that ties the piece files together."""
    points = grid.GetPoints().GetData()
    scalars = grid.GetCellData().GetScalars()
    scalars_attr = f' Scalars="{scalars.GetName()}"' if scalars is not None else ""
    piece_lines = "\n".join(
        f'    <Piece Source="{source}"/>' for source in piece_sources
    )

    xml = f"""<?xml version="1.0"?>
<VTKFile type="PUnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">
  <PUnstructuredGrid GhostLevel="0">
    <PPoints>
      <PDataArray type="{_xml_type(points)}" Name="Points" NumberOfComponents="3"/>
    </PPoints>
    <PPointData>
{_parallel_arrays(grid.GetPointData())}
    </PPointData>
    <PCellData{scalars_attr}>
{_parallel_arrays(grid.GetCellData())}
    </PCellData>
{piece_lines}
  </PUnstructuredGrid>
</VTKFile>
"""
    with open(pvtu_path, "w") as summary:
        summary.write(xml)


def repartition(input_path, output_path, pieces, compress=True):
    if pieces < 1:
        raise ValueError("number of pieces must be >= 1")

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(input_path)
    if not reader.CanReadFile(input_path):
        raise RuntimeError(f"cannot read {input_path}")
    reader.Update()
    grid = reader.GetOutput()

    number_of_cells = grid.GetNumberOfCells()
    if number_of_cells == 0:
        raise RuntimeError(f"{input_path} has no cells")
    pieces = min(pieces, number_of_cells)

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(output_path))[0]

    sources = []
    for p in range(pieces):
        lo = number_of_cells * p // pieces
        hi = number_of_cells * (p + 1) // pieces

        extract = vtk.vtkExtractCells()
        extract.SetInputData(grid)
        extract.AddCellRange(lo, hi - 1)  # AddCellRange is inclusive on both ends
        extract.Update()

        piece = extract.GetOutput()
        # vtkExtractCells injects bookkeeping arrays that are not in the summary
        # and that we do not want streamed into GridData; drop them.
        for name in ("vtkOriginalCellIds",):
            if piece.GetCellData().HasArray(name):
                piece.GetCellData().RemoveArray(name)
        for name in ("vtkOriginalPointIds",):
            if piece.GetPointData().HasArray(name):
                piece.GetPointData().RemoveArray(name)

        source = f"{stem}_{p}.vtu"
        sources.append(source)
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputData(piece)
        writer.SetFileName(os.path.join(output_dir, source))
        # Inline base64 binary (the source snapshot's format): pure-ASCII XML that
        # is portable across VTK versions, unlike raw-appended binary.
        writer.SetDataModeToBinary()
        if compress:
            writer.SetCompressorTypeToZLib()
        else:
            writer.SetCompressorTypeToNone()
        writer.Write()
        print(f"piece {p}: cells [{lo}, {hi}) -> {source}", flush=True)

    _write_summary(output_path, grid, sources)
    print(f"wrote {output_path} with {pieces} pieces ({number_of_cells} cells total)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="single-piece .vtu snapshot")
    parser.add_argument("output", help="output .pvtu summary path")
    parser.add_argument("-k", "--pieces", type=int, default=8,
                        help="number of pieces to split into (default: 8)")
    parser.add_argument("--no-compress", action="store_true",
                        help="write uncompressed piece files")
    args = parser.parse_args(argv)

    repartition(args.input, args.output, args.pieces, compress=not args.no_compress)
    return 0


if __name__ == "__main__":
    sys.exit(main())
