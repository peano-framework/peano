import numpy as np
import pytest

from tarch.plotter.api.input.PatchFileParser import PatchFileParser


# ---------------------------------------------------------------------------
# Writes files like PeanoBinaryPatchFileWriter and PeanoTextPatchFileWriter do.
# The quantities of the binary file are raw doubles between the "begin ...-values"
# and "end ...-values" line.
# ---------------------------------------------------------------------------

DOF = 2
UNKNOWNS = 3
DIMENSIONS = 2

# Raw bytes which look like a line break and like the end of a patch. A parser
# which reads the quantities line by line trips over them.
EVIL = np.frombuffer(b"\nend pat\n", dtype=np.uint8)[:8].tobytes()


def values_of_patch(patch_id, name_offset=0.0, evil=False):
    result = np.array(
        [
            100 * patch_id + 10 * k + u + name_offset
            for k in range(DOF**DIMENSIONS)
            for u in range(UNKNOWNS)
        ]
    )
    if evil:
        # first entry replaced by the bytes in EVIL, as write_file() does
        result[0] = np.frombuffer(EVIL, dtype=np.float64)[0]
    return result


def write_file(path, binary, patches=2, datasets=("Q",), evil=False):
    header = ["# ", "# Peano patch file ", "# Version 0.2 ", "# "]
    header += ["format Binary" if binary else "format ASCII"]
    header += [f"dimensions {DIMENSIONS}", "", ""]
    out = ("\n".join(header) + "\n").encode()

    for name in datasets:
        out += (
            f'begin cell-metadata "{name}"\n'
            f"  number-of-unknowns {UNKNOWNS}\n"
            f"  number-of-dofs-per-axis {DOF}\n"
            f"  description {name}\n"
            f"end cell-metadata\n\n"
        ).encode()

    for patch_id in range(patches):
        out += (
            f"begin patch\n  offset {0.5 * patch_id} 0.0\n  size 0.5 0.5\n"
        ).encode()
        for index, name in enumerate(datasets):
            values = values_of_patch(patch_id, 1000.0 * index)
            out += f'  begin cell-values "{name}"\n'.encode()
            if binary:
                raw = values.astype(np.float64).tobytes()
                if evil:
                    raw = EVIL + raw[8:]
                out += raw + b"\n"
            else:
                out += ("    " + " ".join(repr(float(v)) for v in values) + "\n").encode()
            out += b"  end cell-values\n"
        out += b"end patch\n\n"
    path.write_bytes(out)


def parse(path, identifier="Q", patch_filter=None):
    parser = PatchFileParser(str(path), identifier, 0, patch_filter=patch_filter)
    parser.parse_file()
    assert parser.parsed
    return parser


# ---------------------------------------------------------------------------


@pytest.mark.parametrize("binary", [False, True])
def test_text_and_binary_files_yield_the_same_patches(tmp_path, binary):
    write_file(tmp_path / "f.peano-patch-file", binary)
    parser = parse(tmp_path / "f.peano-patch-file")

    assert parser.file_format == ("Binary" if binary else "ASCII")
    patches = parser.unknown_attributes["Q"].cell_data
    assert len(patches) == 2
    for patch_id, patch in enumerate(patches):
        assert patch.offset == (0.5 * patch_id, 0.0)
        assert np.array_equal(patch.values, values_of_patch(patch_id))


def test_binary_quantities_may_hold_any_bytes(tmp_path):
    write_file(tmp_path / "f.peano-patch-file", True, evil=True)
    parser = parse(tmp_path / "f.peano-patch-file")

    patches = parser.unknown_attributes["Q"].cell_data
    assert len(patches) == 2
    assert patches[0].values.tobytes()[:8] == EVIL
    assert patches[1].values.tobytes() == values_of_patch(1, evil=True).tobytes()


@pytest.mark.parametrize("binary", [False, True])
def test_other_datasets_are_stepped_over(tmp_path, binary):
    write_file(
        tmp_path / "f.peano-patch-file",
        binary,
        datasets=("A", "B", "C"),
        evil=binary,
    )
    parser = parse(tmp_path / "f.peano-patch-file", identifier="B")

    assert list(parser.unknown_attributes.keys()) == ["B"]
    patches = parser.unknown_attributes["B"].cell_data
    assert len(patches) == 2
    assert (
        patches[1].values.tobytes()
        == values_of_patch(1, 1000.0, evil=binary).tobytes()
    )


@pytest.mark.parametrize("binary", [False, True])
def test_patch_filter_skips_patches(tmp_path, binary):
    write_file(tmp_path / "f.peano-patch-file", binary, patches=3, evil=binary)
    parser = parse(
        tmp_path / "f.peano-patch-file",
        patch_filter=lambda offset, size: offset[0] >= 1.0,
    )

    patches = parser.unknown_attributes["Q"].cell_data
    assert len(patches) == 1
    assert patches[0].values.tobytes() == values_of_patch(2, evil=binary).tobytes()
