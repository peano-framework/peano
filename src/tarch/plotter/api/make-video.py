"""Turn an ExaHyPE 2 / Peano patch file into a video
"""
import argparse
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")  # no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter

# Peano's Python API lives in <Peano>/src. Default ~/Peano; override with PEANO_ROOT=/path/to/Peano
sys.path.insert(0, os.path.join(os.path.expanduser(os.environ.get("PEANO_ROOT", "~/Peano")), "src"))
import tarch.plotter.api.notebook as viewer  # noqa: E402


def colour_range(file_name, unknown, snapshots):
    """Smallest and largest value of one unknown over all snapshots."""
    low, high = np.inf, -np.inf
    for _, patch_files in snapshots:
        for f in patch_files:
            for _, _, values in viewer.read_patches(f):
                low = min(low, float(np.min(values[:, :, unknown])))
                high = max(high, float(np.max(values[:, :, unknown])))
    return low, high


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patch_file", help="meta file, e.g. 11_tsunami/solution-SWE.peano-patch-file")
    parser.add_argument("--unknown", type=int, default=0, help="index of the unknown to show (default 0)")
    parser.add_argument("--label", default=None, help="name shown in the title, e.g. h or rho")
    parser.add_argument("--out", default=None, help="output file (.mp4 or .gif)")
    parser.add_argument("--fps", type=int, default=10, help="frames per second (default 10)")
    parser.add_argument("--cmap", default="viridis", help="matplotlib colour map (default viridis)")
    args = parser.parse_args()

    snapshots = viewer.read_snapshots(args.patch_file)
    clim = colour_range(args.patch_file, args.unknown, snapshots)
    print(f"{len(snapshots)} snapshots, colour range {clim[0]:.4g} .. {clim[1]:.4g}")

    out = args.out or os.path.splitext(args.patch_file)[0] + ".mp4"
    if out.endswith(".mp4") and shutil.which("ffmpeg") is None:
        out = out[:-4] + ".gif"
        print("ffmpeg not found, writing a gif instead (sudo apt install ffmpeg for mp4)")
    writer = FFMpegWriter(fps=args.fps) if out.endswith(".mp4") else PillowWriter(fps=args.fps)

    fig, ax = plt.subplots(figsize=(6, 5))
    with writer.saving(fig, out, dpi=120):
        for i in range(len(snapshots)):
            fig.clf()
            ax = fig.add_subplot(111)
            viewer.plot_snapshot(args.patch_file, snapshot=i, unknown=args.unknown, clim=clim,
                                 ax=ax, cmap=args.cmap, label=args.label)
            writer.grab_frame()
    print("wrote", out)


if __name__ == "__main__":
    main()
