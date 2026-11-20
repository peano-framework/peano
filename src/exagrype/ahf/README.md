# Apparent Horizon Finder Build Notes

This folder contains ExaGRyPE's apparent-horizon-finder (AHF) code. At the moment it is a standalone CMake postprocessing tool for VTK/VTU snapshots, not part of the ExaGRyPE library; on-the-fly use inside a simulation is planned (see below).

The source split is:

- `peano-input.*`: on-the-fly input from Peano data structures
- `file-input.*`: offline input from files, currently intended for VTK/VTU data
- `output.*`: shared output helpers for both on-the-fly and offline modes
- `interpolation.*`, `assemble-matrix.*`, `iteration.*`: shared AHF numerical pipeline pieces
- `ahf-postprocess.cpp`: local offline command-line frontend
- `ahf-postprocess-options.*`: its argument parsing and search set-up, kept out of the executable so the tests can run them without VTK

## Cartesian interpolation modes

`CartesianToSurfaceOptions::method` keeps the existing interpolation choices
and also provides an opt-in smooth-support choice:

- `WeightedPolynomial` selects exactly `numberOfNeighbours` nearest cell
  centres and applies the existing Gaussian moving-least-squares weights.
- `RadiusWeightedPolynomial` selects every cell centre at distance strictly
  less than `supportRadius` and applies the compact Wendland C2 weight
  `(1-q)^4(4q+1)` for `q=distance/supportRadius < 1`. The weight is zero at and
  outside the support boundary.

The radius-supported mode is useful when the interpolated target moves during
a Newton iteration: support membership can change while a cell's contribution
vanishes smoothly at the boundary. It requires a finite positive
`supportRadius` and enough geometrically independent cells inside that radius
for the requested `polynomialDegree`. The fixed-count method remains unchanged
and is still the default weighted-polynomial behaviour.

## Local Postprocessing Build

Build from inside this folder:

```bash
cd src/exagrype/ahf
cmake -S . -B build
cmake --build build --target ahf-postprocess
```

It can also be built as part of Peano's CMake build, by adding `-DENABLE_EXAGRYPE=ON -DENABLE_CCZ4_AHF_POSTPROCESSOR=ON` (and `-DENABLE_CCZ4_AHF_TESTS=ON` for the tests) to the usual ExaGRyPE configuration.

Run the postprocessor on a snapshot (see [Reading ExaGRyPE Snapshots](#reading-exagrype-snapshots) for converting a patch file):

```bash
./build/ahf-postprocess --input snapshot.vtu --radius 1.0,0.7,0.5 --output horizon
```

It starts from a sphere of each radius around the centre (`--centre x,y,z`, the origin by default) in turn, and stops at the first search that converges to an acceptable surface. It writes the summary `horizon.txt` and, if it found a horizon, the surface `horizon.vtp`. For the horizon it prints, and the summary records, the coordinate radii about the centre, the area $A$, the areal radius $\sqrt{A/4\pi}$ and the irreducible mass $\sqrt{A/16\pi}$; the area uses the physical metric interpolated onto the surface. In the summary, `meanRadius` is the plain average of the radius over the solver's collocation points, which do not sample the sphere evenly, and `meanCoordinateRadius` is its average over solid angle. It selects the fields by component name unless `--mapping exagrype` or `--mapping legacy` is given, and interpolates with radius-weighted cubic polynomials within three times the finest cell spacing unless `--support-radius` is given. `--help` lists every option. The exit status is 0 if a horizon was found, 1 if not, and 2 on an error.

Newton drives the expansion $\Theta$ to zero only through its spectral projection at the collocation points, so from a starting sphere that is too large it can converge to a surface that aliasing makes look like a horizon. Every search (`FindApparentHorizon`, `FindApparentHorizons`) therefore also evaluates $\Theta$ on a grid twice as fine in each direction and rejects a converged surface whose relative expansion error $\max|\Theta|\,R/2$, with $R$ the areal radius, exceeds `SpectralNewtonOptions::maxRelativeExpansionError` (0.3), then tries its next attempt. On analytic horizons with `lMax` of at least 4 the error stayed below 0.13, and on aliased surfaces above 0.55. The summaries record it as `relativeExpansionError` and each rejected attempt's reason as `rejection`. The common-horizon validation (`ValidateCommonHorizon`) applies the same bound (`relativeExpansionTolerance`); its absolute bound on $\Theta$ at the collocation points, `pointwiseExpansionTolerance`, is off unless set, since grid data reaches $10^{-4}$ to $10^{-2}$ there even on correct horizons.

Near the separation where a common horizon disappears it becomes strongly distorted, and at too low an `lMax` the same bound also rejects it as unresolved; raise `lMax` there.

For several black holes, `FindApparentHorizons` with `AHCommonHorizonSearchOptions` first searches each individual horizon, then plans common searches. A pair is planned when the gap between its individual horizons is at most `searchMargin` plus `searchMarginPerMass` (0.5) times the sum of the two masses given. A common horizon forms while the individual ones are still apart: on equal-mass Brill–Lindquist data it exists up to a separation of $0.766\,M$ ($1.532$ for two unit masses, as published), where the individual horizons are still $0.414\,M$ apart. This margin was calibrated on data at $t = 0$; on evolved data, whose coordinates differ, it has not been measured. An individual search started too large can converge to the common horizon, which passes every other check, so each individual search rejects a surface that encloses another black hole's `physicalPosition` and tries its next attempt (`individualsExcludeOtherBlackHoles`, on by default). `FindBinaryMergerHorizons` plans its common search with the same margin, keeps each component from enclosing the other, and validates its common result (`commonValidation`).

For several black holes, give the postprocessor each puncture's position and mass instead of `--radius`:

```bash
./build/ahf-postprocess --input snapshot.vtu --puncture 0.25,0,0:0.5 --puncture -0.25,0,0:0.5 --output binary
```

This runs `FindApparentHorizons` with black holes `bh0`, `bh1`, ... at the punctures. Each individual search starts from spheres of radius $0.35\,m$, $0.5\,m$ and $0.25\,m$ about its puncture, in that order (`--radius-per-mass` changes the factors): an isolated puncture of bare mass $m$ has its horizon at coordinate radius $m/2$, and a companion makes it smaller. It writes the catalogue summary `binary.txt` and, for each horizon, `binary-individual-<i>-<id>.txt` or `binary-common-<i>-<id>.txt`, with a `.vtp` surface next to it once the horizon is found (a common one once it is fully validated). It prints each puncture's horizon, the gap between each pair of individual horizons and whether a common search was planned, and each common horizon with its validation. The exit status is 0 if every puncture's own horizon was found, 1 if not, and 2 on an error. Common horizons do not count, since a common search also runs a little beyond the separation where one exists. With `--puncture` the default `--lmax` is 8 rather than 4, since a common horizon can be strongly distorted: for equal masses at a separation of $0.7\,M$, close to where the common horizon disappears, `lMax` 4 gave a relative expansion error of 0.126 and an irreducible mass $10^{-4}$ low, against 0.014 and $10^{-6}$ at 8. That run took 54 s instead of 14 s on a snapshot of four million cells; pass `--lmax 4` for a quicker, rougher search. Once the individual horizons have gone, search for the common horizon alone with `--centre` at the centre of mass and `--radius`.

The local postprocessing build uses `file-input.cpp`, so CMake has to find the C++ VTK package. Without VTK, the core and the tests that do not read `.vtu` files are still built, but `ahf-postprocess` and the `ahf-io`, `ahf-multi-horizon-dataset` and `ahf-sample-t0-user-workflow` tests are skipped. If VTK is installed but not found, pass either `VTK_DIR` or `CMAKE_PREFIX_PATH`:

```bash
cmake -S . -B build -DVTK_DIR=/path/to/vtk/lib/cmake/vtk-*
```

or:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/vtk
```

## Reading ExaGRyPE Snapshots

`exagrype.api` runs write Peano patch files, not `.vtu`. Convert them with Peano's renderer from the repository root; it needs Python with the `vtk` module:

```bash
python3 src/tarch/plotter/api/render.py --filter-fine-grid --dir vtu solution-CCZ4.peano-patch-file
```

This writes one `.vtu` per snapshot into `vtu/`. Each cell holds all of the solver's unknowns in one cell-data array, and the renderer names each component after the solver's variables (`G[0]`, ..., `K[5]`, `traceK`, `phi`, ...). `ReadGridFromFile(file, grid)` finds the 14 fields the finder needs by these names, wherever they sit in the array.

A file without component names needs an explicit `SourceVariableMapping` (`ahf-data.h`):

- `ExaGRyPESourceMapping()` for `exagrype.api` output (`FOCCZ4Solver`, `FOSOCCZ4Solver`): the conformal metric `G` at 0-5, the conformal traceless extrinsic curvature `K` at 6-11, trace K at 23 and $\phi$ at 24.
- `LegacyCCZ4SourceMapping()` for output of the legacy `applications/exahype2/ccz4/ccz4.py`, which stores trace K at 53 and $\phi$ at 54. The GitLab-era snapshots that the integration tests read have this layout.

If the file does name its components, a mapping that contradicts the names is rejected rather than reading the wrong fields.

## Splitting a Large Snapshot (optional)

A big `.vtu` snapshot is read in one go, which needs a lot of memory. To read it
more cheaply, first split it into smaller pieces with the helper script:

```bash
python3 repartition_vtu.py snapshot.vtu pieces/snapshot.pvtu -k 8
```

This produces `pieces/snapshot.pvtu` (a summary) plus its piece files. Give that `.pvtu` to the finder instead of the original `.vtu`: it then reads one piece at a time and uses far less memory. You only need to do this once per snapshot. `ahf-io-tests` test both `vtu` and `pvtu` (under a pieces directory) approach.
Requires Python with the `vtk` module. 

## Local Postprocessing Build With Tests

Tests are optional and disabled by default. Enable them with:

```bash
cd src/exagrype/ahf
cmake -S . -B build -DENABLE_CCZ4_AHF_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

The test executables are:

```text
ahf-io-tests
ahf-interpolation-tests
ahf-assemble-matrix-tests
ahf-iteration-tests
ahf-harmonics-tests
ahf-spectral-solver-tests
ahf-spacetimes-tests
ahf-analytic-horizon-tests
ahf-brill-lindquist-tests
```

They can also be run directly:

```bash
./build/ahf-io-tests
./build/ahf-interpolation-tests
./build/ahf-assemble-matrix-tests
./build/ahf-iteration-tests
```

## On-The-Fly Use (planned)

On-the-fly use inside an ExaGRyPE simulation is not available yet. The `--with-ahf` flag of the old `applications/exahype2/ccz4/ccz4.py` script has been removed, as ExaGRyPE runs are now built through `exagrype.api` (see `benchmarks/exagrype`). The plan is to compile the core into the ExaGRyPE library and add an `exagrype.api` action set that switches the finder on with `-DIncludeApparentHorizonFinder`. The Peano-side input, `peano-input.*` (`ReadGridFromPeano`), is still a stub. The on-the-fly build will not compile `file-input.cpp`, so it will not require VTK.

## Cleaning

For a CMake build directory:

```bash
cmake --build build --target clean
```

To remove the local CMake build completely:

```bash
rm -rf build
```
