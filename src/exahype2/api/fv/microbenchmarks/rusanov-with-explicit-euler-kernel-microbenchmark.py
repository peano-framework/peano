import os, sys
import argparse
import peano4.api
import exahype2
import exagrype
from exahype2.api.generators.KernelMicrobenchmarkMain import KernelMicrobenchmarkMain
from applications import set_euler_pde_terms

parser = argparse.ArgumentParser(
    description="ExaHyPE 2 - RusanovWithExplicitEuler Microbenchmarking Script"
)
parser.add_argument(
    "--application",
    type=str,
    choices=["euler", "ccz4"],
    required=True,
    help="Which application should be microbenchmarked",
)
parser.add_argument(
    "-d",
    "--dimensions",
    default=3,
    type=int,
    choices=[2,3],
    help="Number of spatial dimensions.",
)
parser.add_argument(
    "-ps",
    "--patch-size",
    default=16,
    type=int,
    help="Number of finite volumes per axis (dimension) per patch.",
)
parser.add_argument(
    "-p",
    "--num-patches",
    nargs="+",
    default=[32, 64, 128, 256, 512, 1024, 2048],
    type=int,
    help="Number of patches to study",
)
parser.add_argument(
    "-samples",
    "--samples",
    default=10,
    type=int,
    help="Number of samples per measurement",
)
parser.add_argument(
    "-a",
    "--accuracy",
    default=0.0,
    type=float,
    help="Floating point accuracy to which the different kernel variants have to match (absolute). Pass in 0 to disable correctness check. Pass in values < 0 to use machine epsilon (default).",
)
parser.add_argument(
    "-cpu",
    "--cpu",
    action="store_true",
    help="Assess host kernels",
)
parser.add_argument(
    "-gpu",
    "--gpu",
    action="store_true",
    help="Assess device kernels",
)
args = parser.parse_args()

project = exahype2.api.Project(
    namespace=["exahype2", "fv", "rusanov", "microbenchmark"],
    project_name="kernel-microbenchmark",
    directory=".",
    executable="kernel-microbenchmark",
)
solver_name = "EulerFVSmooth"

addition_includes = f"""
#include "kernels/{solver_name}_rusanov.h"
"""

function_calls = ""
if args.cpu == True or args.gpu == False:
    function_calls += f"""
    assessKernel(wrapHostKernel<
        cpp::nested::{solver_name}_timeStepWithRusanovStateless<
            Abstract{solver_name}
            >
        >,
        "host, AoSNested, serial, cpp, stateless",
        launchingThreads,
        ::tarch::accelerator::Device::HostDevice,
        localPatches
    );
    """
    function_calls += f"""
    assessKernel(wrapFlattenedHostKernel<
        cpp::cell_data_flattened_temporaries_nested::{solver_name}_timeStepWithRusanovStateless<
            Abstract{solver_name}
            >
        >,
        "host, AoS_CellDataFlattened_TemporariesNested, serial, cpp, stateless",
        launchingThreads,
        ::tarch::accelerator::Device::HostDevice,
        localPatches
    );
    """
    function_calls += f"""
    assessKernel(wrapHostKernel<
        cpp::cell_data_nested_temporaries_flattened::{solver_name}_timeStepWithRusanovStateless<
            Abstract{solver_name}
            >
        >,
        "host, AoS_CellDataNested_TemporariesFlattened, serial, cpp, stateless",
        launchingThreads,
        ::tarch::accelerator::Device::HostDevice,
        localPatches
    );
    """
    function_calls += f"""
    assessKernel(wrapFlattenedHostKernel<
        cpp::flattened::{solver_name}_timeStepWithRusanovStateless<
            Abstract{solver_name}
            >
        >,
        "host, AoSFlattened, serial, cpp, stateless",
        launchingThreads,
        ::tarch::accelerator::Device::HostDevice,
        localPatches
    );
    """
    function_calls += f"""
    assessKernel(wrapMLIRNestedHostKernel<
        mlir::nested::{solver_name}_timeStepWithRusanovStateless
        >,
        "host, AoSNested, serial, mlir, stateless",
        launchingThreads,
        ::tarch::accelerator::Device::HostDevice,
        localPatches
    );
    """
    function_calls += f"""
    assessKernel(wrapMLIRFlattenedHostKernel<
        mlir::cell_data_flattened_temporaries_nested::{solver_name}_timeStepWithRusanovStateless
        >,
        "host, AoS_CellDataFlattened_TemporariesNested, serial, mlir, stateless",
        launchingThreads,
        ::tarch::accelerator::Device::HostDevice,
        localPatches
    );
    """
    function_calls += f"""
    assessKernel(wrapMLIRNestedHostKernel<
        mlir::cell_data_nested_temporaries_flattened::{solver_name}_timeStepWithRusanovStateless
        >,
        "host, AoS_CellDataNested_TemporariesFlattened, serial, mlir, stateless",
        launchingThreads,
        ::tarch::accelerator::Device::HostDevice,
        localPatches
    );
    """
    function_calls += f"""
    assessKernel(wrapMLIRFlattenedHostKernel<
        mlir::flattened::{solver_name}_timeStepWithRusanovStateless
        >,
        "host, AoSFlattened, serial, mlir, stateless",
        launchingThreads,
        ::tarch::accelerator::Device::HostDevice,
        localPatches
    );
    """
    


if args.gpu:
    function_calls += f"""
    assessKernel(wrapDeviceKernel<
        cpp::flattened::{solver_name}_timeStepWithRusanovStateless<
            Abstract{solver_name}
            >
        >,
        "device, AoS, parallel, omp",
        launchingThreads,
        0,
        localPatches
    );
    """

project.mainfile_generator = KernelMicrobenchmarkMain(project, function_calls, addition_includes)

verbosity = peano4.api.LoggerVerbosity.OFF


if args.application == "euler":
    unknowns = 2 + args.dimensions
    thesolver = exahype2.api.Solver(
        solver_name,
        unknowns,
    )
    thesolver.add_user_solver_includes("#include <cmath>\n" "#include <algorithm>\n")

    for stateless in [True, False]:
        set_euler_pde_terms(thesolver, args.dimensions, stateless)
        
    thesolver.add_pde_term(exahype2.api.InitPointWisely(implementation = """
        constexpr double gamma       = 1.4;
        constexpr double p           = 1.0;
        constexpr double u           = 1.0;
        constexpr double domainLeft  = -0.1;
        constexpr double domainWidth = 0.2;
        const double     rho         = 1.0 + 0.2 * std::sin(2.0 * M_PI * (x[0] - domainLeft) / domainWidth);
        Q[0] = rho;
        Q[1] = rho * u;
        Q[2] = 0.0;
        #if Dimensions == 2
        Q[3] = p / (gamma - 1) + 0.5 * rho * u * u;
        #else
        Q[3] = 0.0;
        Q[4] = p / (gamma - 1) + 0.5 * rho * u * u;
        #endif
        """,
            stateless=False,
        )
    )

elif args.application == "ccz4":
    thesolver = exagrype.api.FOCCZ4Solver(solver_name)

thesolver.numerical_scheme = exahype2.api.fv.RusanovWithExplicitEuler(thesolver, args.patch_size)
solver_variants = []
if args.cpu == True or args.gpu == False:
    solver_variants.append((exahype2.api.Target.Serial, True, exahype2.api.OutputLanguage.CPP, exahype2.api.MemoryLayout.AoS_AllNested))
    solver_variants.append((exahype2.api.Target.Serial, True, exahype2.api.OutputLanguage.CPP, exahype2.api.MemoryLayout.AoS_CellDataFlattened_TemporariesNested))
    solver_variants.append((exahype2.api.Target.Serial, True, exahype2.api.OutputLanguage.CPP, exahype2.api.MemoryLayout.AoS_CellDataNested_TemporariesFlattened))
    solver_variants.append((exahype2.api.Target.Serial, True, exahype2.api.OutputLanguage.CPP, exahype2.api.MemoryLayout.AoS_AllFlattened))

    solver_variants.append((exahype2.api.Target.Serial, True, exahype2.api.OutputLanguage.MLIR, exahype2.api.MemoryLayout.AoS_AllNested))
    solver_variants.append((exahype2.api.Target.Serial, True, exahype2.api.OutputLanguage.MLIR, exahype2.api.MemoryLayout.AoS_CellDataFlattened_TemporariesNested))
    solver_variants.append((exahype2.api.Target.Serial, True, exahype2.api.OutputLanguage.MLIR, exahype2.api.MemoryLayout.AoS_CellDataNested_TemporariesFlattened))
    solver_variants.append((exahype2.api.Target.Serial, True, exahype2.api.OutputLanguage.MLIR, exahype2.api.MemoryLayout.AoS_AllFlattened))
if args.gpu == True:
    solver_variants.append((exahype2.api.Target.Accelerator, True, exahype2.api.OutputLanguage.CPP, exahype2.api.MemoryLayout.AoS_AllFlattened))

thesolver.numerical_scheme._generator = exahype2.api.fv.RusanovWithExplicitEulerGenerator(thesolver, solver_variants)
thesolver.grid_construction = exahype2.api.fv.SetupSimulationOverRegularGrid(
    thesolver, volume_size=0.001
)

project.add_solver(thesolver)

build_mode = peano4.api.output.CompileMode.RELEASE
project.set_global_simulation_parameters(
    dimensions=args.dimensions,
    offset=[-0.1, -0.1] if args.dimensions == 2 else [-0.1, -0.1, -0.1],
    size=[0.2, 0.2] if args.dimensions == 2 else [0.2, 0.2, 0.2],
    min_end_time=0.01,
    max_end_time=0.01,
    periodic_BC=[True, True, True],
)


project.set_Peano4_installation("../../../../../", build_mode)
peano4_project = project.generate_Peano4_project(verbosity=verbosity)

peano4_project.constants.export_constexpr_with_type("Accuracy", str(args.accuracy), "double")
peano4_project.constants.export_constexpr_with_type("NumberOfSamples", str(args.samples), "int")

formatted_num_patches = "{{{}}}".format(", ".join(str(val) for val in args.num_patches))
peano4_project.constants.export_const_with_type("NumberOfPatchesToStudy", str(formatted_num_patches), "tarch::la::Vector<%s, int>" % len(args.num_patches))
peano4_project.constants.export_constexpr_with_type("NumberOfLaunchingThreads", str(1), "int")

peano4_project.generate()
peano4_project.build(make_clean_first=True)