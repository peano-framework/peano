# This file is part of the ExaHyPE2 project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
import peano4.api
import peano4.api.output.Jinja2TemplatedHeaderImplementationFilePair
from peano4.api.taskgraph.compiler.PropagateUseStatements import propagate_use_statements
from peano4.api.taskgraph.compiler.DeriveLoadStoreComputeFlag import derive_load_store_compute_flag

import dastgen2

import abc
import os
import sys


class Project(object):
    """!

    ExaHyPE 2 project

    Represents an ExaHyPE 2 project. An ExaHyPE2 project is a Peano 4
    project with a particular set of actions (algorithmic phases)
    that you can choose from and with particular solver types. It
    realises a builder mechanism, i.e. you build up your ExaHyPE 2
    project and then you finally tell the project "give me the Peano 4
    project". From hereon, you can use this Peano 4 project to actually
    set up the Peano 4 application.

    The project will have a marker per cell that encodes stuff alike
    a boundary marker. But it is also used to coordinate different
    solver types.

    @see generate_Peano4_project()
    """

    class Generator(abc.ABC):
        """!

        Base class for the code generation steps that turn an ExaHyPE 2
        Project's configuration into actual generated files.

        generate_Peano4_project() below currently does this work inline (see
        the "@todo Has to go into a generator" note in there); the various
        steps it performs (writing the solver implementation files, the
        solver repository, the main file, ...) are meant to become Generator
        subclasses instead, of which SolverToCPPFiles is the first.

        """

        def __init__(self, project: "Project"):
            self.project = project

        @abc.abstractmethod
        def construct_output(self, peano_project):
            """!
            Add this generator's output to peano_project (a
            peano4.api.Project-compatible object, or anything else that
            add()'s cleanly into peano_project.output). This is the one
            thing every Generator has to implement.

            peano_project: peano4.api.Project
              The Peano 4 project that self.project.generate_Peano4_project()
              is currently assembling. It is a local variable there, not an
              attribute - merely passed through here, so this generator can
              add its own artefacts to it.
            """

    def __init__(self, namespace, project_name, executable, directory=".", subdirectory="", abstract_overwrite=True):
        # Kept around only to construct the peano4.api.Project afresh each
        # time generate_Peano4_project() runs - see there. That project is a
        # local variable, not an attribute: repeated calls to
        # generate_Peano4_project() must yield an independent snapshot
        # rather than accumulate onto a stale, shared instance.
        self._namespace = namespace
        self._project_name = project_name
        self._directory = directory
        self._subdirectory = subdirectory
        self._abstract_overwrite = abstract_overwrite

        self._solvers = []
        # What is realised: single solvers and hosts of several solvers, in the order of their addition
        self._realisations = []
        self._domain_offset = [0.0, 0.0, 0.0]
        self._domain_size = [1.0, 1.0, 1.0]
        self._dimensions = 3
        self._min_terminal_time = 1.0
        self._max_terminal_time = 0.0
        self._load_balancer_name      = ""
        self._load_balancer_arguments = ""
        self._log_filter_file         = "exahype2.log-filter"
        self._number_of_threads       = "tarch::multicore::Core::UseDefaultNumberOfThreads"
        self._multicore_orchestration = "tarch::multicore::orchestration::createDefaultStrategy()"
        self._timeout                 = 3600
        self._gpus                    = []
        self._step_report_class       = "peano4::DefaultStepReport"
        self._additional_includes     = [ "tarch/logging/LogFilterFileReader.h",
                                          "tarch/accelerator/Device.h"
                                        ]
        self._Peano_src_directory     = "."
        self._build_mode = peano4.api.output.CompileMode.ASSERTS
        self._executable_name = executable
        self._periodic_BC = [False, False, False]
        self._output_path = "./"

        # Turns this project's configuration into the header/implementation
        # file pair holding main(). Defaults to an ExaHyPE-flavoured main
        # similar to python/exahype2/ExaHyPEMain.template.h/.cpp; assign a
        # different Project.Generator subclass instance to use your own.
        from .generators.DefaultSimulationMain import DefaultSimulationMain
        self.mainfile_generator = DefaultSimulationMain(self)

        # Generates repositories/SolverRepository.h/.cpp (one extern
        # instance per registered solver, plus the getMinTimeStamp()/
        # getMaxTimeStamp() aggregators).
        from .generators.SolverRepositoryGenerator import SolverRepositoryGenerator
        self.solver_repository_generator = SolverRepositoryGenerator(self)

        # Translates each of the four algorithmic phases' task graphs into a
        # sequence of Observers (peano4.api.solversteps.Step). Defaults to
        # the simplest possible compiler, which just linearises the task
        # graph without exploiting any of its parallelism; assign a
        # different translate() routine (e.g. from a hypothetical
        # peano4.api.taskgraph.compiler.Parallel) to use a smarter one.
        self.compiler = peano4.api.taskgraph.compiler.Sequential.translate


    def set_load_balancing(self, load_balancer_name, load_balancer_arguments=""):
        """! Set load balancing scheme
        
        The load balancing scheme controls how subtress are distributed both
        among ranks and threads, i.e. the instance is important for MPI but 
        also whenever you use any multithreading strategy.
        
        Similar to set_multicore_orchestration(), you have to let the project
        know about where the load balancing is held. If you rely on the 
        load balancing toolbox and if you have configured Peano with the 
        load balancing toolbox, then simply adding the right eader is 
        sufficient:
        
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        project.add_mainfile_include( "toolbox/loadbalancing/strategies/Hardcoded.h)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        If you use a bespoke load balancing, you also have to add the 
        implementation file or hosting library to the project.
 
        @param load_balancer_name: string
          Should be full-qualified name of the load balancer.
          By default, I recommend to pass "toolbox::loadbalancing::strategies::RecursiveSubdivision".

        @param load_balancer_arguments: string
          If your load balancing requires parameters, add them
          here. It is a string that will be copied into the C++ class instantiation.
          Please add the brackets yourself, i.e. "(3,4,5)" is fine, but "3,4,5" is not.
          The only exception is the empty parameter list. Here, you can/should simply
          add the empty string.
        """
        self._load_balancer_name = load_balancer_name
        self._load_balancer_arguments = load_balancer_arguments


    def set_log_filter_file(self, file_name):
        """! Set a log filter file name
        
        Pass in the empty string, if you don't want the code to use the 
        hard-coded defaults. That is, by default an ExaHyPE2 application
        searches for a file exahype2.log-filter. Through this call, you can 
        instruct the project to use another filter file.
        
        @param file_name String
          Name of the log filter file that the application should load on 
          startup.
          
        """
        self._log_filter_file = file_name


    def set_number_of_threads(self, value):
        """! Set number of threads to use
        
        Pass in number of threads (positive integer) or a string such as
        "tarch::multicore::Core::UseDefaultNumberOfThreads". Please note that
        this value is handed through to Peano which in turn passes it on the
        tarch::multicore::Core::configure(int). It then depends on the 
        multithreading backend if this backend allows you to set the thread
        count manually. Many backends (such as OpenMP) do not allow this, 
        but require users to use environment variables (or scheduler options)
        instead.
        
        """
        self._number_of_threads = value


    def set_multicore_orchestration(self, value):
        """! Set orchestration
        
        The orchestration accepts one value which is a C++ statement. This
        value has to be a string which creates an orchestration object,
        i.e. an object of the type tarch::multicore::orchestration::Orchestration,
        on the heap.
        
        You can either create it directly (in this case you need to add a new),
        or you can use a factory method such as
        tarch::multicore::orchestration::StrategyFactory::createDefaultStrategy().
        
        Depending on where you come from, you might have to add the strategy's
        header file to the project explicitly. By default, the ExaHyPE 2 
        project only is aware of very few strategies. Here's an example:
        
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        project.add_mainfile_include( "tarch/multicore/orchestration/Hardcoded.h")
        project.set_multicore_orchestration(
            "new benchmarks::exahype2::ccz4::MulticoreOrchestration()"
        )
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        @see exahype2.Project.add_mainfile_include()        
        @param value C++ expression delivering a new orchestration object on the heap
        
        """
        self._multicore_orchestration = value


    def set_timeout(self, value):
        """! Set timeout
        value has to be a number greater or equal to zero, this then sets the
        timeout in seconds after which a node waiting for an MPI message shall
        quit an shutdown the whole application with an error report.
        If zero is passed, this feature is switched off and nodes will wait
        indefinitely.
        """
        if value < 0:
            raise Exception("Timeout has to be greater or equal to 0")
        self._timeout = value


    def set_step_report_class(self, class_name):
        """! Set the C++ class the generated main instantiates to report on steps

        class_name: str
          Fully qualified C++ class name (e.g. "peano4::DefaultStepReport",
          the default) of a peano4::RunMultipleSteps::Report subclass. The
          generated main creates exactly one instance of it, so the class
          has to be default-constructible. If you pass in anything other
          than the default, its header is not included automatically - you
          have to make it visible to the generated main yourself.
        """
        self._step_report_class = class_name


    def set_gpus(self, value):
        """! Set GPUs to be used
        value is a list of integers or the empty list if you want the code to
        use all GPUs that are visible to a rank.
        """
        self._gpus = value


    """
    The standard extensions that I use for both Peano and ExaHyPE.
    """
    LibraryDebug = "_debug"
    LibraryRelease = ""
    LibraryTrace = "_trace"
    LibraryAsserts = "_asserts"
    LibraryStats = "_stats"

    def set_Peano4_installation(self, src_path, mode=peano4.api.output.CompileMode.RELEASE):
        """
        src_path: string
          Path (relative or absolute) to the src directory of Peano. This path
          should hold both the headers (in subdirectories) and all the static
          libraries.

        mode: peano4.api.output.CompileMode
        """
        self._Peano_src_directory = src_path
        self._build_mode = mode

    def add_solver(self, solver):
        """!

        Add a solver, or a host of several coupled solvers

        @param solver: exahype2.api.Solver or exahype2.api.SolverHost
          A host (@ref exahype2.api.SolverHost) brings its solvers along. The
          project registers each of them with its solver repository and
          README, but lets the host realise them, so the host can couple the
          task graphs of its solvers.

        """
        from .SolverHost import SolverHost

        if isinstance(solver, SolverHost):
            self._solvers.extend(solver.solvers)
        else:
            self._solvers.append(solver)
        self._realisations.append(solver)

    def remove_all_solvers(self):
        self._solvers = []
        self._realisations = []
        self._peano_project.cleanup()

    def __export_constants(self, peano_project):
        """
        We export ExaHyPE's constants. Besides the constants from ExaHyPE,
        we also export some parameters from Peano onto the ExaHyPE constants
        file. Therefore, it is important that you parse the configure output
        before we export the constants.

        peano_project: peano4.api.Project
          The Peano 4 project that generate_Peano4_project() is currently
          assembling. It is a local variable there, not an attribute -
          merely passed through here, so this routine can export onto it.
        """
        peano_project.constants.clear()
        offset_string = "{" + str(self._domain_offset[0])
        size_string = "{" + str(self._domain_size[0])
        for i in range(1, self._dimensions):
            offset_string += ","
            size_string += ","
            offset_string += str(self._domain_offset[i])
            size_string += str(self._domain_size[i])
        offset_string += "}"
        size_string += "}"
        peano_project.constants.add_include("""#include <bitset>""")
        peano_project.constants.add_include("""#include "tarch/la/Vector.h" """)
        peano_project.constants.add_include("""#include "peano4/utils/Globals.h" """)
        peano_project.constants.export_const_with_type(
            "DomainOffset", offset_string, "tarch::la::Vector<Dimensions,double>"
        )
        peano_project.constants.export_const_with_type(
            "DomainSize", size_string, "tarch::la::Vector<Dimensions,double>"
        )
        peano_project.constants.export("MinTerminalTime", str(self._min_terminal_time))
        if self._max_terminal_time >= self._min_terminal_time:
            peano_project.constants.export(
                "MaxTerminalTime", str(self._max_terminal_time)
            )
        else:
            peano_project.constants.export(
                "MaxTerminalTime", "std::numeric_limits<double>::max()"
            )

        peano_project.constants.export_boolean_sequence("PeriodicBC", self._periodic_BC)

        build_string = "python3 "
        for i in sys.argv:
            build_string += " "
            build_string += i
        peano_project.constants.export_string("BuildInformation", build_string)
        peano_project.constants.export_string(
            "ConfigureInformation", peano_project.output.makefile.configure_call
        )

        for i in self._solvers:
            readme_descriptor = i.create_readme_descriptor(self._domain_offset[0], self._domain_size[0])
            if readme_descriptor is not None:
                peano_project.output.readme.add_entry(readme_descriptor)

    def set_global_simulation_parameters(
        self,
        dimensions,
        offset,
        size,
        min_end_time,
        periodic_BC=[False, False, False],
        max_end_time=0.0,
    ):
        """
        ! Set the global simulation parameters

        @param dimensions: 2 or 3

        @param offset: [Float]
          Has to have dimension number of entries.

        @param size: [Float}
          Has to have dimension number of entries and all of them have to be
          positive. At this point, we strongly suggest that all size entries
          are exactly the same. While Peano and ExaHyPE do support anisotropic
          meshes, most compute kernels do not, so working with something else
          than a square or cube will likely break all numerics.

        @param max_end_time: Float
          If you set it zero (or actually any value smaller than min_end_time), then
          the code will run until the cell that lags behind the most hits the min time.
          If you specify a valid max time however, you can stop the sim as soon as the
          most advanced cell exceeds this threshold.

        There are no plot settings here: when and how often to plot is
        configured per solver on its plotter, e.g.
        exahype2.api.fv.PlotAsVTKFile.
        """
        self._domain_offset = offset
        self._domain_size = size
        self._dimensions = dimensions

        assert (
            len(self._domain_offset) == dimensions
        ), "domain offset vector and dimensions don't match"
        assert (
            len(self._domain_size) == dimensions
        ), "domain offset vector and dimensions don't match"

        self._max_terminal_time = max_end_time
        self._min_terminal_time = min_end_time

        self._periodic_BC = []
        for d in range(0, dimensions):
            self._periodic_BC.append(periodic_BC[d])

    def set_output_path(self, path):
        self._output_path = path
        if not os.path.exists(self._output_path):
            try:
                os.makedirs(self._output_path)
            except:
                print("Couldn't create the defined output path, please ensure that this path exists")
        if not self._output_path.endswith("/"):
            self._output_path += "/"


    _EXAHYPE_PACKAGE_DESCRIPTION = """

### ExaHyPE 2

This code uses ExaHyPE in its second generation. The first generation of the
code was developed through an EU FET HPC project called ExaHyPE. This first
generation was built on top of Peano in its third generation. The present
code uses ExaHyPE 2 which is a complete rewrite built on top of Peano 4. We
do not yet have a release paper for this second generation of ExaHyPE, and
thus appreciate any citation of the original release paper

       @article{Reinarz:2020:ExaHyPE,
         title = {ExaHyPE: An engine for parallel dynamically adaptive simulations of wave problems},
         journal = {Computer Physics Communications},
         volume = {254},
         pages = {107251},
         year = {2020},
         issn = {0010-4655},
         doi = {https://doi.org/10.1016/j.cpc.2020.107251},
         url = {https://www.sciencedirect.com/science/article/pii/S001046552030076X},
         author = {Anne Reinarz and Dominic E. Charrier and Michael Bader and Luke Bovard and Michael Dumbser and Kenneth Duru and Francesco Fambri and Alice-Agnes Gabriel and Jean-Matthieu Gallard and Sven K\"oppel and Lukas Krenz and Leonhard Rannabauer and Luciano Rezzolla and Philipp Samfass and Maurizio Tavelli and Tobias Weinzierl},
         keywords = {Hyperbolic, PDE, ADER-DG, Finite volumes, AMR, MPI, TBB, MPI+X},
         abstract = {ExaHyPE (An Exascale Hyperbolic PDE Engine) is a software engine for solving systems of first-order hyperbolic partial differential equations (PDEs). Hyperbolic PDEs are typically derived from the conservation laws of physics and are useful in a wide range of application areas. Applications powered by ExaHyPE can be run on a student's laptop, but are also able to exploit thousands of processor cores on state-of-the-art supercomputers. The engine is able to dynamically increase the accuracy of the simulation using adaptive mesh refinement where required. Due to the robustness and shock capturing abilities of ExaHyPE's numerical methods, users of the engine can simulate linear and non-linear hyperbolic PDEs with very high accuracy. Users can tailor the engine to their particular PDE by specifying evolved quantities, fluxes, and source terms. A complete simulation code for a new hyperbolic PDE can often be realised within a few hours - a task that, traditionally, can take weeks, months, often years for researchers starting from scratch. In this paper, we showcase ExaHyPE's workflow and capabilities through real-world scenarios from our two main application areas: seismology and astrophysics.
           Program summary
           Program title: ExaHyPE-Engine Program Files doi: http://dx.doi.org/10.17632/6sz8h6hnpz.1 Licensing provisions: BSD 3-clause Programming languages: C++, Python, Fortran Nature of Problem: The ExaHyPE PDE engine offers robust algorithms to solve linear and non-linear hyperbolic systems of PDEs written in first order form. The systems may contain both conservative and non-conservative terms. Solution method: ExaHyPE employs the discontinuous Galerkin (DG) method combined with explicit one-step ADER (arbitrary high-order derivative) time-stepping. An a-posteriori limiting approach is applied to the ADER-DG solution, whereby spurious solutions are discarded and recomputed with a robust, patch-based finite volume scheme. ExaHyPE uses dynamical adaptive mesh refinement to enhance the accuracy of the solution around shock waves, complex geometries, and interesting features.
         }
       }
"""

    def generate_Peano4_project(self, verbosity=peano4.api.LoggerVerbosity.OFF):
        """!

        Construct a Peano 4 project and return it

        Build the Peano 4 project, i.e., all the action sets et al that you require
        to run this ExaHyPE 2 application. The Peano 4 project is a local variable
        here - a fresh peano4.api.Project - that is populated over the course of
        this call and eventually returned.

        This routine generates a Peano project, i.e. the domain-specific ExaHyPE
        view is translated into a Peano model. As the Peano project is not kept
        around as an attribute, calling generate_Peano4_project() again after you
        have altered the ExaHyPE setup gives you an independent, fresh snapshot of
        the Peano setup rather than an accumulation on top of a stale one.
        """
        peano4.api.Logger().verbosity = verbosity

        peano_project = peano4.api.Project(
            self._namespace, self._project_name, self._directory, self._subdirectory
        )
        peano_project.output.readme.add_package_description(self._EXAHYPE_PACKAGE_DESCRIPTION)

        peano_project.output.makefile.set_dimension(self._dimensions)
        peano_project.output.makefile.set_executable_name(self._executable_name)
        peano_project.output.makefile.set_mode(self._build_mode)
        peano_project.output.makefile.parse_configure_script_outcome(self._Peano_src_directory)

        self.__export_constants(peano_project)

        # The task graphs for the four algorithmic phases every ExaHyPE 2
        # project goes through. They only live for the duration of this call
        # - each solver contributes its tasks into them as it is realised.
        # The mesh adaptation phase is optional: It remains empty if no
        # solver has a Solver.MeshAdaptation, and the generated main never
        # runs an empty phase.
        setup_simulation = peano4.api.taskgraph.Graph()
        time_step = peano4.api.taskgraph.Graph()
        plot = peano4.api.taskgraph.Graph()
        mesh_adaptation = peano4.api.taskgraph.Graph()

        for solver in self._realisations:
            solver.generate_solver_realisation(
                self, peano_project, setup_simulation, time_step, plot, mesh_adaptation
            )

        # Each solver's construct_output() above visualises the (input)
        # task graph it just contributed to - Setup, TimeStep, Plot and
        # MeshAdaptation - via Graph.visualise(), which renders without
        # blocking so all of them can appear together; this is where we
        # actually pause and let the user look at them before moving on.
        peano4.api.taskgraph.wait_for_visualisations()

        #
        # The steps tied to one phase. This is really a sequence of
        # observers (aka Peano grid traversals) which the code should
        # run through for a given phase.
        #
        setup_simulation_steps = self.compiler(setup_simulation, "SetupSimulation", peano_project)
        time_step_steps        = self.compiler(time_step, "TimeStep", peano_project)
        plot_steps             = self.compiler(plot, "Plot", peano_project)
        mesh_adaptation_steps  = self.compiler(mesh_adaptation, "MeshAdaptation", peano_project)

        #
        # Setup runs once, as a prequel to both time stepping and plotting;
        # time stepping and plotting may then follow one another in either
        # order, repeatedly (each carries a self-edge for that reason) -
        # @see peano4.api.taskgraph.compiler.PropagateUseStatements for what
        # these edges mean and why a repeating sequence needs a self-edge.
        # This postprocessing runs on the compiled steps, as it counts mesh
        # traversals, and a compiler might fuse several tasks into one.
        #
        # The mesh adaptation only ever runs right after a time step, and
        # it is followed by either a time step or a plot. It never runs
        # twice in a row, as its tasks repeat themselves until the mesh is
        # stationary. It takes part only if any solver has contributed to
        # it: an empty sequence without incoming edges would otherwise
        # count as an entry sequence.
        #
        sequences = [setup_simulation_steps, time_step_steps, plot_steps]
        edges = [
            (setup_simulation_steps, time_step_steps),
            (setup_simulation_steps, plot_steps),
            (time_step_steps, plot_steps),
            (plot_steps, time_step_steps),
            (time_step_steps, time_step_steps),
            (plot_steps, plot_steps),
        ]
        if mesh_adaptation_steps:
            sequences.append(mesh_adaptation_steps)
            edges += [
                (time_step_steps, mesh_adaptation_steps),
                (mesh_adaptation_steps, time_step_steps),
                (mesh_adaptation_steps, plot_steps),
            ]
        propagate_use_statements(sequences, edges=edges)

        # Only now do we actually know which Steps this project runs, and
        # under which qualifier each of them uses a given submodel - the
        # compiler above is free to fuse, drop or reorder tasks, so this
        # cannot be derived from the pre-compilation task graphs above -
        # @see peano4.api.taskgraph.compiler.DeriveLoadStoreComputeFlag.
        derive_load_store_compute_flag(
            setup_simulation_steps + time_step_steps + plot_steps + mesh_adaptation_steps
        )

        # Symmetric to the wait_for_visualisations() call above, but for
        # the *compiled* task graphs each self.compiler() call
        # just reported (@see
        # peano4.api.taskgraph.compiler.ReportCompiledStepSequence).
        peano4.api.taskgraph.wait_for_visualisations()

        #
        # Add these observers to the Peano project, so a main file can
        # trigger them.
        #
        for step in setup_simulation_steps + time_step_steps + plot_steps + mesh_adaptation_steps:
            peano_project.solversteps.add_step(step)

        #
        # Create one runner object per phase. A runner object has only one 
        # big routine (run()) and this routine triggers all the observers 
        # (aka Peano grid traversals) that logically make up a simulation phase.
        #
        peano4.api.taskgraph.create_runner(peano_project, "SetupSimulation", setup_simulation_steps)
        peano4.api.taskgraph.create_runner(peano_project, "TimeStep", time_step_steps)
        peano4.api.taskgraph.create_runner(peano_project, "Plot", plot_steps)
        peano4.api.taskgraph.create_runner(peano_project, "MeshAdaptation", mesh_adaptation_steps)

        self.solver_repository_generator.construct_output(peano_project)

        self.mainfile_generator.construct_output(peano_project)

        return peano_project

    @property
    def periodic_BC(self):
        """!
        One boolean per axis (dimensions entries) telling whether the domain
        is periodic along this axis. Read-only copy - @see
        set_global_simulation_parameters() to change it.
        """
        return list(self._periodic_BC)

    @property
    def subdirectory(self):
        return self._subdirectory

    @property
    def additional_includes(self):
        return self._additional_includes
