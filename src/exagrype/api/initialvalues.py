""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

import exahype2.api

FIRST_ORDER_FUNCTIONS = (
    "gaugeWave",
    "diagonal_gaugeWave",
    "linearWave",
    "flat",
    "singleBlackHole",
)
"""!
The routines in exagrype::initialvalues which fill all 58 entries of
first-order CCZ4. exagrype.api.FOSOCCZ4Solver rejects initial conditions which
call one of them, as it hands the initial condition 25 entries only.
"""


def _init_point_wisely(cpp_function: str) -> exahype2.api.InitPointWisely:
    return exahype2.api.InitPointWisely(
        implementation=f"::exagrype::initialvalues::{cpp_function}(Q, x, 0.0);"
    )


def gauge_wave() -> exahype2.api.InitPointWisely:
    """!

    Gauge wave along the x-axis, @see exagrype::initialvalues::gaugeWave()

    """
    return _init_point_wisely("gaugeWave")


def diagonal_gauge_wave() -> exahype2.api.InitPointWisely:
    """!

    Gauge wave along the diagonal, @see exagrype::initialvalues::diagonal_gaugeWave()

    """
    return _init_point_wisely("diagonal_gaugeWave")


def linear_wave() -> exahype2.api.InitPointWisely:
    """!

    Linearised gravitational wave, @see exagrype::initialvalues::linearWave()

    """
    return _init_point_wisely("linearWave")


def flat() -> exahype2.api.InitPointWisely:
    """!

    Flat (Minkowski) spacetime, @see exagrype::initialvalues::flat()

    """
    return _init_point_wisely("flat")


def gauge_wave_evolved_variables() -> exahype2.api.InitPointWisely:
    """!

    Gauge wave along the x-axis, the 25 evolved unknowns only

    For exagrype.api.FOSOCCZ4Solver
    (@ref exagrype.api.initialvalues.gauge_wave(),
    exagrype::initialvalues::gaugeWaveEvolvedVariables()).

    """
    return _init_point_wisely("gaugeWaveEvolvedVariables")


def diagonal_gauge_wave_evolved_variables() -> exahype2.api.InitPointWisely:
    """!

    Gauge wave along the diagonal, the 25 evolved unknowns only

    For exagrype.api.FOSOCCZ4Solver
    (@ref exagrype.api.initialvalues.diagonal_gauge_wave(),
    exagrype::initialvalues::diagonal_gaugeWaveEvolvedVariables()).

    """
    return _init_point_wisely("diagonal_gaugeWaveEvolvedVariables")


def linear_wave_evolved_variables() -> exahype2.api.InitPointWisely:
    """!

    Linearised gravitational wave, the 25 evolved unknowns only

    For exagrype.api.FOSOCCZ4Solver
    (@ref exagrype.api.initialvalues.linear_wave(),
    exagrype::initialvalues::linearWaveEvolvedVariables()).

    """
    return _init_point_wisely("linearWaveEvolvedVariables")


def flat_evolved_variables() -> exahype2.api.InitPointWisely:
    """!

    Flat (Minkowski) spacetime, the 25 evolved unknowns only

    For exagrype.api.FOSOCCZ4Solver
    (@ref exagrype.api.initialvalues.flat(),
    exagrype::initialvalues::flatEvolvedVariables()).

    """
    return _init_point_wisely("flatEvolvedVariables")


def single_black_hole(mass: float = 1.0) -> exahype2.api.InitPointWisely:
    """!

    Single Schwarzschild black hole at the origin

    The data stems from TwoPunctures, @see exagrype::initialvalues::singleBlackHole().
    TwoPunctures is only part of the ExaGRyPE library if Peano is configured
    with GSL (--with-gsl). Otherwise, the generated solver refuses to
    compile.

    The black hole's puncture sits at the origin, where the data is
    singular. Choose the mesh such that no degree of freedom coincides with
    the origin, e.g. a Finite Volume patch with an even number of volumes
    per axis on a domain that is symmetric around the origin.

    @param mass: float
      ADM mass of the black hole.

    """
    return exahype2.api.InitPointWisely(
        implementation=f"""
#ifdef UseGSL
  ::exagrype::initialvalues::singleBlackHole(Q, x, 0.0, {mass});
#else
#error "exagrype.api.initialvalues.single_black_hole() requires TwoPunctures: configure Peano with --with-gsl (or -DWITH_GSL=ON)"
#endif
"""
    )
