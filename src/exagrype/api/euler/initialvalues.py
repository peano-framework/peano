""" This file is part of the ExaGRyPE project. For conditions of distribution and
 use, please see the copyright notice at www.peano-framework.org
"""

import exahype2.api


def smooth_wave(
    amplitude: float = 0.2, velocity: float = 1.0, pressure: float = 1.0
) -> exahype2.api.InitPointWisely:
    """!

    Density wave on constant pressure, @see exagrype::euler::initialvalues::smoothWave()

    The wave is an exact solution of the Euler equations which the fluid
    advects along the x-axis. The adiabatic index is the one of the solver
    (its attribute Gamma), i.e. the routine is for exagrype.api.euler.Euler.

    @param amplitude: float
      Amplitude of the density wave. Has to be smaller than one in modulus
      to keep the density positive.

    @param velocity: float
      Speed along the x-axis.

    @param pressure: float
      Constant pressure.

    """
    if abs(amplitude) >= 1.0:
        raise ValueError("amplitude has to be smaller than one in modulus")
    return exahype2.api.InitPointWisely(
        implementation=f"""
  ::exagrype::euler::initialvalues::smoothWave( Q, x, 0.0, {amplitude!r}, {velocity!r}, {pressure!r}, Gamma );
"""
    )
