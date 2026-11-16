// This file is part of the ExaGRyPE project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once


#include <cmath>
#include "tarch/tarch.h"
#include "tarch/compiler/CompilerSpecificSettings.h"
#include "tarch/multicore/multicore.h"
#include "tarch/accelerator/accelerator.h"
#include "tarch/la/Vector.h"


namespace exagrype {
  namespace ccz4 {

      /**
       * Non-conservative product of the first-order CCZ4 formulation.
       *
       * The 58 entries of CCZ4 (see @ref page_exagrype_foccz4) are handed
       * over as two pointers: one to the 25 primary entries (indices 0 to
       * 24) and one to the 33 secondary entries (indices 25 to 57, i.e. the
       * derivatives of the lapse, the shift, the metric and phi).
       * The first-order formulation stores them contiguously and passes Q
       * and Q+25. The second-order formulation (exagrype.api.FOSOCCZ4Solver)
       * reconstructs the secondary entries from the primary ones and hence
       * stores them in separate memory, namely in its auxiliary variables.
       * The same split applies to all other arguments below which hold CCZ4
       * entries.
       *
       * The first-order and the second-order variant share one
       * implementation in Kernels.cpp (a template over the formulation), so
       * the physics is identical. The second-order variant omits the
       * advection terms and the terms of the secondary entries which the
       * first-order formulation needs, and it only writes the 25 primary
       * entries of its result.
       *
       * Neither variant holds the advection terms
       * @f$ \beta^k \partial_k Q @f$ which need the derivative of the
       * advected entry itself. The numerical scheme has to add them, as
       * they are discretised with lopsided stencils (see
       * exagrype.api.CCZ4SolverBase.lopsided_advection_quantities()). The
       * implementation in Kernels.cpp still knows these terms, but only
       * compiles them if its template argument Advection is set, which
       * none of the kernels below does.
       *
       * @param BgradQ The 25 primary entries of the result.
       * @param BgradQaux The 33 secondary entries of the result (first-order
       *   variant only).
       * @param Q The 25 primary entries of the solution.
       * @param Qaux The 33 secondary entries of the solution.
       * @param gradQ The derivative of the 25 primary entries along normal.
       * @param gradQaux The derivative of the 33 secondary entries along normal.
       * @param normal Axis along which the derivatives are given.
       * @param CCZ4b Integer switch: variant of the Gamma driver. 1 is the
       *   shifting shift, 0 the non-shifting shift (see below).
       * @param CCZ4s Integer switch: 1 evolves the shift (Gamma driver), 0
       *   freezes it.
       *
       * The Gamma driver reads
       * @f$ \partial_t \beta^i = s ( b \beta^k B_k^i + f b^i ) @f$,
       * @f$ \partial_t b^i = s ( \partial_t \hat\Gamma^i - \eta b^i + b ( \beta^k \partial_k b^i - \beta^k \partial_k \hat\Gamma^i ) ) @f$
       * and
       * @f$ \partial_t B_j^i = s ( b ( \beta^k \partial_k B_j^i + B_k^i B_j^k ) + f \partial_j b^i ) @f$
       * plus the constraint terms with CCZ4mu. Here, s is CCZ4s and the
       * factor b (without an index) is CCZ4b. With b=1, all three equations
       * hold their advection terms (shifting shift). With b=0, none does
       * (non-shifting shift). The time derivative of
       * @f$ \hat\Gamma^i @f$ in the second equation is the complete one,
       * i.e. it comprises the advection of @f$ \hat\Gamma^i @f$. As the
       * kernels leave this advection to the numerical scheme, the
       * non-conservative product adds it to the equation of
       * @f$ b^i @f$ for b=0. The third equation is the derivative of
       * the first one, which is why b also switches off the product of the
       * derivatives of the shift
       * (src/exagrype/ccz4/adrs/0002-ccz4b-selects-the-shifting-or-the-non-shifting-shift.md).
       *
       * The slicing condition is
       * @f$ \partial_t \alpha = -\alpha^2 g(\alpha) (K - 2 c \Theta) @f$,
       * i.e. there is no offset K0 for the trace of the extrinsic
       * curvature, and K0 is no entry of CCZ4. The former parameters CCZ4ds
       * and CCZ4xi are fixed to one. The rationale is written down in
       * src/exagrype/ccz4/adrs/0001-ccz4-has-58-entries-and-integer-switches.md.
       */
      #if defined(SharedOMP)
      #pragma omp declare simd
      #endif
      void ncpFirstOrder(double* BgradQ, double* BgradQaux, const double* const Q, const double* const Qaux, const double* const gradQ, const double* const gradQaux, const int normal,
        const int CCZ4LapseType,
        const double CCZ4c,
        const double CCZ4e,
        const double CCZ4f,
        const int CCZ4b,
        const int CCZ4s,
        const double CCZ4mu
      );

      /**
       * Second-order variant of ncpFirstOrder(): updates the 25 primary entries only.
       */
      #if defined(SharedOMP)
      #pragma omp declare simd
      #endif
      void ncpSecondOrder(double* BgradQ, const double* const Q, const double* const Qaux, const double* const gradQ, const double* const gradQaux, const int normal,
        const int CCZ4LapseType,
        const double CCZ4c,
        const double CCZ4e,
        const double CCZ4f,
        const int CCZ4b,
        const int CCZ4s,
        const double CCZ4mu
      );

      /**
       * The source term is one out of two terms that we use in our CCZ4
       * formulation. As it is used within the compute kernel, we want to
       * vectorise over it aggressively, hence the simd annotation.
       *
       * Q and Qaux are the primary and the secondary entries as documented
       * for @ref ncpFirstOrder(). S and Saux are the right-hand side for the
       * two groups.
       */
      #if defined(SharedOMP)
      #pragma omp declare simd
      #endif
      void sourceFirstOrder(double* S, double* Saux, const double* const Q, const double* const Qaux,
        const int CCZ4LapseType,
        const double CCZ4c,
        const double CCZ4e,
        const double CCZ4f,
        const int CCZ4b,
        const int CCZ4s,
        const double CCZ4itau,
        const double CCZ4eta,
        const double CCZ4k1,
        const double CCZ4k2,
        const double CCZ4k3
      );

      /**
       * Second-order variant of sourceFirstOrder(): updates the 25 primary entries only.
       */
      #if defined(SharedOMP)
      #pragma omp declare simd
      #endif
      void sourceSecondOrder(double* S, const double* const Q, const double* const Qaux,
        const int CCZ4LapseType,
        const double CCZ4c,
        const double CCZ4e,
        const double CCZ4f,
        const int CCZ4b,
        const int CCZ4s,
        const double CCZ4itau,
        const double CCZ4eta,
        const double CCZ4k1,
        const double CCZ4k2,
        const double CCZ4k3
      );

      #if defined(SharedOMP)
      #pragma omp declare simd
      #endif
      void maxEigenvalue(
        const double* const Q,
        int                 normal,
        const double        CCZ4e,
        const double        CCZ4GLMc,
        const double        CCZ4GLMd,
        double*             maxEigenvalue
      );

      /**
       * This is a postprocessing routine to monitor if the physical constraints
       * are fulfilled. Used in python script.
       */
      void admconstraints(double* constraints, const double* const Q, const double* const gradQSerialised);

      /**
       * A temporary test function, to output Hamilton constraint related term in theta, 1 terms: RPlusTwoNablaZNCP
       */
      void ThetaOutputNCP(double* NCPterm, const double* const Q, const double* const gradQSerialised, int normal);
      /**
       * A temporary test function, to output some testing values 
       * 0,1 entries: Hamilton constraint related term in theta, 2 terms: RPlusTwoNablaZSrc, pure Src
       * 2-10 entry: no-symmetry version of R_ij
       */
      void TestingOutput(double* terms, const double* const Q, const double* const gradQSerialised);

      /**
       * This function is for the calculation of psi4, a quantity related to gravitional wave. 
       * Not used yet. 
       */
      void Psi4Calc(double* Psi4, const double* const Q, const double* const gradQSerialised, double* coor);


      /**
       * A postprocessing routine which pushes the volume solution back into the
       * area of the CCZ4 constraints. This is a local operation that we can
       * invoke after each time step per volume.
       *
       * It works on the full set of 58 unknowns of the first-order formulation.
       * It first calls enforceCCZ4constraintsOfEvolvedVariables() and then
       * additionally makes the derivatives of the conformal metric trace-free.
       */
      void enforceCCZ4constraints(double* __restrict__  Q);

      /**
       * The part of enforceCCZ4constraints() which only touches the first 25
       * unknowns
       *
       * These are the unknowns that the second-order formulation evolves
       * (exagrype.api.FOSOCCZ4Solver). The routine gives the conformal
       * metric the determinant one, makes the extrinsic curvature
       * trace-free and bounds the lapse and phi from below. The derivatives
       * of the metric, i.e. entries 37 to 54, are reconstructed from the
       * metric in the second-order formulation and hence cannot violate
       * any constraint on their own.
       */
      void enforceCCZ4constraintsOfEvolvedVariables(double* __restrict__  Q);

      /**
       * Add the Kreiss-Oliger dissipation of a patch to its right-hand side
       *
       * The routine adds
       *
       * @f$ \frac{\sigma}{64 \Delta x_d} (Q_{i-3} - 6 Q_{i-2} + 15 Q_{i-1} - 20 Q_i + 15 Q_{i+1} - 6 Q_{i+2} + Q_{i+3}) @f$
       *
       * per axis d to the right-hand side of each grid cell i of a patch,
       * i.e. a sixth-order dissipation term which damps oscillations on the
       * scale of the grid cells. @f$ \Delta x_d @f$ is the patch size along
       * d divided by the number of grid cells per axis. The term is added to
       * the selected quantities only, all other entries of the right-hand
       * side remain unchanged.
       *
       * The routine is the dissipation which the legacy fourth-order Finite
       * Differences of ExaHyPE built into their kernels. These kernels have
       * been removed.
       * The Finite Differences of exahype2.api.rkfd4 have no dissipation.
       * exagrype.api.FD4WithKODissipation invokes this routine after their
       * right-hand side kernel, within each Runge-Kutta stage.
       *
       * Both arrays are stored lexicographically in AoS format, i.e. all
       * quantities of one grid cell are stored together, and the first axis
       * runs fastest. The stencil reaches three grid cells into the halo,
       * i.e. haloSize has to be at least three.
       *
       * @param QWithHalo Input of the Runge-Kutta stage with its halo, i.e.
       *   (numberOfGridCellsPerPatchPerAxis+2*haloSize)^3 grid cells with
       *   unknowns+auxiliaryVariables entries each.
       * @param rightHandSide Right-hand side of the stage without halo, i.e.
       *   numberOfGridCellsPerPatchPerAxis^3 grid cells with unknowns entries
       *   each. The routine adds the dissipation to it.
       * @param patchSize Size of the patch, i.e. of the mesh cell.
       * @param quantities Indices of the unknowns which are subject to the
       *   dissipation. Each index has to be smaller than unknowns.
       * @param numberOfQuantities Number of entries of quantities.
       */
      void addKreissOligerDissipation(
        const double* __restrict__          QWithHalo,
        double* __restrict__                rightHandSide,
        const tarch::la::Vector<3, double>& patchSize,
        int                                 numberOfGridCellsPerPatchPerAxis,
        int                                 haloSize,
        int                                 unknowns,
        int                                 auxiliaryVariables,
        double                              sigma,
        const int*                          quantities,
        int                                 numberOfQuantities
      );
  }
}
