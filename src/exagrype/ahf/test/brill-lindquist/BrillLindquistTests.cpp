#include "../require-asserts.h"

#include "../../apparent-horizon-finder.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

namespace exahype2::ccz4::ahf::tests::brill_lindquist {

  namespace {

    const double SixteenPi = 16.0 * std::numbers::pi;

    bool approxEqual(double a, double b, double tol) {
      return std::abs(a - b) <= tol;
    }

    ReferencePoint toReference(const FieldVector<double>& p) {
      return ReferencePoint{p[0], p[1], p[2]};
    }

    // Drive a full damped-Newton horizon find on an analytic spacetime from a
    // coordinate sphere of radius r0, centred on referencePoint.
    template <class Spacetime>
    SpectralNewtonResult solveHorizon(
      const SpectralBasis&  basis,
      const Spacetime&      spacetime,
      const ReferencePoint& referencePoint,
      double                r0
    ) {
      const SpectralResidual residual = MakeAnalyticResidual(
        basis,
        spacetime,
        referencePoint
      );
      return SolveSpectralNewton(
        basis,
        residual,
        SphereCoefficients(basis, r0)
      );
    }

    // Smallest synthesized radius over the surface (star-shapedness check).
    double minRadius(
      const SpectralBasis&       basis,
      const std::vector<double>& coefficients
    ) {
      const SynthesizedField field    = Synthesize(basis, coefficients);
      double                 smallest = field.h.front();
      for (double value : field.h) {
        smallest = std::min(smallest, value);
      }
      return smallest;
    }

    // Coefficient of the (l, m) mode, or 0 if the basis does not carry it.
    double coefficientFor(
      const SpectralBasis&       basis,
      const std::vector<double>& a,
      int                        l,
      int                        m
    ) {
      for (std::size_t j = 0; j < basis.modes.size(); ++j) {
        if (basis.modes[j].first == l && basis.modes[j].second == m) {
          return a[j];
        }
      }
      return 0.0;
    }

    const std::vector<Vector3> probes{
      {0.3, -0.4, 3.2},
      {1.1, 0.7, 2.6},
      {-0.6, 0.9, 3.5},
      {0.2, 1.3, 2.9}};

  } // namespace

  // A single puncture at the origin reproduces Schwarzschild isotropic exactly.
  void testSinglePunctureEqualsSchwarzschildIsotropic() {
    const BrillLindquist         puncture = BrillLindquist::single(1.3);
    const SchwarzschildIsotropic schwarzschild{1.3};
    for (const Vector3& X : probes) {
      const Tensor3 blMetric            = MetricData(puncture, X).spatialMetric;
      const Tensor3 schwarzschildMetric = MetricData(schwarzschild, X)
                                            .spatialMetric;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          assert(blMetric[i][j] == schwarzschildMetric[i][j]);
        }
      }
    }
    std::cout
      << "Single puncture reproduces Schwarzschild isotropic gamma exactly"
      << std::endl;
  }

  // The binary slice is time symmetric, so K_ij = 0 everywhere.
  void testTimeSymmetricExtrinsicCurvatureVanishes() {
    const BrillLindquist binary = BrillLindquist::binary(1.0, 2.0);
    for (const Vector3& X : probes) {
      const InterpolatedGeometryData data = MetricData(binary, X);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          assert(approxEqual(data.extrinsicCurvature[i][j], 0.0, 1.0e-12));
        }
      }
    }
    std::cout << "Binary slice gives K_ij = 0" << std::endl;
  }

  // ADM mass, centroid and puncture positions bookkeeping.
  void testAdmMassAndBookkeeping() {
    const BrillLindquist binary = BrillLindquist::
      binary(0.7, 3.0, BrillLindquist::Axis::Z);
    assert(approxEqual(binary.admMass(), 1.4, 1.0e-12));

    const FieldVector<double> centre = binary.centroid();
    for (int i = 0; i < 3; ++i) {
      assert(approxEqual(centre[i], 0.0, 1.0e-12));
    }

    const std::vector<FieldVector<double>> positions = binary.puncturePositions(
    );
    assert(positions.size() == 2);
    assert(approxEqual(positions[0][2], 1.5, 1.0e-12));
    assert(approxEqual(positions[1][2], -1.5, 1.0e-12));

    std::cout
      << "Binary ADM mass, centroid and puncture positions correct"
      << std::endl;
  }

  // The single-puncture conveniences must not pretend to work for a binary.
  void testSinglePunctureConveniences() {
    assert(approxEqual(
      BrillLindquist::single(2.0).horizonArea(),
      SixteenPi * 4.0,
      1.0e-9
    ));

    bool threw = false;
    try {
      (void)BrillLindquist::binary(1.0, 2.0).M();
    } catch (const std::logic_error&) {
      threw = true;
    }
    assert(threw);

    std::cout
      << "Single-puncture conveniences reject multi-puncture slices"
      << std::endl;
  }

  // Single-puncture horizon: areal radius 2M, irreducible mass M.
  void testSinglePunctureHorizon() {
    const SpectralBasis        basis    = BuildSpectralBasis(4);
    const BrillLindquist       puncture = BrillLindquist::single(1.0);
    const SpectralNewtonResult result   = solveHorizon(
      basis,
      puncture,
      ReferencePoint{0.0, 0.0, 0.0},
      0.8
    );
    assert(result.converged);

    const double area = HorizonArea(
      basis,
      result.coefficients,
      puncture,
      ReferencePoint{0.0, 0.0, 0.0}
    );
    assert(approxEqual(ArealRadius(area), 2.0, 1.0e-6));
    assert(approxEqual(IrreducibleMass(area), 1.0, 1.0e-6));
    std::cout
      << "Single-puncture horizon: areal radius " << ArealRadius(area)
      << std::endl;
  }

  namespace {

    // Individual horizon around the first puncture of a binary at separation d.
    double binaryIndividualArea(const SpectralBasis& basis, double separation) {
      const BrillLindquist binary = BrillLindquist::
        binary(1.0, separation, BrillLindquist::Axis::Z);
      const ReferencePoint s0 = toReference(binary.puncturePositions()[0]);
      const SpectralNewtonResult result = solveHorizon(basis, binary, s0, 0.6);
      assert(result.converged);
      assert(minRadius(basis, result.coefficients) > 0.0); // star-shaped
      return HorizonArea(basis, result.coefficients, binary, s0);
    }

  } // namespace

  // The companion enlarges the individual horizon beyond the isolated value,
  // less so the further away it is: 16 pi m^2 < area(d=8) < area(d=4) < 16 pi
  // M_ADM^2.
  void testBinaryIndividualHorizon() {
    const SpectralBasis basis     = BuildSpectralBasis(8);
    const double        areaClose = binaryIndividualArea(basis, 4.0);
    const double        areaFar   = binaryIndividualArea(basis, 8.0);
    const double        admArea   = SixteenPi * 2.0 * 2.0; // M_ADM = 2

    assert(SixteenPi < areaFar);
    assert(areaFar < areaClose);
    assert(areaClose < admArea);
    std::cout
      << "Binary individual horizon: area(d=4) = " << areaClose
      << ", area(d=8) = " << areaFar << " (isolated " << SixteenPi << ")"
      << std::endl;
  }

  // The individual horizon area -> 16 pi m^2 as the separation grows.
  void testBinaryIndividualApproachesIsolated() {
    const SpectralBasis basis = BuildSpectralBasis(8);
    const double        area  = binaryIndividualArea(basis, 30.0);
    assert(std::abs(area / SixteenPi - 1.0) < 0.05);
    std::cout
      << "Binary individual horizon at d=30 approaches isolated (area/16pi = "
      << area / SixteenPi << ")" << std::endl;
  }

  namespace {

    // Common horizon enclosing both punctures, reference point at the centroid.
    SpectralNewtonResult binaryCommon(
      const SpectralBasis& basis,
      double               separation,
      double&              area,
      BrillLindquist&      binary
    ) {
      binary = BrillLindquist::binary(1.0, separation, BrillLindquist::Axis::Z);
      const ReferencePoint       s0     = toReference(binary.centroid());
      const SpectralNewtonResult result = solveHorizon(basis, binary, s0, 2.2);
      area                              = result.converged
                                            ? HorizonArea(basis, result.coefficients, binary, s0)
                                            : 0.0;
      return result;
    }

  } // namespace

  // Penrose bound (area <= merged Schwarzschild area) and merger limit (smaller
  // separation is closer to the merged area).
  void testCommonHorizonPenroseAndMergerLimit() {
    const SpectralBasis        basis     = BuildSpectralBasis(8);
    double                     areaSmall = 0.0;
    double                     areaMid   = 0.0;
    BrillLindquist             binary    = BrillLindquist::single(1.0);
    const SpectralNewtonResult small     = binaryCommon(
      basis,
      0.5,
      areaSmall,
      binary
    );
    const SpectralNewtonResult mid = binaryCommon(basis, 1.2, areaMid, binary);
    assert(small.converged && mid.converged);

    const double merged = SixteenPi * binary.admMass()
                          * binary.admMass();         // d -> 0 limit
    assert(areaSmall <= merged && areaMid <= merged); // Penrose
    assert(areaSmall > areaMid);                      // merger limit
    assert(std::abs(ArealRadius(areaSmall) - 2.0 * binary.admMass()) < 0.05);
    std::cout
      << "Binary common horizon: area(d=0.5) = " << areaSmall
      << ", area(d=1.2) = " << areaMid << " (merged " << merged << ")"
      << std::endl;
  }

  // No common horizon exists once the punctures are well separated: the
  // finder's runaway guard flags the divergence.
  void testCommonHorizonAbsentAtLargeSeparation() {
    const SpectralBasis        basis  = BuildSpectralBasis(8);
    double                     area   = 0.0;
    BrillLindquist             binary = BrillLindquist::single(1.0);
    const SpectralNewtonResult result = binaryCommon(basis, 2.5, area, binary);
    assert(!result.converged);
    assert(
      result.message.find("spurious") != std::string::npos
      || result.message.find("non-star") != std::string::npos
    );
    std::cout
      << "Binary common horizon absent at d=2.5: \"" << result.message << "\""
      << std::endl;
  }

  // Three equal punctures on an equilateral triangle: bookkeeping.
  void testTriangleGeometry() {
    const BrillLindquist triangle = BrillLindquist::
      triangle(1.0, 1.8, BrillLindquist::Plane::XY);
    assert(approxEqual(triangle.admMass(), 3.0, 1.0e-12));

    const FieldVector<double> centre = triangle.centroid();
    for (int i = 0; i < 3; ++i) {
      assert(approxEqual(centre[i], 0.0, 1.0e-12));
    }

    const std::vector<FieldVector<double>>
      positions = triangle.puncturePositions();
    assert(positions.size() == 3);
    for (const FieldVector<double>& p : positions) {
      assert(approxEqual(p[2], 0.0, 1.0e-12)); // triangle lies in the x-y plane
    }
    for (int i = 0; i < 3; ++i) {
      const FieldVector<double>& p    = positions[i];
      const FieldVector<double>& q    = positions[(i + 1) % 3];
      const double               side = std::sqrt(
        (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1])
        + (p[2] - q[2]) * (p[2] - q[2])
      );
      assert(approxEqual(side, 1.8, 1.0e-12)); // equilateral, side = separation
    }
    std::cout
      << "Triangle geometry: equilateral, ADM mass 3, centroid at origin"
      << std::endl;
  }

  namespace {

    SpectralNewtonResult triangleCommon(
      const SpectralBasis& basis,
      double               side,
      double&              area,
      BrillLindquist&      triangle
    ) {
      triangle = BrillLindquist::triangle(1.0, side, BrillLindquist::Plane::XY);
      const ReferencePoint s0 = toReference(triangle.centroid());
      const SpectralNewtonResult result = solveHorizon(basis, triangle, s0, 3.0);
      area = result.converged
               ? HorizonArea(basis, result.coefficients, triangle, s0)
               : 0.0;
      return result;
    }

  } // namespace

  // Penrose bound and merger limit for the three-puncture common horizon.
  void testTriangleCommonHorizonPenroseAndMergerLimit() {
    const SpectralBasis        basis     = BuildSpectralBasis(8);
    double                     areaSmall = 0.0;
    double                     areaMid   = 0.0;
    BrillLindquist             triangle  = BrillLindquist::single(1.0);
    const SpectralNewtonResult small     = triangleCommon(
      basis,
      0.5,
      areaSmall,
      triangle
    );
    const SpectralNewtonResult mid = triangleCommon(
      basis,
      1.5,
      areaMid,
      triangle
    );
    assert(small.converged && mid.converged);

    const double merged = SixteenPi * triangle.admMass() * triangle.admMass();
    assert(areaSmall <= merged && areaMid <= merged); // Penrose
    assert(areaSmall > areaMid);                      // merger limit
    assert(std::abs(ArealRadius(areaSmall) - 2.0 * triangle.admMass()) < 0.05);
    std::cout
      << "Triangle common horizon: area(s=0.5) = " << areaSmall
      << ", area(s=1.5) = " << areaMid << std::endl;
  }

  // The three-fold symmetry carries l=3 power in the m=+/-3 harmonics, absent
  // for one or two punctures; l=3, m=0 is not excited, and there is no dipole
  // (centroid is correct).
  void testTriangleCommonHorizonNonAxisymmetric() {
    const SpectralBasis        basis    = BuildSpectralBasis(8);
    double                     area     = 0.0;
    BrillLindquist             triangle = BrillLindquist::single(1.0);
    const SpectralNewtonResult result   = triangleCommon(
      basis,
      1.8,
      area,
      triangle
    );
    assert(result.converged);

    const double c33  = coefficientFor(basis, result.coefficients, 3, 3);
    const double c3m3 = coefficientFor(basis, result.coefficients, 3, -3);
    const double c30  = coefficientFor(basis, result.coefficients, 3, 0);
    assert(c33 * c33 + c3m3 * c3m3 > 1.0e-3); // three-fold (m=+/-3) content
    assert(c30 * c30 < 1.0e-8);               // l=3 is not axisymmetric

    const std::vector<double> power = PowerSpectrum(basis, result.coefficients);
    assert(power[1] < 1.0e-8); // no dipole -> centroid is correct
    std::cout
      << "Triangle common horizon is non-axisymmetric (m=+/-3 power "
      << c33 * c33 + c3m3 * c3m3 << ")" << std::endl;
  }

  // Individual horizon around one puncture of the triangle: two companions
  // enlarge it.
  void testTriangleIndividualHorizon() {
    const SpectralBasis  basis    = BuildSpectralBasis(8);
    const BrillLindquist triangle = BrillLindquist::
      triangle(1.0, 4.0, BrillLindquist::Plane::XY);
    const ReferencePoint s0 = toReference(triangle.puncturePositions()[0]);
    const SpectralNewtonResult result = solveHorizon(basis, triangle, s0, 0.6);
    assert(result.converged);
    assert(minRadius(basis, result.coefficients) > 0.0); // star-shaped

    const double area = HorizonArea(basis, result.coefficients, triangle, s0);
    const double admArea = SixteenPi * triangle.admMass() * triangle.admMass();
    assert(SixteenPi < area && area < admArea);
    std::cout
      << "Triangle individual horizon: area = " << area << " (isolated "
      << SixteenPi << ")" << std::endl;
  }

} // namespace exahype2::ccz4::ahf::tests::brill_lindquist

int main() {
  using namespace exahype2::ccz4::ahf::tests::brill_lindquist;

  std::cout
    << "Apparent Horizon Finder Brill-Lindquist tests begin..." << std::endl;

  testSinglePunctureEqualsSchwarzschildIsotropic();
  testTimeSymmetricExtrinsicCurvatureVanishes();
  testAdmMassAndBookkeeping();
  testSinglePunctureConveniences();
  testSinglePunctureHorizon();
  testBinaryIndividualHorizon();
  testBinaryIndividualApproachesIsolated();
  testCommonHorizonPenroseAndMergerLimit();
  testCommonHorizonAbsentAtLargeSeparation();
  testTriangleGeometry();
  testTriangleCommonHorizonPenroseAndMergerLimit();
  testTriangleCommonHorizonNonAxisymmetric();
  testTriangleIndividualHorizon();

  std::cout << "All Brill-Lindquist tests completed." << std::endl;
  return 0;
}
