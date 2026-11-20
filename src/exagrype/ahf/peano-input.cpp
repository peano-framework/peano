#include "peano-input.h"

namespace exahype2::ccz4::ahf {

void ReadGridFromPeano(const PeanoGridData& peanoGridData) {
  (void)peanoGridData;
  // Loading grid data from the Peano data structure among timesteppings.
  // Must be done in parallel as each core only holds a subdomain of the grid. 
  // This approach requires binding codes to ExaGRyPE, reducing its generality.
  // – Input: Peano grid data
  // – Output: Full grid dataset. A full grid dataset giving solution $\{Q_{i,j,k}\}$ on each cell/volume, 
  //           and the corresponding Cartesian coordinates $\{x_I,y_I,z_I\}$. 
  //           They may be stored in subdomains, managed by multiple cores when the domain is large.
}

}  // namespace exahype2::ccz4::ahf
