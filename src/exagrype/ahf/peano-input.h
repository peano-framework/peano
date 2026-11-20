#ifndef APPARENT_HORIZON_FINDER_PEANO_INPUT_H_
#define APPARENT_HORIZON_FINDER_PEANO_INPUT_H_

#include <string>
#include <vector>

namespace exahype2::ccz4::ahf {

struct PeanoGridData;

/**
 * Load grid dataset from the Peano simulation on-the-fly.
 */
void ReadGridFromPeano(const PeanoGridData& peanoGridData);

}  // namespace exahype2::ccz4::ahf

#endif  // APPARENT_HORIZON_FINDER_PEANO_INPUT_H_
