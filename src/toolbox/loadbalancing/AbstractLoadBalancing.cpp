#include "AbstractLoadBalancing.h"

#include "peano4/parallel/SpacetreeSet.h"

tarch::logging::Log toolbox::loadbalancing::AbstractLoadBalancing::_log("toolbox::loadbalancing::AbstractLoadBalancing"
);


std::ostream& operator<<(std::ostream& out, const toolbox::loadbalancing::AbstractLoadBalancing& balancing) {
  out << balancing.toString();
  return out;
}


toolbox::loadbalancing::AbstractLoadBalancing::AbstractLoadBalancing(
  Configuration* configuration, CostMetrics* costMetrics
):
  _blacklist(),
  _statistics(),
  _configuration(configuration),
  _costMetrics(costMetrics),
  _state(State::Undefined) {}


toolbox::loadbalancing::AbstractLoadBalancing::~AbstractLoadBalancing() {
  if (_configuration != nullptr) {
    delete _configuration;
  }
  if (_costMetrics != nullptr) {
    delete _costMetrics;
  }
}


void toolbox::loadbalancing::AbstractLoadBalancing::setConfigurationAndMetricsNullWithoutDelete() {
  _configuration = nullptr;
  _costMetrics   = nullptr;
}


std::string toolbox::loadbalancing::AbstractLoadBalancing::toString() const {
  std::ostringstream msg;

  msg << "(state=" << ::toString(_state) << ",statistics=" << _statistics.toString();

  if (_costMetrics!=nullptr) {
    msg << ",cost-metrics=" << _costMetrics->toString();
  }
  else {
    msg << ",no-cost-metrics";
  }

  msg<< ",blacklist=" << _blacklist.toString() << ")";

  return msg.str();
}


bool toolbox::loadbalancing::AbstractLoadBalancing::hasSplitRecently() const {
  return _statistics.getNumberOfStateUpdatesWithoutAnySplit() < NumberOfMeshSweepsAfterSplitToMakeMeshConsistentAgain or _statistics.getLocalNumberOfSplits() > 0;
}


bool toolbox::loadbalancing::AbstractLoadBalancing::hasMeshBeenStationary() const {
  return _statistics.getNumberOfStateUpdatesWithoutAnySplit() >= NumberOfMeshSweepsAfterSplitToMakeMeshConsistentAgain + 1 and _statistics.getLocalNumberOfSplits() == 0;
}


void toolbox::loadbalancing::AbstractLoadBalancing::enable(bool value) {
  if (value and _state == State::SwitchedOff) {
    _state = State::InterRankBalancing;
    logInfo("enable()", "switched load balancing on (again) and therefore change to state " << ::toString(_state));
  } else if (not value and _state != State::SwitchedOff) {
    logInfo("enable()", "switched load balancing off");
    _state = State::SwitchedOff;
  }

  _statistics.notifyOfStateChange(_state);
}


bool toolbox::loadbalancing::AbstractLoadBalancing::isEnabled(bool globally) const {
  logDebug("isEnabled(bool)", toString());
  return globally ? (_statistics.getGlobalNumberOfRanksWithEnabledLoadBalancing() > 0) : (_state != State::SwitchedOff);
}


bool toolbox::loadbalancing::AbstractLoadBalancing::hasStagnated() const {
  return _state == State::Stagnation or _state == State::SwitchedOff;
}


int toolbox::loadbalancing::AbstractLoadBalancing::getGlobalNumberOfTrees() const {
  return _statistics.getGlobalNumberOfTrees();
}


bool toolbox::loadbalancing::AbstractLoadBalancing::fitsIntoMemory([[maybe_unused]] State state) const {
  // This is a magic parameter with the 2.
  // You might still run out of memory if you refine while you rebalance.
  if (_configuration->makeSplitDependOnMemory(_state)) {
    int worstCaseEstimateForSizeOfSpacetree = 2 * tarch::getMemoryUsage(tarch::MemoryUsageFormat::MByte);
    return tarch::getFreeMemory(tarch::MemoryUsageFormat::MByte) >= worstCaseEstimateForSizeOfSpacetree;
  } else {
    return true;
  }
}


void toolbox::loadbalancing::AbstractLoadBalancing::finishSimulation() {}


bool toolbox::loadbalancing::AbstractLoadBalancing::isInterRankBalancingBad() const {
  if (_statistics.getGlobalNumberOfTrees() < tarch::mpi::Rank::getInstance().getNumberOfRanks()) {
    return true;
  } else if ( static_cast<int>(peano4::parallel::SpacetreeSet::getInstance().getLocalSpacetrees().size()) >= _configuration->getMaxLocalTreesPerRank(_state) ) {
    return false;
  } else {
    const double optimalCost = static_cast<double>(_costMetrics->getGlobalCost())
                               / tarch::mpi::Rank::getInstance().getNumberOfRanks();

    const double illbalancing   = (_costMetrics->getCostOfLocalRank() - optimalCost) / optimalCost;
    const double balancingRatio = _configuration->getWorstCaseBalancingRatio(_state);
    bool         result         = illbalancing > 1.0 - balancingRatio;

    if (result) {
      logInfo(
        "doesRankViolateBalancingCondition()",
        "rank does violate balancing as we have ill-balancing of "
          << illbalancing << " (global cost=" << _costMetrics->getGlobalCost() << ")"
      );
    }

    return result;
  }
}


bool toolbox::loadbalancing::AbstractLoadBalancing::isIntraRankBalancingBad() const {
  if (peano4::parallel::SpacetreeSet::getInstance().getLocalSpacetrees().size() == 0) {
    return false;
  } else if (peano4::parallel::SpacetreeSet::getInstance().getLocalSpacetrees().size() == 1) {
    return true;
  } else if ( static_cast<int>(peano4::parallel::SpacetreeSet::getInstance().getLocalSpacetrees().size()) >= _configuration->getMaxLocalTreesPerRank(_state) ) {
    return false;
  } else {
    double maxCost = 0;
    double minCost = std::numeric_limits<double>::max();

    for (auto p : peano4::parallel::SpacetreeSet::getInstance().getLocalSpacetrees()) {
      maxCost = std::max(maxCost, _costMetrics->getCostOfLocalTree(p));
      minCost = std::min(minCost, _costMetrics->getCostOfLocalTree(p));
    }

    assertion(minCost >= 0.0);
    assertion(maxCost >= 0.0);

    bool result = false;

    if (maxCost > 0.0 and minCost > 0.0) {
      double relativeWeightSmalltestTree = minCost / maxCost;
      double balanceThreshold            = _configuration->getWorstCaseBalancingRatio(_state);

      result = relativeWeightSmalltestTree < balanceThreshold;

      logDebug(
        "isLocalBalancingBad()",
        "local trees are ill-balanced="
          << result << ", rel-weight=" << relativeWeightSmalltestTree << ", threshold=" << balanceThreshold
          << ", #min-cost=" << minCost << ", #max-cost=" << maxCost
      );
    }

    return result;
  }
}


bool toolbox::loadbalancing::AbstractLoadBalancing::areRanksUnemployed() const {
  return tarch::la::equals(_costMetrics->getMinimumOfMaximumRankWeights(), 0.0)
         and _statistics.getGlobalNumberOfTrees() > 0;
}


double toolbox::loadbalancing::AbstractLoadBalancing::getWeightOfHeaviestLocalSpacetree() const {
  const int heaviestSpacetree = getIdOfHeaviestLocalSpacetree();
  logDebug(
    "getWeightOfHeaviestLocalSpacetree()",
    "id="
      << heaviestSpacetree << ", #octants="
      << peano4::parallel::SpacetreeSet::getInstance()
           .getGridStatistics(heaviestSpacetree)
           .getNumberOfLocalUnrefinedCells()
  );
  return heaviestSpacetree == NoHeaviestTreeAvailable ? -1 : _costMetrics->getCostOfLocalTree(heaviestSpacetree);
}


int toolbox::loadbalancing::AbstractLoadBalancing::getIdOfHeaviestLocalSpacetree() const {
  std::set<int> idsOfLocalSpacetrees = peano4::parallel::SpacetreeSet::getInstance().getLocalSpacetrees();
  int           result               = NoHeaviestTreeAvailable;
  double        maxLocalCost         = -1;

  for (auto p : idsOfLocalSpacetrees) {
    if (
      _costMetrics->getCostOfLocalTree(p)>maxLocalCost
      and
      peano4::parallel::SpacetreeSet::getInstance().getGridStatistics(p).getNumberOfLocalUnrefinedCells()>=ThreePowerD
    ) {
      maxLocalCost = _costMetrics->getCostOfLocalTree(p) > maxLocalCost;
      result       = p;
    }
  }
  return result;
}


int toolbox::loadbalancing::AbstractLoadBalancing::getIdOfHeaviestLocalSpacetree(double tolerance) const {
  std::set<int> idsOfLocalSpacetrees = peano4::parallel::SpacetreeSet::getInstance().getLocalSpacetrees();
  int           result               = NoHeaviestTreeAvailable;
  int           maxLocalCost         = -1;

  for (auto p : idsOfLocalSpacetrees) {
    if (
      _costMetrics->getCostOfLocalTree(p)>maxLocalCost
      and
      peano4::parallel::SpacetreeSet::getInstance().getGridStatistics(p).getNumberOfLocalUnrefinedCells()>=ThreePowerD
    ) {
      maxLocalCost = _costMetrics->getCostOfLocalTree(p);
    }
  }

  if (maxLocalCost > 0.0) {
    logInfo(
      "getIdOfHeaviestLocalSpacetree(double)",
      "try to find one maximum tree which has weight close to "
        << maxLocalCost << ", i.e. has more than " << (tolerance * maxLocalCost) << " cost"
    );
    for (auto p : idsOfLocalSpacetrees) {
      if (
        tarch::la::greaterEquals( _costMetrics->getCostOfLocalTree(p), tolerance * maxLocalCost )
        and
        peano4::parallel::SpacetreeSet::getInstance().getGridStatistics(p).getNumberOfLocalUnrefinedCells()>=ThreePowerD
        and
        (rand() % 2 == 1 or result == NoHeaviestTreeAvailable)
      ) {
        logInfo(
          "getIdOfHeaviestLocalSpacetree(double)",
          "tree " << p << " meets criterion with weight of " << _costMetrics->getCostOfLocalTree(p)
        );
        result = p;
      }
    }
  }
  return result;
}
