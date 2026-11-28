// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once


#include <map>

#include "Blacklist.h"
#include "Configuration.h"
#include "CostMetrics.h"
#include "Statistics.h"
#include "peano4/parallel/Node.h"
#include "tarch/logging/Log.h"
#include "tarch/mpi/mpi.h"


namespace toolbox {
  namespace loadbalancing {
    class AbstractLoadBalancing;
  } // namespace loadbalancing
} // namespace toolbox


/**
 * Abstract superclass of any load balancing
 */
class toolbox::loadbalancing::AbstractLoadBalancing {
public:
  /**
   * Is used by tree identification and either indicates that there are no trees
   * at all or means that the heaviest tree is on the blacklist. See implementation
   * remarks in class description.
   */
  static constexpr int NoHeaviestTreeAvailable = -1;

  /**
   * Number of mesh sweeps a split needs until the mesh is consistent again
   *
   * A split needs three sweeps after the one at whose end the strategy has
   * triggered it:
   *
   * 1. The new trees are created. The statistics register the split at the end
   *    of this sweep, i.e. they set the number of state updates without any
   *    split to zero.
   * 2. The new trees receive their cells.
   * 3. The old tree hands over these cells, so the cell counts are consistent.
   *
   * The constant counts the state updates after the one which has registered
   * the split, i.e. it stands for the sweeps two and three.
   *
   * @see hasSplitRecently()
   */
  static constexpr int NumberOfMeshSweepsAfterSplitToMakeMeshConsistentAgain = 2;

  /**
   * Constructor
   *
   * Data ownership is handed over to the load balancing, i.e. the balancing
   * has to delete the configuration object unless you pass in the nullptr.
   * If you pass in the nullptr, you cannot use the object, so this should
   * be done with care by any subclass.
   *
   * Many implementations of AbstractLoadBalancing change the _state field
   * immediately in their constructor. Please ensure that you invoke
   *
   *       _statistics.notifyOfStateChange(_state);
   *
   * afterwards to ensure that the statistics are correct right from the
   * start.
   *
   *
   * ## Rationale
   *
   * I originally thought it might be reasonable to add assertions to ensure
   * that configuration and costMetrics are not nullptr. However, there are
   * specific hard-coded load balancing schemes which do explicitly use
   * nullptr arguments. Hardcoded is the prime example. Therefore, I do not
   * nullptr checks here and I leave it to subclasses to add the
   * corresponding assertions if they require proper pointers here.
   */
  AbstractLoadBalancing(Configuration* configuration, CostMetrics* costMetrics);

  /**
   * Destructor
   *
   * Free the configuration and cost metrics objects. See the discussion in
   * the constructor for contextual information re the validity of nullptr
   * arguments here.
   */
  virtual ~AbstractLoadBalancing();


  /**
   * Create a string representation of load balancing object
   *
   * Generic string serialisation. You might want to extend it. Therefore,
   * the routine is labelled as virtual.
   *
   * The routine has to be able to deal with the fact that _costMetrics
   * might be nullptr according to AbstractLoadBalancing(). The configuration
   * in return does not feed into toString(). Therefore, we don't have to check
   * if it is pointing to nullptr.
   */
  virtual std::string toString() const;


  /**
   * Switch on/off.
   */
  virtual void enable(bool);


  /**
   * Has mesh split recently
   *
   * Means that mesh has split over the past
   * NumberOfMeshSweepsAfterSplitToMakeMeshConsistentAgain iterations, i.e.
   * the iterations a proper split needs until the mesh is consistent again.
   *
   * If we have then not come to the conclusion that we should immediately
   * split again, then we haven't split recently.
   *
   * Splits which this rank has triggered since the last state update count,
   * too. Most strategies trigger their splits in finishStep() after they
   * have updated their statistics. Without this check, the predicate would
   * hence not hold right after a strategy has triggered a split, but only
   * once the subsequent finishStep() has updated the statistics. Code which
   * decides right after finishStep() whether to run another grid sweep,
   * such as the repeat condition of a mesh construction, would then stop
   * before the split has been carried out.
   */
  virtual bool hasSplitRecently() const;


  /**
   * Has the mesh been stationary with respect to splits
   *
   * Holds one state update after hasSplitRecently() has ceased to hold, and as
   * long as this rank has not triggered a new split. It does not tell whether
   * the grid itself is stationary (@ref peano4::grid::isGridStationary()).
   *
   * ## Why a mesh construction cannot use hasSplitRecently() to terminate
   *
   * A mesh construction pauses its refinement while hasSplitRecently() holds,
   * as the new trees have to catch up first. A paused refinement leaves the
   * grid stationary. Each sweep decides at its start whether it refines, while
   * the repetition is checked after finishStep(), i.e. at its end. When
   * hasSplitRecently() ceases to hold, the grid hence is stationary because
   * the last sweep has not been allowed to refine. If the construction
   * terminates now, it never resumes the refinement. With the strategy
   * SpreadOut, which splits as soon as every thread can get a tree, the mesh
   * stopped at nine cells. A construction therefore has to run until
   * hasMeshBeenStationary() holds, which grants it one sweep with refinement.
   */
  virtual bool hasMeshBeenStationary() const;


  /**
   * Is load balancing enabled
   *
   * Clarifies whether load balancing is, in principle, enabled. We we return
   * true, it means that we are not disabled. The balancer might still be
   * temporarily suspended.
   */
  bool isEnabled(bool globally) const;


  /**
   * Delegate to stats.
   */
  virtual int getGlobalNumberOfTrees() const;


  /**
   * A load balancing can either be stagnating or be switched off for
   * this predicate to hold.
   */
  virtual bool hasStagnated() const;


  virtual void finishSimulation();


  /**
   * Finish the step
   *
   * No matter what you do here, you have to invoke
   *
   *          _statistics.updateGlobalView();
   *          _blacklist.update();
   *
   * in this routine.
   *
   * If you use the actionset toolbox.laodbalancing.api.LoadBalancing, then
   * the unprepareTraversal() invokes the finishStep() for you. This in turn
   * means, as per contract, that the statistics have an updated global view.
   */
  virtual void finishStep() = 0;

  /**
   * This is only used when you concatenate balancing rules and you want
   * to disable any deletion. Please do not invoke it unless you know exactly
   * what you are doing.
   */
  void setConfigurationAndMetricsNullWithoutDelete();

protected:
  static tarch::logging::Log _log;

  Blacklist _blacklist;

  Statistics _statistics;

  Configuration* _configuration;

  CostMetrics* _costMetrics;

  /**
   * Ensure that you invoke
   *
   *        _statistics.notifyOfStateChange(_state)
   *
   * whenever you change the state. Otherwise, the MPI will have an invalid
   * view of the whole world.
   */
  State _state;

  /**
   * Ensure enough memory is left-over.
   */
  bool fitsIntoMemory(State state) const;


  /**
   * Is the balancing between the ranks ok
   *
   * This operation checks if the inter-rank load decomposition does
   * violate the overall load balancing constraints. The global balancing
   * is determined by a cascade of checks:
   *
   * - If we have an inconsistent data view, i.e. if the stats are not valid,
   *   we have to assume that everything is fine.
   * - If we have fewer trees globally than we have ranks, the balancing is
   *   by definition poor.
   * - If we host more trees than allowed, the balancing by definition is
   *   good, since we could not refine further anyway.
   * - Otherwise, we look if the max load per rank and the min load per rank
   *   differ more than the tolerance.
   */
  bool isInterRankBalancingBad() const;


  /**
   * Is the balancing on the rank ok
   *
   * Similar to isInterRankBalancingBad(), this operation runs through a
   * series of checks:
   *
   * - If there are no trees on the rank, then the result is false. By
   *   definition, everything is well-balanced.
   * - If there is only one tree, the balancing is by definition bad, as
   *   we cannot make a balancing statement.
   * - If we host more trees than allowed, the balancing by definition is
   *   good, since we could not refine further anyway.
   * - Otherwise, we compare the minimum and maximum tree.
   */
  bool isIntraRankBalancingBad() const;


  bool areRanksUnemployed() const;

  /**
   * @return -1 if there is no local tree yet
   */
  double getWeightOfHeaviestLocalSpacetree() const;

  /**
   * Determines the maximum spacetree size a tree should have in the
   * optimal case.
   *
   * As this routine does not really adopt the blacklist, it can introduce
   * cyclles: If we repeatedly try to split the same
   * rank this means that we have tried to split it, have not been
   * successful, and tried again. This can happen, as not all trees
   * can be split. See  peano4::grid::Spacetree::isCellTopDownSplitCandidate()
   * and  peano4::grid::Spacetree::isCellBottomUpSplitCandidate()
   * for a discussion which cells can be split and which can't. As
   * not all cells can't be given away, not all trees can be split up.
   *
   * ## Randomisation
   *
   * If two or more trees have the same weight, the routine should
   * randomly return one of them.
   *
   * @return NoHeaviestTreeAvailable If there are no local trees or
   *   if the heaviest tree is on the blacklist, i.e. we have to
   *   assume that it still is splitting.
   */
  int getIdOfHeaviestLocalSpacetree() const;


  /**
   * Similar to getIdOfHeaviestLocalSpacetree() but you might get one of
   * the trees back that is close to the heaviest one up to tolerance.
   * Tolerance here is the relative difference: So you are guaranteed
   * that
   *
   * @f$ m_i \geq tolerance \cdot m_{max} @f$
   *
   * or
   *
   * @f$ \frac{m_i}{m_{max}} \geq tolerance @f$
   *
   * @param tolerance Relative weight a tree has to have compared to the
   *   heaviest local tree to be a candidate. By default, you get the last
   *   heaviest tree. But you can ask for a random tree close to it.
   */
  int getIdOfHeaviestLocalSpacetree(double tolerance) const;
};

std::ostream& operator<<(std::ostream& out, const toolbox::loadbalancing::AbstractLoadBalancing& balancing);
