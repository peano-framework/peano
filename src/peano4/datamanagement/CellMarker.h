// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "peano4/grid/GridTraversalEvent.h"
#include "peano4/utils/Globals.h"
#include "tarch/la/Vector.h"

#ifdef Parallel
#include <functional>
#include <mpi.h>
#endif

namespace peano4 {
  namespace datamanagement {
    struct CellMarker;
  } // namespace datamanagement
} // namespace peano4

std::ostream& operator<<(std::ostream& out, const peano4::datamanagement::CellMarker& marker);

/**
 * Cell marker
 *
 * This object provides information on the cell aka octant within the
 * spacetree. This includes spatial
 * information such as size and position. It also comprises mesh topology
 * properties such as "will this cell be refined" or "is its parent a
 * local cell".
 *
 * It is an object which users are not supposed to manipulate. The mesh
 * traversal automaton creates it once per octant while it runs through
 * the mesh and passes in information about adjacent vertices, parent
 * data, and so forth. The constructor then distills the user
 * representation from that data and, from hereon, users can use this
 * distilled information to make logical decisions what to do and how
 * to do things within the spacetree octant.
 */
struct peano4::datamanagement::CellMarker {
private:
  tarch::la::Vector<Dimensions, double> _centre;

  tarch::la::Vector<Dimensions, double> _h;

  bool _hasBeenRefined;
  bool _willBeRefined;
  bool _isLocal;
  bool _isParentLocal;
  bool _areAllVerticesRefined;
  bool _isOneVertexHanging;
  bool _isOneVertexCreatedOrDestroyed;

  bool _areAllVerticesInsideDomain;
  bool _invokingSpacetreeIsNotInvolvedInAnyDynamicLoadBalancing;

  bool _willBeEnclave;
  bool _hasBeenEnclave;

  /**
   * Entries from (0,1,2). (0,0) or (0,0,0) is the left, bottom cell.
   */
  tarch::la::Vector<Dimensions, int> _relativePositionOfCellWithinFatherCell;

public:
  CellMarker(const peano4::grid::GridTraversalEvent& event);

  /**
   * Has the cell been refined when we kicked off this mesh traversal
   *
   * If the attribute holds, this implies that the cell is refined in this
   * sweep, as cells do not change their state throughout the traversal. It
   * does not mean that the cell will be refined in the next mesh traversal
   * however. To find this out, you have to study willBeRefined().
   *
   * @see willBeRefined()
   */
  bool hasBeenRefined() const;

  /**
   * Will the cell be refined in the subsequent iteration
   *
   *
   * If hasBeenRefined()
   * returns false but willBeRefined() holds, we have a cell which Peano 4
   * will refine in the current mesh sweep. It might however
   * not yet be refined. This analysis makes sense once we take into account that
   * refinement in Peano first have to be triggered. In the subsequent mesh
   * traversal, the affected vertices then switch to refining and the new
   * mesh elements are actually added.
   *
   * The function which ultimately defines if this flag is set or not is
   * peano4::grid::willVerticesBeRefined().
   *
   * If willBeRefined() is false but hasBeenRefined() is true, the current
   * cell is refined, but it will be coarsened in the very mesh traversal in
   * which we run through the mesh.
   */
  bool willBeRefined() const;

  /**
   * Centre of a cell
   *
   * If you want to know the position of one of the cell's vertices, you have
   * to reconstruct this one manually. It is particularly simple if you already
   * know the number of the vertex as a bitset:
   *
   * ~~~~~~~~~~~~~~~~~~~~~~~~~
   * marker.x() - 0.5 * marker.h() + tarch::la::multiplyComponents(tarch::la::Vector<Dimensions,double>(targetVertex),marker.h())
   * ~~~~~~~~~~~~~~~~~~~~~~~~~
   *
   *
   * @return Centre of cell.
   */
  tarch::la::Vector<Dimensions, double> x() const;

  /**
   * See getInvokingCellCentre(). This routine gives you the centre of its
   * parent cell.
   */
  tarch::la::Vector<Dimensions, double> getInvokingParentCellsCentre() const;

  /**
   * Is x contained within cell identified by marker object
   *
   * Check whether the point x is contained within the marker. This is an
   * operation which works with smaller equals and greater equals and
   * numerical tolerances. Therefore, the routine includes any x that is
   * exactly on a cell boundary. The other way round: a particle for
   * example that resides exactly on the cell boundary, subject to the
   * chosen floating point precision in the tarch::la library, will be
   * considered to belong both to its left and its right neighbour.
   *
   * On top of the fact that we work with closed intervals, it is
   * important to realise that we work with floating point comparisons
   * from the tarch::la library. Therefore, we employ some built-in
   * floating point precision. You might want to switch to another precision.
   * Most popular is a relative size that takes the dimensions of the marker
   * into account:
   *
   *        marker.isContained(
   *          p->getX(), tarch::la::relativeEps( marker.h()(0) )
   *        )
   *
   * This variant uses relativeEps() as defined in tarch/la/Scalar.h.
   *
   *
   * @see   The routine calls the static isContained() routine and passes the
   *        local attributes in.
   *
   * @param tolerance Absolute tolerance when we compare two values.
   *
   */
  bool isContained(const tarch::la::Vector<Dimensions, double>& x, double tolerance = tarch::la::NUMERICAL_ZERO_DIFFERENCE) const;

  /**
   * Is point contained in cell
   *
   * Evaluate if x is in a h/2 environment around cellCentre. We use
   * tarch::la comparison functions as we operate with floating point
   * values. tolerance is the tolerance we pass into these functions.
   *
   * See the other isContained() routine for a description, but keep in
   * mind that this other routine refers to this place, i.e. this routine
   * provides the actual behaviour.
   *
   * @param x          Point to query
   * @param cellCentre Centre of cell of interest
   * @param h          Mesh size on this level
   * @param tolerance  Search tolerance
   *
   */
  static bool isContained(
    const tarch::la::Vector<Dimensions, double>&   x,
    const tarch::la::Vector<Dimensions, double>&   cellCentre,
    const tarch::la::Vector<Dimensions, double>&   h,
    double                                         tolerance
  );

  bool overlaps(const tarch::la::Vector<Dimensions, double>& offset, const tarch::la::Vector<Dimensions, double>& size) const;

  /**
   * @return Size of cell
   */
  tarch::la::Vector<Dimensions, double> h() const;

  /**
   * @return Offset of cell, i.e. the bottom left vertex's coordinate
   */
  tarch::la::Vector<Dimensions, double> getOffset() const;

  std::string toString() const;

  /**
   * Usually if you have an event of a cell, then the cell is alo local.
   * Otherwise, an event not be called.
   */
  bool isLocal() const;

  /**
   * A cell can be local and its parent still might not be local. This happens
   * if we have a horizontal tree cut. It also happens on the very top of the
   * spacetree: Level 0 is by definition not persistent and thus is not
   * considered to be local.
   *
   * @see isLocal()
   */
  bool isParentLocal() const;

  /**
   * Define enclave cell
   *
   * A enclave cell in the definition of Charrier, Hazelwood, Weinzierl is a
   * cell that is not a skeleton cell. A skeleton cell is cell which either
   *
   * - is adjacent to a resolution transitions; or
   * - is adjacent to a domain boundary.
   *
   * Enclave cells are cells that you can potentially (if your algorithm
   * allows) deploy to a task running in the background. The only thing you
   * have to do is to ensure that it is passed all data via firstprivate copy
   * and that you never refine an enclave cell.
   *
   * ## Dynamic load balancing
   *
   * In line with the above definition, we may label a cell as enclave cell
   * if some dynamic load balancing is triggered or actually running. In
   * this case, any enclave cell might be subject of a domain transfer. If
   * we deployed its computation to the background, we could move around a
   * cell whose computations is done by a thread currently.
   *
   * @see page_exahype_solvers_enclave_solvers for a high-level description
   */
  bool hasBeenEnclaveCell() const;

  /**
   * @see hasBeenEnclaveCell()
   */
  bool willBeEnclaveCell() const;

  /**
   * Is the cell next to a resolution transition or about to change the mesh
   *
   * True if one of the cell's vertices is hanging, created or destroyed, or if the cell is or
   * will be refined. This is the part of the definition of a skeleton cell that concerns adaptive
   * mesh refinement (see hasBeenEnclaveCell()). A task which works on such a cell touches
   * data, e.g. hanging faces, which the traversal interpolates or restricts on the
   * main thread right before or after the cell.
   */
  bool isAdjacentToResolutionTransition() const {
    return _isOneVertexHanging or _isOneVertexCreatedOrDestroyed or _hasBeenRefined or _willBeRefined;
  }

  /**
   * Is the cell next to the boundary between two trees or two ranks
   *
   * This is the part of the definition of a skeleton cell that concerns the domain decomposition (see
   * hasBeenEnclaveCell()). As every cell belongs to exactly one tree, the cell's own data are
   * not exchanged. The faces and vertices around the cell are.
   */
  bool isAdjacentToParallelDomainBoundary() const { return not _areAllVerticesInsideDomain; }

  /**
   * A skeleton cell is a not-enclave cell
   *
   * @see hasBeenEnclaveCell()
   */
  bool hasBeenSkeletonCell() const;

  /**
   * @see hasBeenEnclaveCell()
   */
  bool willBeSkeletonCell() const;

  tarch::la::Vector<Dimensions, int> getRelativePositionWithinFatherCell() const;

#if PeanoDebug > 0
  /**
   * Used for debugging
   */
  void setRelativePositionWithinFatherCell(int axis, int value);
#endif

#ifdef Parallel
  /**
   * To be called prior to any MPI usage of this class.
   */
  static void initDatatype();
  static void shutdownDatatype();

  /**
   * In DaStGen (the first version), I had a non-static version of the send
   * as well as the receive. However, this did not work with newer C++11
   * versions, as a member function using this as pointer usually doesn't
   * see the vtable while the init sees the object from outside, i.e.
   * including a vtable. So this routine now is basically an alias for a
   * blocking MPI_Send.
   */
  static void send(const CellMarker& buffer, int destination, int tag, MPI_Comm communicator);
  static void receive(CellMarker& buffer, int source, int tag, MPI_Comm communicator);

  /**
   * Alternative to the other send() where I trigger a non-blocking send an
   * then invoke the functor until the corresponding MPI_Test tells me that
   * the message went through. In systems with heavy MPI usage, this can
   * help to avoid deadlocks.
   */
  static void send(const CellMarker& buffer, int destination, int tag, std::function<void()> startCommunicationFunctor, std::function<void()> waitFunctor, MPI_Comm communicator);
  static void receive(CellMarker& buffer, int source, int tag, std::function<void()> startCommunicationFunctor, std::function<void()> waitFunctor, MPI_Comm communicator);

  static void sendAndPollDanglingMessages(const CellMarker& message, int destination, int tag, MPI_Comm communicator);
  static void receiveAndPollDanglingMessages(CellMarker& message, int source, int tag, MPI_Comm communicator);

  static MPI_Datatype Datatype;
#endif
};
