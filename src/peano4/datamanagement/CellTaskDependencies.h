// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include <set>

#include "peano4/utils/Globals.h"
#include "tarch/la/Vector.h"
#include "tarch/multicore/multicore.h"

namespace peano4 {
  namespace datamanagement {
    /**
     * @page peano4_datamanagement_CellTaskDependencies Dependencies of the tasks spawned per cell
     *
     * The producer of the producer-consumer task graph compiler
     * (peano4.api.taskgraph.compiler.CellTaskProducerConsumer) spawns the
     * touchCellFirstTime() and the touchCellLastTime() of an action set as
     * tasks. The routines here compute the dependencies of these tasks, so
     * the generated code only has to call them.
     *
     * Each cell holds the number of its first task, the number of its last
     * task and an array with 3^d entries, the children tasks. The numbers
     * are tarch::multicore::TaskEnumerator::NoNumber if there is no task.
     * The first task of a cell depends on the first task of its parent. A
     * cell reserves the number of its last task in touchCellFirstTime() and
     * writes it into the entry of its parent's array which belongs to its
     * position within the parent. The parent's touchCellLastTime() is invoked
     * after all children have been visited, i.e. it finds all numbers in the
     * array. Its last task depends on its own first task and on the last
     * tasks of all children, i.e. it runs after the cell's whole subtree.
     * Why the last number is reserved so early: the child tells the parent
     * about a task which is spawned only once the child has left.
     *
     * If an action set spawns several tasks per cell, the first task depends
     * on the first task of the previous action set, and the last task on the
     * last task of the previous action set. This preserves the order of the
     * action sets.
     */

    /**
     * Position of a cell within the array of children tasks of its parent
     *
     * @param relativePositionWithinParent Marker's getRelativePositionWithinFatherCell()
     * @return Value between 0 and 3^d-1, or -1 if the position is not a valid
     *   position within a parent (this is the case for the root cell, whose
     *   "parent" is a dummy).
     */
    int getIndexWithinParent(const tarch::la::Vector<Dimensions, int>& relativePositionWithinParent);

    /**
     * Set all entries of the children tasks to tarch::multicore::TaskEnumerator::NoNumber
     *
     * @param setChildTask Functor with signature void(int index, int value),
     *   typically a setter of the cell's task label.
     */
    template <typename SetChildTask>
    void clearChildrenTasks(SetChildTask setChildTask);

    /**
     * In-dependencies of the task spawned by touchCellFirstTime()
     *
     * @param previousFirstTask First task of the previous action set of the same producer
     *   or tarch::multicore::TaskEnumerator::NoNumber.
     * @param parentFirstTask First task of the parent cell (NoNumber if there is none,
     *   e.g. for the root cell, as the label of its dummy parent holds NoNumber).
     * @return The valid numbers of the arguments.
     */
    std::set<tarch::multicore::TaskNumber> getInDependenciesOfFirstTask(int previousFirstTask, int parentFirstTask);

    /**
     * In-dependencies of the task spawned by touchCellLastTime()
     *
     * @param firstTask First task of the cell itself (or NoNumber)
     * @param previousLastTask Last task of the previous action set of the same
     *   producer (or NoNumber)
     * @param getChildTask Functor with signature int(int index) which returns
     *   the last task of the child at the index.
     * @return The valid numbers of the arguments and of all children.
     */
    template <typename GetChildTask>
    std::set<tarch::multicore::TaskNumber> getInDependenciesOfLastTask(int firstTask, int previousLastTask, GetChildTask getChildTask);

    /**
     * Spawn a task which does nothing
     *
     * The parent waits for the number of the last task of each child.
     * If the guard of the last task does not hold, the producer still has to
     * spawn something under this number.
     */
    void spawnEmptyTask(const std::set<tarch::multicore::TaskNumber>& inDependencies, tarch::multicore::TaskNumber taskNumber);
  } // namespace datamanagement
} // namespace peano4

#include "CellTaskDependencies.cpph"
