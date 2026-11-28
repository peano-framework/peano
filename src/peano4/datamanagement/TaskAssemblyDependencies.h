// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "peano4/utils/Globals.h"
#include "tarch/la/Vector.h"
#include "tarch/multicore/multicore.h"
#include "tarch/multicore/Task.h"
#include "tarch/multicore/TaskEnumerator.h"

#include <initializer_list>
#include <set>

namespace peano4 {
  namespace datamanagement {
    namespace taskassembly {
      /**
       * @page peano4_datamanagement_TaskAssemblyDependencies Task numbers of
       * the task graph assembly
       *
       * The task graph compiler peano4.api.taskgraph.compiler.TaskGraphAssembly
       * wraps the six events touchVertexFirstTime(), touchVertexLastTime(),
       * touchFaceFirstTime(), touchFaceLastTime(), touchCellFirstTime() and
       * touchCellLastTime() of an action set into tasks. Each entity (vertex,
       * face, cell) reserves two task numbers per wrapped action set in its
       * first touch: one for the task of the first and one for the task of the
       * last event. The routines here assemble the in-dependencies of these
       * tasks from the numbers which the entities hold, so the generated code
       * only has to call them. All routines ignore numbers which are
       * tarch::multicore::TaskEnumerator::NoNumber.
       *
       * ## Dependencies
       *
       * - Within an entity: The tasks of the wrapped action sets follow each
       *   other (latestTask()), and the last task depends on the first task.
       * - Parent to child (multiscale): The first task of an entity depends
       *   on the first tasks of the coarse entities which it is interpolated
       *   from.
       * - Child to parent (multiscale): A fine entity writes the number of its
       *   last task into its coarse entity (index within the parent, see
       *   getIndexWithinParentFace()). It does so in its first touch, as the
       *   coarse entity's last touch is invoked after all fine entities have
       *   been visited, but the fine entity's last task is spawned only
       *   afterwards. The coarse entity's last task depends on these numbers.
       * - Across entities of one level: The first task of a cell depends on
       *   the first tasks of its faces and vertices, the first task of a face
       *   on the first tasks of its vertices. The cell writes the number of
       *   its latest task into the vertices and faces around it (index
       *   getIndexOfCellAroundVertex() and getIndexOfCellAroundFace()). The
       *   last task of the vertex and of the face depends on these numbers.
       */

      /**
       * The one task enumerator of all task numbers of the task graph assembly
       *
       * Several tarch::multicore::TaskEnumerator instances do not hand out
       * disjoint numbers, i.e. the vertex tasks, the face tasks and the cell
       * tasks share this one instance. A task enumerator per label type would
       * give the task of a face the number of the task of a cell, and the two
       * would wait for each other. The instance is created on first use.
       */
      inline tarch::multicore::TaskEnumerator& getTaskEnumerator() {
        static tarch::multicore::TaskEnumerator enumerator;
        return enumerator;
      }

      /**
       * Add a task number to a dependency set
       *
       * @param inDependencies Is extended by number unless the latter is
       * NoNumber
       */
      inline void addTask(
        std::set<tarch::multicore::TaskNumber>& inDependencies,
        int                                     number
      ) {
        if (number >= 0) {
          inDependencies.insert(number);
        }
      }

      /**
       * Add all valid numbers of a range
       *
       * @param count Number of entries
       * @param getTask Functor with signature int(int index), returns the
       * number of the i-th entry
       */
      template <typename GetTask>
      void addTasks(
        std::set<tarch::multicore::TaskNumber>& inDependencies,
        int                                     count,
        GetTask                                 getTask
      ) {
        for (int i = 0; i < count; i++) {
          addTask(inDependencies, getTask(i));
        }
      }

      /**
       * Return the first valid number of a list
       *
       * The generated code passes the numbers of the wrapped action sets,
       * the latest one first. As the tasks of the action sets follow each
       * other, the first valid number identifies everything that happened
       * before.
       */
      inline int latestTask(std::initializer_list<int> numbersLatestFirst) {
        for (int number : numbersLatestFirst) {
          if (number >= 0) {
            return number;
          }
        }
        return tarch::multicore::TaskEnumerator::NoNumber;
      }

      /**
       * Wait for a task, return its number to the enumerator and invalidate it
       */
      inline void waitForAndReleaseTask(
        tarch::multicore::TaskEnumerator& enumerator,
        int&                              number
      ) {
        if (number >= 0) {
          tarch::multicore::waitForTask(number);
          enumerator.releaseNumber(number);
          number = tarch::multicore::TaskEnumerator::NoNumber;
        }
      }

      /**
       * Spawn a task which does nothing
       *
       * The numbers of last tasks are announced to the coarse entities before
       * we know whether the task is spawned. If the guard of the last event
       * does not hold, we still have to spawn something under this number.
       */
      inline void spawnEmptyTask(
        const std::set<tarch::multicore::TaskNumber>& inDependencies,
        tarch::multicore::TaskNumber                  taskNumber
      ) {
        tarch::multicore::spawnTask(
          new tarch::multicore::TaskWithCopyOfFunctor(
            tarch::multicore::Task::DontFuse,
            tarch::multicore::Task::DefaultPriority,
            []() mutable {}
          ),
          inDependencies,
          taskNumber
        );
      }

      /**
       * Slot of a cell in the array of the cells around a vertex
       *
       * The vertex with the number vertexWithinCell of a cell sees this cell
       * from the opposite corner.
       *
       * @param vertexWithinCell Number of the vertex within the cell, from 0 to
       * 2^d-1
       */
      inline int getIndexOfCellAroundVertex(int vertexWithinCell) {
        return TwoPowerD - 1 - vertexWithinCell;
      }

      /**
       * Slot of a cell in the array of the two cells around a face
       *
       * Faces 0 to d-1 of a cell are its left faces, faces d to 2d-1 its right
       * faces. The two cells which share a face thus see it from different
       * sides.
       *
       * @param faceWithinCell Number of the face within the cell, from 0 to
       * 2d-1
       */
      inline int getIndexOfCellAroundFace(int faceWithinCell) {
        return faceWithinCell < Dimensions ? 0 : 1;
      }

      /**
       * Slot of a fine face within the array of fine faces of its coarse face
       *
       * @param relativePositionWithinFatherFace Marker's
       * getRelativePositionWithinFatherFace()
       * @return Value between 0 and 3^(d-1)-1, or -1 if the position is not
       * valid
       */
      inline int getIndexWithinParentFace(
        const tarch::la::Vector<Dimensions - 1, int>&
          relativePositionWithinFatherFace
      ) {
        int result = 0;
        int base   = 1;
        for (int d = 0; d < Dimensions - 1; d++) {
          if (relativePositionWithinFatherFace(d) < 0 or relativePositionWithinFatherFace(d) >= 3) {
            return -1;
          }
          result += relativePositionWithinFatherFace(d) * base;
          base *= 3;
        }
        return result;
      }

      /**
       * Slot of a vertex's coarse vertex within the 2^d coarse vertices
       *
       * Only meaningful if the vertex coincides with a coarse vertex.
       *
       * @param relativePositionWithinFatherCell Marker's
       * getRelativePositionWithinFatherCell(), entries from 0 to 3
       */
      inline int getIndexOfCoincidingCoarseVertex(
        const tarch::la::Vector<Dimensions, int>&
          relativePositionWithinFatherCell
      ) {
        int result = 0;
        int base   = 1;
        for (int d = 0; d < Dimensions; d++) {
          result += (relativePositionWithinFatherCell(d) / 3) * base;
          base *= 2;
        }
        return result;
      }

      /**
       * Slot of a cell within the array of the children of its parent
       *
       * @param relativePositionWithinParent Marker's
       * getRelativePositionWithinFatherCell()
       * @return Value between 0 and 3^d-1, or -1 if the position is not a
       *   valid position within a parent (the root cell).
       */
      inline int getIndexWithinParentCell(
        const tarch::la::Vector<Dimensions, int>& relativePositionWithinParent
      ) {
        int result = 0;
        int base   = 1;
        for (int d = 0; d < Dimensions; d++) {
          if (relativePositionWithinParent(d) < 0 or relativePositionWithinParent(d) >= 3) {
            return -1;
          }
          result += relativePositionWithinParent(d) * base;
          base *= 3;
        }
        return result;
      }
    } // namespace taskassembly
  }   // namespace datamanagement
} // namespace peano4
