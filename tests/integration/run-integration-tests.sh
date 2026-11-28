#!/bin/bash
# This file is part of the Peano project. For conditions of distribution and
# use, please see the copyright notice at www.peano-framework.org
#
# Run the three studies of the ExaHyPE 2 integration test with short end times,
# i.e. at most 20 time steps per run. See
# documentation/exahype2/integration-test/SKILL.md for how to read the results.
#
#   tests/integration/run-integration-tests.sh <work directory>
#
# The work directory has to be outside of the repository. Finished runs in it
# are reused, i.e. use a new directory after changes to Peano or to the end
# times below. Each study writes its output into <work directory>/<study>.log.
#
# The studies have no timeout. A watchdog hence sends SIGTERM to any simulation
# in the work directory which runs longer than RUN_TIMEOUT seconds (default:
# 900), so the study records a failure and continues. It notes this in
# watchdog.log. We use SIGTERM, as the gauge wave study takes SIGKILL for a run
# which ran out of memory.
#
# The generated code is compiled with the Intel compiler. It is taken from
# INTEL_COMPILER_DIRECTORY if this directory exists, and from the PATH otherwise.
#
# The script exits with a non-zero status if any study fails.

## End times. The runs on the finest mesh take at most 20 time steps. All meshes
## of a convergence study stop at the same time, i.e. coarser ones take fewer.
POINT_SOURCE_END_TIME=0.009
ENTROPY_WAVE_END_TIME=0.00158
GAUGE_WAVE_END_TIME=0.0185

if [ $# -ne 1 ]; then
  echo "usage: $0 <work directory outside of the repository>" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
WORK=$(realpath -m "$1")
case "$WORK/" in
  "$ROOT"/*)
    echo "the work directory $WORK is inside the repository $ROOT" >&2
    exit 2
    ;;
esac
mkdir -p "$WORK" || exit 2

INTEL_COMPILER_DIRECTORY=${INTEL_COMPILER_DIRECTORY:-/opt/intel/oneapi/compiler/2024.2/bin}
if [ -d "$INTEL_COMPILER_DIRECTORY" ]; then
  export PATH="$INTEL_COMPILER_DIRECTORY:$PATH"
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RUN_TIMEOUT=${RUN_TIMEOUT:-900}

watchdog() {
  while true; do
    for process in /proc/[0-9]*; do
      executable=$(readlink "$process/exe" 2>/dev/null)
      case "$executable" in
        "$WORK"/*/gauge-wave | "$WORK"/*/smooth-convergence-test | "$WORK"/*/point-source)
          pid=${process#/proc/}
          age=$(ps -o etimes= -p "$pid" | tr -d ' ')
          if [ -n "$age" ] && [ "$age" -gt "$RUN_TIMEOUT" ]; then
            echo "$(date +%T) SIGTERM after ${age}s: $executable" >>"$WORK/watchdog.log"
            kill "$pid"
          fi
          ;;
      esac
    done
    sleep 30
  done
}
watchdog &
trap "kill $! 2>/dev/null" EXIT

FAILED=0

## Run one study: name, script (relative to benchmarks/), further arguments
study() {
  local name=$1 script=$2
  shift 2
  echo "$(date +%T) $name, log in $WORK/$name.log"
  python3 -u "$ROOT/benchmarks/$script" --work-directory "$WORK/$name" "$@" \
    >"$WORK/$name.log" 2>&1
  local status=$?
  echo "$(date +%T) $name exit $status"
  [ $status -eq 0 ] || FAILED=1
}

# The studies run one after another, as each of them uses all cores
cd "$WORK" || exit 2
study point-source exahype2/euler/point-source/amr-study.py \
  --end-time $POINT_SOURCE_END_TIME
study entropy-wave exahype2/euler/smooth-convergence-test/convergence-study.py \
  --end-time $ENTROPY_WAVE_END_TIME
study gauge-wave exagrype/gauge-wave/convergence-study.py \
  --end-time $GAUGE_WAVE_END_TIME --output-directory "$WORK/gauge-wave/out"

exit $FAILED
