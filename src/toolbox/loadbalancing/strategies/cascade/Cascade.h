// This file is part of the Peano project. For conditions of distribution and
// use, please see the copyright notice at www.peano-framework.org
#pragma once

#include "toolbox/loadbalancing/AbstractLoadBalancing.h"
#include "toolbox/loadbalancing/State.h"
#include "toolbox/loadbalancing/metrics/CellCount.h"

namespace toolbox {
  namespace loadbalancing {
    namespace strategies {
      namespace cascade {
        template<typename HostedLoadBalancing0, typename HostedLoadBalancing1>
        class Cascade;
      }
    }
  }
}


/**
 * Cascade of load balancing schemes
 *
 * We run through a series of load balancing schemes. If one is stagnating or
 * switching off, we switch to the next one.
 *
 * I tried to play around with variadic templates, but that did not work for
 * multiple reasons, so I went down the manual way.
 */
template<typename HostedLoadBalancing0, typename HostedLoadBalancing1>
class toolbox::loadbalancing::strategies::cascade::Cascade: public toolbox::loadbalancing::AbstractLoadBalancing {
  public:
    Cascade(
        Configuration* configuration = new DefaultConfiguration(),
        CostMetrics*   costMetrics   = new toolbox::loadbalancing::metrics::CellCount()
    ):
      AbstractLoadBalancing( configuration, costMetrics),
      _hostedLoadBalancing0(configuration, costMetrics),
      _hostedLoadBalancing1(configuration, costMetrics),
      _activeLoadBalancing(0) {
    }


    /**
     * We have piped through any pointers to a configuration and the cost
     * metrics. Now we set these pointers manually to nullptr, so no piped
     * through attribute is actually deleted. Instead, we trust on the
     * current class' supertype to actually destroy object.
     */
    virtual ~Cascade() {
      _hostedLoadBalancing0.setConfigurationAndMetricsNullWithoutDelete();
      _hostedLoadBalancing1.setConfigurationAndMetricsNullWithoutDelete();
    }


    /**
     * Inform the active aggregate about finishStep() and then check
     * afterwards if the active load balancing has stagnated. If so,
     * switch to the next one in the cascade.
     */
    virtual void finishStep() override {
      switch (_activeLoadBalancing) {
        case 0:
          {
            _hostedLoadBalancing0.finishStep();
            if (_hostedLoadBalancing0.hasStagnated()) {
              _activeLoadBalancing++;
              logInfo( "finishStep()", "one load balancing scheme has stagnated, to switch to " << _activeLoadBalancing );
            }
          }
          break;
        case 1:
          {
            _hostedLoadBalancing1.finishStep();
          }
          break;
        default:
          assertion1( false, _activeLoadBalancing );
          break;
      }
    }


    virtual bool hasMeshBeenStationary() const override {
      switch (_activeLoadBalancing) {
        case 0:
          return _hostedLoadBalancing0.hasMeshBeenStationary();
        case 1:
          return _hostedLoadBalancing1.hasMeshBeenStationary();
        default:
          assertion1(false,_activeLoadBalancing);
          break;
      }
      return false;
    }


    virtual bool hasSplitRecently() const override {
      switch (_activeLoadBalancing) {
        case 0:
          return _hostedLoadBalancing0.hasSplitRecently();
        case 1:
          return _hostedLoadBalancing1.hasSplitRecently();
        default:
          assertion1(false,_activeLoadBalancing);
          break;
      }
      return false;
    }


    virtual int getGlobalNumberOfTrees() const override {
      switch (_activeLoadBalancing) {
        case 0:
          return _hostedLoadBalancing0.getGlobalNumberOfTrees();
        case 1:
          return _hostedLoadBalancing1.getGlobalNumberOfTrees();
        default:
          assertion1(false,_activeLoadBalancing);
          break;
      }
      return -1;
    }


    virtual bool hasStagnated() const override {
      return _state == State::SwitchedOff
          or (
            _activeLoadBalancing == 1
            and
            _hostedLoadBalancing1.hasStagnated()
          );
    }


    virtual void finishSimulation() override {
      switch (_activeLoadBalancing) {
        case 0:
          _hostedLoadBalancing0.finishSimulation();
          break;
        case 1:
          _hostedLoadBalancing1.finishSimulation();
          break;
        default:
          assertion1(false,_activeLoadBalancing);
          break;
      }
    }


    virtual void enable(bool value) override {
      _hostedLoadBalancing0.enable(value);
      _hostedLoadBalancing1.enable(value);
      AbstractLoadBalancing::enable(value);
      if (_activeLoadBalancing==0) {
        logInfo( "enable(bool)", "lb0=" << _hostedLoadBalancing0.toString() );
      }
      else {
        logInfo( "enable(bool)", "lb1=" << _hostedLoadBalancing1.toString() );
      }
    }

  private:
    static tarch::logging::Log  _log;

    HostedLoadBalancing0  _hostedLoadBalancing0;
    HostedLoadBalancing1  _hostedLoadBalancing1;

    /**
     * Pick the active load balancing.
     */
    int                   _activeLoadBalancing;
};


template<typename HostedLoadBalancing0, typename HostedLoadBalancing1>
tarch::logging::Log toolbox::loadbalancing::strategies::cascade::Cascade<HostedLoadBalancing0, HostedLoadBalancing1>::_log( "toolbox::loadbalancing::strategies::cascade::Cascade" );
