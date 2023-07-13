/*
 *  BoundaryEvaluatorTrialNeoHookeanStress.cpp
 *
 *  Created on: July 13, 2023
 */

#include "elliptic/mechanical/nonlinear/nitsche/BoundaryEvaluatorTrialNeoHookeanStress_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/mechanical/nonlinear/nitsche/BoundaryEvaluatorTrialNeoHookeanStress_def.hpp"

#include "ThermalElement.hpp"
#include "MechanicsElement.hpp"
#include "element/ThermoElasticElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::BoundaryEvaluatorTrialNeoHookeanStress, Plato::ThermalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::BoundaryEvaluatorTrialNeoHookeanStress, Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::BoundaryEvaluatorTrialNeoHookeanStress, Plato::ThermoElasticElement)

#endif