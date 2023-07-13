/*
 *  BoundaryEvaluatorTestNeoHookeanStress.cpp
 *
 *  Created on: July 13, 2023
 */

#include "elliptic/mechanical/nonlinear/nitsche/BoundaryEvaluatorTestNeoHookeanStress_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/mechanical/nonlinear/nitsche/BoundaryEvaluatorTestNeoHookeanStress_def.hpp"

#include "ThermalElement.hpp"
#include "MechanicsElement.hpp"
#include "element/ThermoElasticElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::BoundaryEvaluatorTestNeoHookeanStress, Plato::ThermalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::BoundaryEvaluatorTestNeoHookeanStress, Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::BoundaryEvaluatorTestNeoHookeanStress, Plato::ThermoElasticElement)

#endif