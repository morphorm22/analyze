/*
 * FactoryNitscheEvaluator.cpp
 *
 *  Created on: July 14, 2023
 */

#include "elliptic/evaluators/nitsche/FactoryNitscheEvaluator_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/evaluators/nitsche/FactoryNitscheEvaluator_def.hpp"

#include "ThermalElement.hpp"
#include "MechanicsElement.hpp"
#include "element/ThermoElasticElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::FactoryNitscheEvaluator, Plato::ThermalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::FactoryNitscheEvaluator, Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::FactoryNitscheEvaluator, Plato::ThermoElasticElement)

#endif