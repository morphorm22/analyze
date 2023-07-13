/*
 * NitscheTrialElasticStressEvaluator.cpp
 *
 *  Created on: July 13, 2023
 */

#include "elliptic/mechanical/linear/nitsche/NitscheTrialElasticStressEvaluator_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/mechanical/linear/nitsche/NitscheTrialElasticStressEvaluator_def.hpp"

#include "ThermalElement.hpp"
#include "MechanicsElement.hpp"
#include "element/ThermoElasticElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::NitscheTrialElasticStressEvaluator, Plato::ThermalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::NitscheTrialElasticStressEvaluator, Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::NitscheTrialElasticStressEvaluator, Plato::ThermoElasticElement)

#endif