/*
 * NitscheTrialThermalHyperElasticStressEvaluator.cpp
 *
 *  Created on: July 13, 2023
 */

#include "elliptic/thermomechanics/nonlinear/nitsche/NitscheTrialThermalHyperElasticStressEvaluator_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/thermomechanics/nonlinear/nitsche/NitscheTrialThermalHyperElasticStressEvaluator_def.hpp"

#include "ThermalElement.hpp"
#include "MechanicsElement.hpp"
#include "element/ThermoElasticElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::NitscheTrialThermalHyperElasticStressEvaluator,Plato::ThermalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::NitscheTrialThermalHyperElasticStressEvaluator,Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::NitscheTrialThermalHyperElasticStressEvaluator,Plato::ThermoElasticElement)

#endif