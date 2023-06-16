/*
 * FactoryStressEvaluator.cpp
 *
 *  Created on: May 31, 2023
 */

#include "elliptic/mechanical/nonlinear/FactoryStressEvaluator_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/mechanical/nonlinear/FactoryStressEvaluator_def.hpp"

#include "MechanicsElement.hpp"
#include "element/ThermoElasticElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::FactoryStressEvaluator,Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::FactoryStressEvaluator,Plato::ThermoElasticElement)

#endif