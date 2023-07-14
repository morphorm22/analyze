/*
 * NitscheTempMisfitEvaluator.cpp
 *
 *  Created on: July 14, 2023
 */

#include "elliptic/thermal/nitsche/NitscheTempMisfitEvaluator_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/thermal/nitsche/NitscheTempMisfitEvaluator_def.hpp"

#include "ThermalElement.hpp"
#include "MechanicsElement.hpp"
#include "element/ThermoElasticElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::NitscheTempMisfitEvaluator, Plato::ThermalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::NitscheTempMisfitEvaluator, Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::NitscheTempMisfitEvaluator, Plato::ThermoElasticElement)

#endif