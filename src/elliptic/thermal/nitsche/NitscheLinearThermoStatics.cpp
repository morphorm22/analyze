/*
 * NitscheLinearThermoStatics.cpp
 *
 *  Created on: July 14, 2023
 */

#include "elliptic/thermal/nitsche/NitscheLinearThermoStatics_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/thermal/nitsche/NitscheLinearThermoStatics_def.hpp"

#include "ThermalElement.hpp"
#include "MechanicsElement.hpp"
#include "element/ThermoElasticElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::NitscheLinearThermoStatics, Plato::ThermalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::NitscheLinearThermoStatics, Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::NitscheLinearThermoStatics, Plato::ThermoElasticElement)

#endif