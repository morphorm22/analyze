/*
 * NitscheLinearMechanics.cpp
 *
 *  Created on: July 14, 2023
 */

#include "elliptic/mechanical/linear/nitsche/NitscheLinearMechanics_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/mechanical/linear/nitsche/NitscheLinearMechanics_def.hpp"

#include "ThermalElement.hpp"
#include "MechanicsElement.hpp"
#include "element/ThermoElasticElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::NitscheLinearMechanics, Plato::ThermalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::NitscheLinearMechanics, Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::NitscheLinearMechanics, Plato::ThermoElasticElement)

#endif