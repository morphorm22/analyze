/*
 * NitscheNonlinearThermoMechanics.cpp
 *
 *  Created on: July 14, 2023
 */

#include "elliptic/thermomechanics/nonlinear/nitsche/NitscheNonlinearThermoMechanics_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/thermomechanics/nonlinear/nitsche/NitscheNonlinearThermoMechanics_def.hpp"

#include "ThermalElement.hpp"
#include "MechanicsElement.hpp"
#include "element/ThermoElasticElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::NitscheNonlinearThermoMechanics, Plato::ThermalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::NitscheNonlinearThermoMechanics, Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::NitscheNonlinearThermoMechanics, Plato::ThermoElasticElement)

#endif