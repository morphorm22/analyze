/*
 * ResidualThermoElastoStaticTotalLagrangian.cpp
 *
 *  Created on: June 17, 2023
 */

#include "elliptic/mechanical/nonlinear/ResidualThermoElastoStaticTotalLagrangian_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/mechanical/nonlinear/ResidualThermoElastoStaticTotalLagrangian_def.hpp"

#include "MechanicsElement.hpp"
#include "element/ThermoElasticElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::ResidualThermoElastoStaticTotalLagrangian,Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::ResidualThermoElastoStaticTotalLagrangian,Plato::ThermoElasticElement)

#endif