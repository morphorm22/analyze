/*
 * FactoryMechanicalMaterials.cpp
 *
 *  Created on: July 13, 2023
 */

#include "materials/mechanical/FactoryMechanicalMaterials_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "materials/mechanical/FactoryMechanicalMaterials_def.hpp"

#include "ThermalElement.hpp"
#include "MechanicsElement.hpp"
#include "element/ThermoElasticElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::FactoryMechanicalMaterials,Plato::ThermalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::FactoryMechanicalMaterials,Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::FactoryMechanicalMaterials,Plato::ThermoElasticElement)

#endif