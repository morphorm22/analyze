/*
 * FactoryElasticMaterial.cpp
 *
 *  Created on: July 13, 2023
 */

#include "materials/mechanical/FactoryElasticMaterial_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "materials/mechanical/FactoryElasticMaterial_def.hpp"

#include "ThermalElement.hpp"
#include "MechanicsElement.hpp"
#include "element/ThermoElasticElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::FactoryElasticMaterial,Plato::ThermalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::FactoryElasticMaterial,Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::FactoryElasticMaterial,Plato::ThermoElasticElement)

#endif