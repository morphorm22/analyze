/*
 * CriterionAugLagStrength.cpp
 *
 *  Created on: May 5, 2023
 */

#include "elliptic/mechanical/linear/CriterionAugLagStrength_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/mechanical/linear/CriterionAugLagStrength_def.hpp"

#include "MechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::CriterionAugLagStrength, Plato::MechanicsElement)

#endif