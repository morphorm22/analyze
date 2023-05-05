/*
 * AugLagStrengthCriterion.cpp
 *
 *  Created on: May 5, 2023
 */

#include "AugLagStrengthCriterion_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "AugLagStrengthCriterion_def.hpp"

#include "MechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::AugLagStrengthCriterion, Plato::MechanicsElement)

#endif