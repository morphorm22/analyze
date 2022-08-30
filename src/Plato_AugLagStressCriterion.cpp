/*
 * Plato_AugLagStressCriterion.cpp
 *
 *  Created on: Apr 2, 2019
 */

#include "Plato_AugLagStressCriterion_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "Plato_AugLagStressCriterion_def.hpp"

#include "MechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::AugLagStressCriterion, Plato::MechanicsElement)

#endif
