/*
 * Plato_AugLagStressCriterionGeneral.cpp
 *
 *  Created on: Apr 2, 2019
 */

#include "Plato_AugLagStressCriterionGeneral_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "Plato_AugLagStressCriterionGeneral_def.hpp"

#include "MechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::AugLagStressCriterionGeneral, Plato::MechanicsElement)

#endif
