/*
 * Plato_AugLagStressCriterionQuadratic.cpp
 *
 */

#include "elliptic/mechanical/linear/Plato_AugLagStressCriterionQuadratic_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/mechanical/linear/Plato_AugLagStressCriterionQuadratic_def.hpp"

#include "MechanicsElement.hpp"
#include "ThermomechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::AugLagStressCriterionQuadratic, Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::AugLagStressCriterionQuadratic, Plato::ThermomechanicsElement)

#endif
