/*
 * BoundaryEvaluatorTrialIsotropicElasticStress.cpp
 *
 *  Created on: July 13, 2023
 */

#include "elliptic/mechanical/linear/nitsche/BoundaryEvaluatorTrialIsotropicElasticStress_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/mechanical/linear/nitsche/BoundaryEvaluatorTrialIsotropicElasticStress_def.hpp"

#include "ThermalElement.hpp"
#include "MechanicsElement.hpp"
#include "element/ThermoElasticElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::BoundaryEvaluatorTrialIsotropicElasticStress, Plato::ThermalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::BoundaryEvaluatorTrialIsotropicElasticStress, Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::BoundaryEvaluatorTrialIsotropicElasticStress, Plato::ThermoElasticElement)

#endif