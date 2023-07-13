/*
 *  BoundaryEvaluatorTrialKirchhoffStress_decl.hpp
 *
 *  Created on: July 13, 2023
 */

#include "elliptic/mechanical/nonlinear/nitsche/BoundaryEvaluatorTrialKirchhoffStress_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/mechanical/nonlinear/nitsche/BoundaryEvaluatorTrialKirchhoffStress_def.hpp"

#include "ThermalElement.hpp"
#include "MechanicsElement.hpp"
#include "element/ThermoElasticElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::BoundaryEvaluatorTrialKirchhoffStress, Plato::ThermalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::BoundaryEvaluatorTrialKirchhoffStress, Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::BoundaryEvaluatorTrialKirchhoffStress, Plato::ThermoElasticElement)

#endif