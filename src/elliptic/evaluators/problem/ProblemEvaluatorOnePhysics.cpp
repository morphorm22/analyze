/*
 * ProblemEvaluatorOnePhysics.cpp
 *
 *  Created on: June 21, 2023
 */

#include "elliptic/evaluators/problem/ProblemEvaluatorOnePhysics_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/evaluators/problem/ProblemEvaluatorOnePhysics_def.hpp"

#include "elliptic/mechanical/linear/Mechanics.hpp"
#include "elliptic/thermal/Thermal.hpp"
#include "elliptic/thermomechanics/linear/Thermomechanics.hpp"
#include "elliptic/electrical/Electrical.hpp"
#include "elliptic/electromechanics/Electromechanics.hpp"
#include "BaseExpInstMacros.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::ProblemEvaluatorOnePhysics, Plato::Elliptic::Linear::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::ProblemEvaluatorOnePhysics, Plato::Elliptic::Linear::Electrical)
PLATO_ELEMENT_DEF(Plato::Elliptic::ProblemEvaluatorOnePhysics, Plato::Elliptic::Linear::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::ProblemEvaluatorOnePhysics, Plato::Elliptic::Linear::Electromechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::ProblemEvaluatorOnePhysics, Plato::Elliptic::Linear::Mechanics)

#endif