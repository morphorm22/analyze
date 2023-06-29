/*
 * ProblemEvaluatorThermoMechanics.cpp
 *
 *  Created on: June 27, 2023
 */

#include "elliptic/evaluators/problem/ProblemEvaluatorThermoMechanics_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/evaluators/problem/ProblemEvaluatorThermoMechanics_def.hpp"

#include "elliptic/thermal/Thermal.hpp"
#include "elliptic/electrical/Electrical.hpp"
#include "elliptic/mechanical/linear/Mechanics.hpp"
#include "elliptic/electromechanics/Electromechanics.hpp"
#include "elliptic/thermomechanics/linear/Thermomechanics.hpp"

#include "elliptic/mechanical/nonlinear/Mechanics.hpp"
#include "elliptic/thermomechanics/nonlinear/ThermoMechanics.hpp"
#include "BaseExpInstMacros.hpp"

// linear applications
PLATO_ELEMENT_DEF(Plato::Elliptic::ProblemEvaluatorThermoMechanics, Plato::Elliptic::Linear::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::ProblemEvaluatorThermoMechanics, Plato::Elliptic::Linear::Electrical)
PLATO_ELEMENT_DEF(Plato::Elliptic::ProblemEvaluatorThermoMechanics, Plato::Elliptic::Linear::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::ProblemEvaluatorThermoMechanics, Plato::Elliptic::Linear::Electromechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::ProblemEvaluatorThermoMechanics, Plato::Elliptic::Linear::Mechanics)
// nonlinear applications
PLATO_ELEMENT_DEF(Plato::Elliptic::ProblemEvaluatorThermoMechanics, Plato::Elliptic::Nonlinear::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::ProblemEvaluatorThermoMechanics, Plato::Elliptic::Nonlinear::ThermoMechanics)

#endif