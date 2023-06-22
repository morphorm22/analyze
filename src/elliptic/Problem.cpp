/*
 * Problem.cpp
 *
 *  Created on: June 21, 2023
 */

#include "elliptic/Problem_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/Problem_def.hpp"

#include "elliptic/mechanical/linear/Mechanics.hpp"
#include "elliptic/thermal/Thermal.hpp"
#include "elliptic/thermomechanics/linear/Thermomechanics.hpp"
#include "elliptic/electrical/Electrical.hpp"
#include "elliptic/electromechanics/Electromechanics.hpp"
#include "BaseExpInstMacros.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::Problem, Plato::Elliptic::Linear::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::Problem, Plato::Elliptic::Linear::Electrical)
PLATO_ELEMENT_DEF(Plato::Elliptic::Problem, Plato::Elliptic::Linear::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::Problem, Plato::Elliptic::Linear::Electromechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::Problem, Plato::Elliptic::Linear::Mechanics)

#endif