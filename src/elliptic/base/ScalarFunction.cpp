/*
 * ScalarFunction.cpp
 *
 *  Created on: June 7, 2023
 */

#include "elliptic/base/ScalarFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/base/ScalarFunction_def.hpp"

#include "elliptic/mechanical/linear/Mechanics.hpp"
#include "elliptic/thermal/Thermal.hpp"
#include "elliptic/thermomechanics/Thermomechanics.hpp"
#include "elliptic/electrical/Electrical.hpp"
#include "elliptic/electromechanics/Electromechanics.hpp"
#include "BaseExpInstMacros.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::ScalarFunction, Plato::Elliptic::Linear::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::ScalarFunction, Plato::Elliptic::Linear::Electrical)
PLATO_ELEMENT_DEF(Plato::Elliptic::ScalarFunction, Plato::Elliptic::Linear::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::ScalarFunction, Plato::Elliptic::Linear::Electromechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::ScalarFunction, Plato::Elliptic::Linear::Mechanics)

#endif
