#include "elliptic/WeightedSumFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/WeightedSumFunction_def.hpp"

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Thermomechanics.hpp"
#include "Electromechanics.hpp"
#include "BaseExpInstMacros.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::WeightedSumFunction, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::WeightedSumFunction, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::WeightedSumFunction, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::WeightedSumFunction, Plato::Electromechanics)

#endif
