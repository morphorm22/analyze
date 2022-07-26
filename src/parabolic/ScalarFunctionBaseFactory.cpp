#include "parabolic/ScalarFunctionBaseFactory_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "Thermal.hpp"
#include "Thermomechanics.hpp"
#include "BaseExpInstMacros.hpp"
#include "parabolic/ScalarFunctionBaseFactory_def.hpp"

PLATO_ELEMENT_DEF(Plato::Parabolic::ScalarFunctionBaseFactory, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Parabolic::ScalarFunctionBaseFactory, Plato::Thermomechanics)

#endif
