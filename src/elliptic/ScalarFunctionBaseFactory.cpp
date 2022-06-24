#include "elliptic/ScalarFunctionBaseFactory.hpp"
#include "elliptic/ScalarFunctionBaseFactory_def.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::ScalarFunctionBaseFactory, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::ScalarFunctionBaseFactory, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::ScalarFunctionBaseFactory, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::ScalarFunctionBaseFactory, Plato::Electromechanics)


#ifdef PLATO_STABILIZED
PLATO_ELEMENT_DEF(Plato::Elliptic::ScalarFunctionBaseFactory, Plato::StabilizedMechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::ScalarFunctionBaseFactory, Plato::StabilizedThermomechanics)
#endif
