#include "elliptic/ScalarFunctionBaseFactory_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "BaseExpInstMacros.hpp"
#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Thermomechanics.hpp"
#include "Electromechanics.hpp"
#include "stabilized/Mechanics.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::ScalarFunctionBaseFactory, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::ScalarFunctionBaseFactory, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::ScalarFunctionBaseFactory, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::ScalarFunctionBaseFactory, Plato::Electromechanics)

#ifdef PLATO_STABILIZED
  #include "elliptic/ScalarFunctionBaseFactory_def.hpp"
  PLATO_ELEMENT_DEF(Plato::Elliptic::ScalarFunctionBaseFactory, Plato::Stabilized::Mechanics)
#endif


#endif
