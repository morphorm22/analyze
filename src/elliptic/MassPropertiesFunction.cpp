#include "elliptic/MassPropertiesFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/MassPropertiesFunction_def.hpp"

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Thermomechanics.hpp"
#include "Electromechanics.hpp"
#include "BaseExpInstMacros.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::MassPropertiesFunction, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::MassPropertiesFunction, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::MassPropertiesFunction, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::MassPropertiesFunction, Plato::Electromechanics)

#ifdef PLATO_STABILIZED
  #include "stabilized/Mechanics.hpp"
  #include "stabilized/Thermomechanics.hpp"
  PLATO_ELEMENT_DEF(Plato::Elliptic::MassPropertiesFunction, Plato::Stabilized::Mechanics)
  PLATO_ELEMENT_DEF(Plato::Elliptic::MassPropertiesFunction, Plato::Stabilized::Thermomechanics)
#endif

#endif
