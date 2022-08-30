#include "elliptic/DivisionFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/DivisionFunction_def.hpp"

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Thermomechanics.hpp"
#include "Electromechanics.hpp"
#include "BaseExpInstMacros.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::DivisionFunction, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::DivisionFunction, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::DivisionFunction, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::DivisionFunction, Plato::Electromechanics)

#ifdef PLATO_STABILIZED
  #include "stabilized/Mechanics.hpp"
  #include "stabilized/Thermomechanics.hpp"
  PLATO_ELEMENT_DEF(Plato::Elliptic::DivisionFunction, Plato::Stabilized::Mechanics)
  PLATO_ELEMENT_DEF(Plato::Elliptic::DivisionFunction, Plato::Stabilized::Thermomechanics)
#endif

#endif
