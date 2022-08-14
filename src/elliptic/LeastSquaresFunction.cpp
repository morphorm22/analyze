#include "elliptic/LeastSquaresFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/LeastSquaresFunction_def.hpp"

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Thermomechanics.hpp"
#include "Electromechanics.hpp"
#include "BaseExpInstMacros.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::LeastSquaresFunction, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::LeastSquaresFunction, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::LeastSquaresFunction, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::LeastSquaresFunction, Plato::Electromechanics)

#ifdef PLATO_STABILIZED
  #include "stabilized/Mechanics.hpp"
  #include "stabilized/Thermomechanics.hpp"
  PLATO_ELEMENT_DEF(Plato::Elliptic::LeastSquaresFunction, Plato::Stabilized::Mechanics)
  PLATO_ELEMENT_DEF(Plato::Elliptic::LeastSquaresFunction, Plato::Stabilized::Thermomechanics)
#endif

#endif
