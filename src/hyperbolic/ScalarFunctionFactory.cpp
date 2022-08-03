#include "hyperbolic/ScalarFunctionFactory.hpp"
#include "hyperbolic/ScalarFunctionFactory_def.hpp"

PLATO_ELEMENT_DEF(Plato::Hyperbolic::ScalarFunctionFactory, Plato::Hyperbolic::Mechanics)
#ifdef PLATO_MICROMORPHIC
PLATO_ELEMENT_DEF(Plato::Hyperbolic::ScalarFunctionFactory, Plato::Hyperbolic::MicromorphicMechanics)
#endif
