#include "hyperbolic/HyperbolicScalarFunctionFactory.hpp"
#include "hyperbolic/HyperbolicScalarFunctionFactory_def.hpp"

PLATO_ELEMENT_DEF(Plato::Hyperbolic::ScalarFunctionFactory, Plato::Hyperbolic::Mechanics)
#ifdef PLATO_MICROMORPHIC
PLATO_ELEMENT_DEF(Plato::Hyperbolic::ScalarFunctionFactory, Plato::Hyperbolic::MicromorphicMechanics)
#endif
