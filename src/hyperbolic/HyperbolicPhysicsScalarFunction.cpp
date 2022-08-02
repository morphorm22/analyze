#include "hyperbolic/HyperbolicPhysicsScalarFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "hyperbolic/HyperbolicPhysicsScalarFunction_def.hpp"

#include "BaseExpInstMacros.hpp"
#include "HyperbolicMechanics.hpp"
PLATO_ELEMENT_DEF(Plato::Hyperbolic::PhysicsScalarFunction, Plato::Hyperbolic::Mechanics)

#ifdef PLATO_MICROMORPHIC
#include "MicromorphicMechanics.hpp"
PLATO_ELEMENT_DEF(Plato::Hyperbolic::PhysicsScalarFunction, Plato::Hyperbolic::MicromorphicMechanics)
#endif

#endif
