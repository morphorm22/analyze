#include "hyperbolic/PhysicsScalarFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "hyperbolic/PhysicsScalarFunction_def.hpp"

#include "BaseExpInstMacros.hpp"
#include "hyperbolic/Mechanics.hpp"
PLATO_ELEMENT_DEF(Plato::Hyperbolic::PhysicsScalarFunction, Plato::Hyperbolic::Mechanics)

#ifdef PLATO_MICROMORPHIC
#include "hyperbolic/micromorphic/MicromorphicMechanics.hpp"
PLATO_ELEMENT_DEF(Plato::Hyperbolic::PhysicsScalarFunction, Plato::Hyperbolic::MicromorphicMechanics)
#endif

#endif
