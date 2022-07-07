#include "parabolic/PhysicsScalarFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "parabolic/PhysicsScalarFunction_def.hpp"

#include "Thermal.hpp"
#include "Thermomechanics.hpp"
#include "BaseExpInstMacros.hpp"

PLATO_ELEMENT_DEF(Plato::Parabolic::PhysicsScalarFunction, Plato::Thermal);
PLATO_ELEMENT_DEF(Plato::Parabolic::PhysicsScalarFunction, Plato::Thermomechanics);

#endif
