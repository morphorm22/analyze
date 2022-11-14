#include "parabolic/Problem_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "Thermal.hpp"
#include "Thermomechanics.hpp"
#include "BaseExpInstMacros.hpp"
#include "parabolic/Problem_def.hpp"

PLATO_ELEMENT_DEF(Plato::Parabolic::Problem, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Parabolic::Problem, Plato::Thermomechanics)

#endif
