#include "elliptic/Problem_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/Problem_def.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::Problem, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::Problem, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::Problem, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::Problem, Plato::Electromechanics)

#endif
