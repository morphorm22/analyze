#include "stabilized/Problem_decl.hpp"
  
#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "stabilized/Mechanics.hpp"
#include "stabilized/Thermomechanics.hpp"
#include "stabilized/Problem_def.hpp"
#include "BaseExpInstMacros.hpp"

PLATO_ELEMENT_DEF(Plato::Stabilized::Problem, Plato::Stabilized::Mechanics)
PLATO_ELEMENT_DEF(Plato::Stabilized::Problem, Plato::Stabilized::Thermomechanics)

#endif
