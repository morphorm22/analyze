#include "hyperbolic/HyperbolicProblem_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "hyperbolic/HyperbolicProblem_def.hpp"
#include "BaseExpInstMacros.hpp"

#include "HyperbolicMechanics.hpp"
PLATO_ELEMENT_DEF(Plato::Hyperbolic::HyperbolicProblem, Plato::Hyperbolic::Mechanics)

#ifdef PLATO_MICROMORPHIC
#include "MicromorphicMechanics.hpp"
PLATO_ELEMENT_DEF(Plato::Hyperbolic::HyperbolicProblem, Plato::Hyperbolic::MicromorphicMechanics)
#endif

#endif