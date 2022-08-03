#include "hyperbolic/StressPNorm_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "hyperbolic/StressPNorm_def.hpp"

#include "MechanicsElement.hpp"
#include "hyperbolic/ExpInstMacros.hpp"

PLATO_HYPERBOLIC_EXP_INST(Plato::Hyperbolic::StressPNorm, Plato::MechanicsElement)

#endif