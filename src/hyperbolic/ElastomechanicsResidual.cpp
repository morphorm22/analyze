#include "hyperbolic/ElastomechanicsResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "hyperbolic/ElastomechanicsResidual_def.hpp"

#include "MechanicsElement.hpp"
#include "hyperbolic/ExpInstMacros.hpp"

PLATO_HYPERBOLIC_EXP_INST(Plato::Hyperbolic::TransientMechanicsResidual, Plato::MechanicsElement)

#endif