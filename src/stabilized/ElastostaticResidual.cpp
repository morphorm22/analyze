#include "stabilized/ElastostaticResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "stabilized/ElastostaticResidual_def.hpp"

#include "stabilized/MechanicsElement.hpp"
#include "stabilized/ExpInstMacros.hpp"

PLATO_STABILIZED_EXP_INST(Plato::Stabilized::ElastostaticResidual, Plato::Stabilized::MechanicsElement)

#endif
