#include "stabilized/ElastostaticEnergy_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "stabilized/ElastostaticEnergy_def.hpp"
#include "stabilized/MechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Stabilized::ElastostaticEnergy, Plato::Stabilized::MechanicsElement)

#endif
