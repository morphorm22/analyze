#include "elliptic/StressPNorm_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/StressPNorm_def.hpp"

#include "MechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::StressPNorm, Plato::MechanicsElement)

#endif
