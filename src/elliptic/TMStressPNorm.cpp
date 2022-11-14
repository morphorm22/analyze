#include "elliptic/TMStressPNorm_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/TMStressPNorm_def.hpp"

#include "ThermomechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::TMStressPNorm, Plato::ThermomechanicsElement)

#endif
