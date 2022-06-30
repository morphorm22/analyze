#include "elliptic/FluxPNorm_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/FluxPNorm_def.hpp"

#include "ThermalElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::FluxPNorm, Plato::ThermalElement)

#endif
