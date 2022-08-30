#include "elliptic/EMStressPNorm_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/EMStressPNorm_def.hpp"

#include "ElectromechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::EMStressPNorm, Plato::ElectromechanicsElement)

#endif
