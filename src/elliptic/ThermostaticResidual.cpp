#include "elliptic/ThermostaticResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/ThermostaticResidual_def.hpp"

#include "ThermalElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::ThermostaticResidual, Plato::ThermalElement)

#endif
