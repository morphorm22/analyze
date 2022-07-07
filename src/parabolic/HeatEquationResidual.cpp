#include "parabolic/HeatEquationResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "parabolic/HeatEquationResidual_def.hpp"

#include "ThermalElement.hpp"
#include "parabolic/ExpInstMacros.hpp"

PLATO_PARABOLIC_EXP_INST(Plato::Parabolic::HeatEquationResidual, Plato::ThermalElement)

#endif
