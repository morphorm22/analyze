#include "parabolic/TemperatureAverage_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "parabolic/TemperatureAverage_def.hpp"

#include "ThermalElement.hpp"
#include "parabolic/ExpInstMacros.hpp"

PLATO_PARABOLIC_EXP_INST(Plato::Parabolic::TemperatureAverage, Plato::ThermalElement)

#endif
