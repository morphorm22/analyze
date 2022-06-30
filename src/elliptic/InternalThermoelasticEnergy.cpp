#include "elliptic/InternalThermoelasticEnergy_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/InternalThermoelasticEnergy_def.hpp"

#include "ThermomechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::InternalThermoelasticEnergy, Plato::ThermomechanicsElement)

#endif
