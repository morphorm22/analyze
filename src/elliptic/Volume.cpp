#include "elliptic/Volume_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/Volume_def.hpp"

#include "ThermalElement.hpp"
#include "MechanicsElement.hpp"
#include "ThermomechanicsElement.hpp"
#include "ElectromechanicsElement.hpp"
#include "element/ThermoElasticElement.hpp"
#include "elliptic/electrical/ElectricalElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::Volume, Plato::ThermalElement)
PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::Volume, Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::Volume, Plato::ElectricalElement)
PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::Volume, Plato::ThermoElasticElement)
PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::Volume, Plato::ThermomechanicsElement)
PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::Volume, Plato::ElectromechanicsElement)

#endif

