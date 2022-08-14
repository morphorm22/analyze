#include "elliptic/MassMoment_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/MassMoment_def.hpp"

#include "ThermalElement.hpp"
#include "MechanicsElement.hpp"
#include "ThermomechanicsElement.hpp"
#include "ElectromechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::MassMoment, Plato::ThermalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::MassMoment, Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::MassMoment, Plato::ThermomechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::MassMoment, Plato::ElectromechanicsElement)

#ifdef PLATO_STABILIZED
  #include "stabilized/MechanicsElement.hpp"
  #include "stabilized/ThermomechanicsElement.hpp"
  PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::MassMoment, Plato::Stabilized::MechanicsElement)
  PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::MassMoment, Plato::Stabilized::ThermomechanicsElement)
#endif

#endif
