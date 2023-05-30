#include "elliptic/mechanical/linear/CriterionMassMoment_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/mechanical/linear/CriterionMassMoment_def.hpp"

#include "ThermalElement.hpp"
#include "MechanicsElement.hpp"
#include "ThermomechanicsElement.hpp"
#include "ElectromechanicsElement.hpp"
#include "elliptic/electrical/ElectricalElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::CriterionMassMoment, Plato::ThermalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::CriterionMassMoment, Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::CriterionMassMoment, Plato::ElectricalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::CriterionMassMoment, Plato::ThermomechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::CriterionMassMoment, Plato::ElectromechanicsElement)

#endif
