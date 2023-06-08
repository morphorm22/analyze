/*
 * WorksetBuilder.cpp
 *
 *  Created on: June 7, 2023
 */

#include "elliptic/base/WorksetBuilder_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/base/WorksetBuilder_def.hpp"

#include "ThermalElement.hpp"
#include "MechanicsElement.hpp"
#include "ThermomechanicsElement.hpp"
#include "ElectromechanicsElement.hpp"
#include "elliptic/electrical/ElectricalElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::WorksetBuilder, Plato::ThermalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::WorksetBuilder, Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::WorksetBuilder, Plato::ElectricalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::WorksetBuilder, Plato::ThermomechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::WorksetBuilder, Plato::ElectromechanicsElement)

#endif