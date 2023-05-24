/*
 * MaterialElectricalConductivityTwoPhaseAlloy.cpp
 *
 *  Created on: May 23, 2023
 */

#include "elliptic/electrical/MaterialElectricalConductivityTwoPhaseAlloy_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/electrical/MaterialElectricalConductivityTwoPhaseAlloy_def.hpp"

#include "elliptic/electrical/ElectricalElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::MaterialElectricalConductivityTwoPhaseAlloy, Plato::ElectricalElement)

#endif