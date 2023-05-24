/*
 * DarkCurrentDensityTwoPhaseAlloy.cpp
 *
 *  Created on: May 24, 2023
 */

#include "elliptic/electrical/DarkCurrentDensityTwoPhaseAlloy_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/electrical/DarkCurrentDensityTwoPhaseAlloy_def.hpp"

#include "elliptic/electrical/ElectricalElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::DarkCurrentDensityTwoPhaseAlloy, Plato::ElectricalElement)

#endif