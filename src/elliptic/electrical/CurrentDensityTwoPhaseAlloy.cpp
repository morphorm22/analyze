/*
 *  CurrentDensityTwoPhaseAlloy.cpp
 *
 *  Created on: June 2, 2023
 */

#include "elliptic/electrical/CurrentDensityTwoPhaseAlloy_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/electrical/CurrentDensityTwoPhaseAlloy_def.hpp"

#include "elliptic/electrical/ElectricalElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::CurrentDensityTwoPhaseAlloy, Plato::ElectricalElement)

#endif