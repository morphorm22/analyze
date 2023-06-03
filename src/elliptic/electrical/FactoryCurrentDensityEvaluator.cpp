/*
 *  FactoryCurrentDensityEvaluator.cpp
 *
 *  Created on: June 2, 2023
 */

#include "elliptic/electrical/FactoryCurrentDensityEvaluator_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/electrical/FactoryCurrentDensityEvaluator_def.hpp"

#include "elliptic/electrical/ElectricalElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::FactoryCurrentDensityEvaluator, Plato::ElectricalElement)

#endif