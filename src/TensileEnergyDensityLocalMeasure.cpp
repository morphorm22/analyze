/*
 * TensileEnergyDensityLocalMeasure.cpp
 *
 */
#include "TensileEnergyDensityLocalMeasure_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "TensileEnergyDensityLocalMeasure_def.hpp"

#include "MechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::TensileEnergyDensityLocalMeasure, Plato::MechanicsElement)

#endif
