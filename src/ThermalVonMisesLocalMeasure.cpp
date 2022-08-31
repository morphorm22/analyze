/*
 * ThermalVonMisesLocalMeasure.cpp
 *
 */

#include "ThermalVonMisesLocalMeasure_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "ThermalVonMisesLocalMeasure_def.hpp"

#include "ThermomechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::ThermalVonMisesLocalMeasure, Plato::ThermomechanicsElement)

#endif
