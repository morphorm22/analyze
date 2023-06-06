/*
 * ThermalVonMisesLocalMeasure.cpp
 *
 */

#include "elliptic/thermomechanics/ThermalVonMisesLocalMeasure_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/thermomechanics/ThermalVonMisesLocalMeasure_def.hpp"

#include "ThermomechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::ThermalVonMisesLocalMeasure, Plato::ThermomechanicsElement)

#endif
