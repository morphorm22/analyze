/*
 * LocalMeasureTensileEnergyDensity.cpp
 *
 */
#include "LocalMeasureTensileEnergyDensity_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "LocalMeasureTensileEnergyDensity_def.hpp"

#include "MechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::LocalMeasureTensileEnergyDensity, Plato::MechanicsElement)

#endif
