/*
 * LocalMeasureThermalVonMises.cpp
 *
 */

#include "elliptic/thermomechanics/linear/LocalMeasureThermalVonMises_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/thermomechanics/linear/LocalMeasureThermalVonMises_def.hpp"

#include "ThermomechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::LocalMeasureThermalVonMises, Plato::ThermomechanicsElement)

#endif
