#include "stabilized/PressureGradientProjectionResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "stabilized/PressureGradientProjectionResidual_def.hpp"

#include "stabilized/ExpInstMacros.hpp"
#include "stabilized/MechanicsElement.hpp"
#include "stabilized/ProjectionElement.hpp"
#include "stabilized/ThermomechanicsElement.hpp"

PLATO_STABILIZED_EXP_INST_2(Plato::Stabilized::PressureGradientProjectionResidual, Plato::Stabilized::ProjectionElement, Plato::Stabilized::MechanicsElement)
PLATO_STABILIZED_EXP_INST_2(Plato::Stabilized::PressureGradientProjectionResidual, Plato::Stabilized::ProjectionElement, Plato::Stabilized::ThermomechanicsElement)
#endif
