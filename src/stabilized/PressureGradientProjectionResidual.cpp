#include "stabilized/PressureGradientProjectionResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "stabilized/PressureGradientProjectionResidual_def.hpp"

#include "stabilized/MechanicsElement.hpp"
#include "stabilized/ProjectionElement.hpp"
#include "stabilized/ExpInstMacros.hpp"

//PLATO_STABILIZED_EXP_INST(Plato::Stabilized::PressureGradientProjectionResidual, Plato::Stabilized::Mechanics::ProjectorType::ElementType)
//template class Plato::Stabilized::PressureGradientProjectionResidual<Plato::Stabilized::JacobianNTypes<Plato::Stabilized::ProjectionElement<Plato::Tet10, 4, 3, 1, 1> >, Plato::RAMP>;

PLATO_STABILIZED_EXP_INST_2(Plato::Stabilized::PressureGradientProjectionResidual, Plato::Stabilized::ProjectionElement, Plato::Stabilized::MechanicsElement)

//namespace Plato {
//namespace Stabilized {
//using ElementType = MechanicsElement<Plato::Tet10>;
//template class PressureGradientProjectionResidual<JacobianNTypes<ProjectionElement<Plato::Tet10, ElementType::mNumDofsPerNode, ElementType::mPressureDofOffset> >, Plato::RAMP>;
//}
//}

#endif
