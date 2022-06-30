#include "elliptic/VolumeAverageCriterion_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/VolumeAverageCriterion_def.hpp"

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Thermomechanics.hpp"
#include "Electromechanics.hpp"
#include "BaseExpInstMacros.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::VolumeAverageCriterion, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::VolumeAverageCriterion, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::VolumeAverageCriterion, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::VolumeAverageCriterion, Plato::Electromechanics)

#ifdef PLATO_STABILIZED
#include "StabilizedMechanics.hpp"
#include "StabilizedThermomechanics.hpp"
PLATO_ELEMENT_DEF(Plato::Elliptic::VolumeAverageCriterion, Plato::StabilizedMechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::VolumeAverageCriterion, Plato::StabilizedThermomechanics)
#endif

#endif
