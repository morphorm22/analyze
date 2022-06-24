#include <elliptic/VolumeAverageCriterion.hpp>

PLATO_ELEMENT_DEF(Plato::Elliptic::VolumeAverageCriterion, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::VolumeAverageCriterion, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::VolumeAverageCriterion, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::VolumeAverageCriterion, Plato::Electromechanics)

#ifdef PLATO_STABILIZED
PLATO_ELEMENT_DEF(Plato::Elliptic::VolumeAverageCriterion, Plato::StabilizedMechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::VolumeAverageCriterion, Plato::StabilizedThermomechanics)
#endif
