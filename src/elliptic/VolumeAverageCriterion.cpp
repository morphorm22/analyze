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

#endif
