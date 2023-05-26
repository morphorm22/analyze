#include "elliptic/criterioneval/CriterionEvaluatorVolumeAverage_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/criterioneval/CriterionEvaluatorVolumeAverage_def.hpp"

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Thermomechanics.hpp"
#include "Electromechanics.hpp"
#include "BaseExpInstMacros.hpp"
#include "elliptic/electrical/Electrical.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorVolumeAverage, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorVolumeAverage, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorVolumeAverage, Plato::Electrical)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorVolumeAverage, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorVolumeAverage, Plato::Electromechanics)

#endif
