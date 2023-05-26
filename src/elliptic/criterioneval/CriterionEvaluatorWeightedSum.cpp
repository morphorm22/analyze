#include "elliptic/criterioneval/CriterionEvaluatorWeightedSum_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/criterioneval/CriterionEvaluatorWeightedSum_def.hpp"

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Thermomechanics.hpp"
#include "Electromechanics.hpp"
#include "BaseExpInstMacros.hpp"
#include "elliptic/electrical/Electrical.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorWeightedSum, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorWeightedSum, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorWeightedSum, Plato::Electrical)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorWeightedSum, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorWeightedSum, Plato::Electromechanics)

#endif
