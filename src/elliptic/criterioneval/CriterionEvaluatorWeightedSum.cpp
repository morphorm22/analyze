#include "elliptic/criterioneval/CriterionEvaluatorWeightedSum_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/criterioneval/CriterionEvaluatorWeightedSum_def.hpp"

#include "elliptic/mechanical/linear/Mechanics.hpp"
#include "elliptic/thermal/Thermal.hpp"
#include "elliptic/thermomechanics/Thermomechanics.hpp"
#include "elliptic/electrical/Electrical.hpp"
#include "elliptic/electromechanics/Electromechanics.hpp"
#include "BaseExpInstMacros.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorWeightedSum, Plato::Elliptic::Linear::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorWeightedSum, Plato::Elliptic::Linear::Electrical)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorWeightedSum, Plato::Elliptic::Linear::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorWeightedSum, Plato::Elliptic::Linear::Electromechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorWeightedSum, Plato::Elliptic::Linear::Mechanics)

#endif