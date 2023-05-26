#include "elliptic/criterioneval/FactoryCriterionEvaluator_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/criterioneval/FactoryCriterionEvaluator_def.hpp"

#include "BaseExpInstMacros.hpp"
#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Thermomechanics.hpp"
#include "Electromechanics.hpp"
#include "elliptic/electrical/Electrical.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::FactoryCriterionEvaluator, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::FactoryCriterionEvaluator, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::FactoryCriterionEvaluator, Plato::Electrical)
PLATO_ELEMENT_DEF(Plato::Elliptic::FactoryCriterionEvaluator, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::FactoryCriterionEvaluator, Plato::Electromechanics)


#endif
