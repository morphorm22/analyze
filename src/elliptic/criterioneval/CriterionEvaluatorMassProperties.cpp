#include "elliptic/criterioneval/CriterionEvaluatorMassProperties_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/criterioneval/CriterionEvaluatorMassProperties_def.hpp"

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Thermomechanics.hpp"
#include "Electromechanics.hpp"
#include "BaseExpInstMacros.hpp"
#include "elliptic/electrical/Electrical.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorMassProperties, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorMassProperties, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorMassProperties, Plato::Electrical)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorMassProperties, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorMassProperties, Plato::Electromechanics)

#endif
