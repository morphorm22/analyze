#include "elliptic/criterioneval/CriterionEvaluatorSolutionFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/criterioneval/CriterionEvaluatorSolutionFunction_def.hpp"

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Thermomechanics.hpp"
#include "Electromechanics.hpp"
#include "BaseExpInstMacros.hpp"
#include "elliptic/electrical/Electrical.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorSolutionFunction, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorSolutionFunction, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorSolutionFunction, Plato::Electrical)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorSolutionFunction, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorSolutionFunction, Plato::Electromechanics)

#endif
