#include "elliptic/criterioneval/CriterionEvaluatorLeastSquares_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/criterioneval/CriterionEvaluatorLeastSquares_def.hpp"

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Thermomechanics.hpp"
#include "Electromechanics.hpp"
#include "BaseExpInstMacros.hpp"
#include "elliptic/electrical/Electrical.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorLeastSquares, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorLeastSquares, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorLeastSquares, Plato::Electrical)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorLeastSquares, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorLeastSquares, Plato::Electromechanics)

#endif
