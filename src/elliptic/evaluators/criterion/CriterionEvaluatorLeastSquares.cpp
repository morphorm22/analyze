#include "elliptic/evaluators/criterion/CriterionEvaluatorLeastSquares_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/evaluators/criterion/CriterionEvaluatorLeastSquares_def.hpp"

#include "elliptic/thermal/Thermal.hpp"
#include "elliptic/electrical/Electrical.hpp"
#include "elliptic/mechanical/linear/Mechanics.hpp"
#include "elliptic/electromechanics/Electromechanics.hpp"
#include "elliptic/thermomechanics/linear/Thermomechanics.hpp"

#include "elliptic/mechanical/nonlinear/Mechanics.hpp"
#include "elliptic/thermomechanics/nonlinear/ThermoMechanics.hpp"
#include "BaseExpInstMacros.hpp"

// linear applications
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorLeastSquares, Plato::Elliptic::Linear::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorLeastSquares, Plato::Elliptic::Linear::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorLeastSquares, Plato::Elliptic::Linear::Electrical)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorLeastSquares, Plato::Elliptic::Linear::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorLeastSquares, Plato::Elliptic::Linear::Electromechanics)
// nonlinear applications
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorLeastSquares, Plato::Elliptic::Nonlinear::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::CriterionEvaluatorLeastSquares, Plato::Elliptic::Nonlinear::ThermoMechanics)

#endif
