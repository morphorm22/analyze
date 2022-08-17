#include "ExpressionEvaluator_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "FadTypes.hpp"
#include "ExpressionEvaluator_def.hpp"

template class Plato::ExpressionEvaluator<
    Plato::ScalarMultiVectorT<Plato::Scalar>, 
    Plato::ScalarMultiVectorT<Plato::Scalar>, 
    Plato::ScalarVectorT<Plato::Scalar>, Plato::Scalar>;

template class Plato::ExpressionEvaluator<
    Plato::HostScalarMultiVectorT<Plato::Scalar>, 
    Plato::HostScalarMultiVectorT<Plato::Scalar>, 
    Plato::HostScalarVectorT<Plato::Scalar>, Plato::Scalar>;

#define PLATO_CONFIG_DEF_(C) Plato::Config<C>::FadType

#define PLATO_EVALUATOR_DEF(C) \
template class Plato::ExpressionEvaluator< \
    Plato::ScalarMultiVectorT<PLATO_CONFIG_DEF_(C)>, \
    Plato::ScalarMultiVectorT<PLATO_CONFIG_DEF_(C)>, \
    Plato::ScalarVectorT<PLATO_CONFIG_DEF_(C)>, Plato::Scalar>;

#include "Tri3.hpp"
#include "Tet10.hpp"
#include "Tet4.hpp"

PLATO_EVALUATOR_DEF(Plato::Tri3)
PLATO_EVALUATOR_DEF(Plato::Tet10)
PLATO_EVALUATOR_DEF(Plato::Tet4)

#ifdef PLATO_HEX_ELEMENTS

#include "Hex8.hpp"
#include "Quad4.hpp"
#include "Hex27.hpp"

PLATO_EVALUATOR_DEF(Plato::Hex8)
PLATO_EVALUATOR_DEF(Plato::Quad4)
PLATO_EVALUATOR_DEF(Plato::Hex27)

#endif

#endif
