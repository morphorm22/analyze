#pragma once

#include "Tet4.hpp"
#include "Tri3.hpp"
#include "Bar2.hpp"

#include "Tet10.hpp"
#include "Tri6.hpp"

#include "Hex8.hpp"
#include "Quad4.hpp"

#include "Hex27.hpp"
#include "Quad9.hpp"

#include "helmholtz/EvaluationTypes.hpp"

#define SKIP_HELMHOLTZ_EXP_INST

#ifdef SKIP_HELMHOLTZ_EXP_INST
#define PLATO_HELMHOLTZ_DEF_3_(C, T)
#define PLATO_HELMHOLTZ_DEC_3_(C, T)
#define PLATO_HELMHOLTZ_DEF_3(C, T)
#define PLATO_HELMHOLTZ_DEC_3(C, T)
#else

#define PLATO_HELMHOLTZ_DEF_3_(C, T) \
extern template class C<Plato::Helmholtz::ResidualTypes<T>>; \
extern template class C<Plato::Helmholtz::JacobianTypes<T>>; \

#define PLATO_HELMHOLTZ_DEC_3_(C, T) \
template class C<Plato::Helmholtz::ResidualTypes<T>>; \
template class C<Plato::Helmholtz::JacobianTypes<T>>; \

#define PLATO_HELMHOLTZ_DEF_3(C, T) \
PLATO_HELMHOLTZ_DEF_3_(C, T<Plato::Tet4>); \
PLATO_HELMHOLTZ_DEF_3_(C, T<Plato::Tri3>); \
PLATO_HELMHOLTZ_DEF_3_(C, T<Plato::Tet10>); \
PLATO_HELMHOLTZ_DEF_3_(C, T<Plato::Hex8>); \
PLATO_HELMHOLTZ_DEF_3_(C, T<Plato::Quad4>); \
PLATO_HELMHOLTZ_DEF_3_(C, T<Plato::Hex27>);

#define PLATO_HELMHOLTZ_DEC_3(C, T) \
PLATO_HELMHOLTZ_DEC_3_(C, T<Plato::Tet4>); \
PLATO_HELMHOLTZ_DEC_3_(C, T<Plato::Tri3>); \
PLATO_HELMHOLTZ_DEC_3_(C, T<Plato::Tet10>); \
PLATO_HELMHOLTZ_DEC_3_(C, T<Plato::Hex8>); \
PLATO_HELMHOLTZ_DEC_3_(C, T<Plato::Quad4>); \
PLATO_HELMHOLTZ_DEC_3_(C, T<Plato::Hex27>);

#endif
