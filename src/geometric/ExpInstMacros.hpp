#pragma once

#include "Ramp.hpp"
#include "Simp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

#include "Tet4.hpp"
#include "Tri3.hpp"
#include "Bar2.hpp"

#include "Tet10.hpp"
#include "Tri6.hpp"

#include "Hex8.hpp"
#include "Quad4.hpp"

#include "Hex27.hpp"
#include "Quad9.hpp"

#include "geometric/EvaluationTypes.hpp"

#define SKIP_GEOMETRIC_EXP_INST

#ifdef SKIP_GEOMETRIC_EXP_INST
#define PLATO_GEOMETRIC_DEF_(C, T)
#define PLATO_GEOMETRIC_DEC_(C, T)
#define PLATO_GEOMETRIC_DEF(C, T)
#define PLATO_GEOMETRIC_DEC(C, T)
#define PLATO_GEOMETRIC_DEF_2(C, T)
#define PLATO_GEOMETRIC_DEC_2(C, T)
#define PLATO_GEOMETRIC_DEF_3_(C, T)
#define PLATO_GEOMETRIC_DEC_3_(C, T)
#define PLATO_GEOMETRIC_DEF_3(C, T)
#define PLATO_GEOMETRIC_DEC_3(C, T)
#else


#define PLATO_GEOMETRIC_DEF_(C, T) \
template class C<Plato::Geometric::ResidualTypes<T>, Plato::MSIMP >; \
template class C<Plato::Geometric::ResidualTypes<T>, Plato::RAMP >; \
template class C<Plato::Geometric::ResidualTypes<T>, Plato::Heaviside >; \
template class C<Plato::Geometric::ResidualTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Geometric::GradientXTypes<T>, Plato::MSIMP >; \
template class C<Plato::Geometric::GradientXTypes<T>, Plato::RAMP >; \
template class C<Plato::Geometric::GradientXTypes<T>, Plato::Heaviside >; \
template class C<Plato::Geometric::GradientXTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Geometric::GradientZTypes<T>, Plato::MSIMP >; \
template class C<Plato::Geometric::GradientZTypes<T>, Plato::RAMP >; \
template class C<Plato::Geometric::GradientZTypes<T>, Plato::Heaviside >; \
template class C<Plato::Geometric::GradientZTypes<T>, Plato::NoPenalty >;

#define PLATO_GEOMETRIC_DEC_(C, T) \
extern template class C<Plato::Geometric::ResidualTypes<T>, Plato::MSIMP >; \
extern template class C<Plato::Geometric::ResidualTypes<T>, Plato::RAMP >; \
extern template class C<Plato::Geometric::ResidualTypes<T>, Plato::Heaviside >; \
extern template class C<Plato::Geometric::ResidualTypes<T>, Plato::NoPenalty >; \
extern template class C<Plato::Geometric::GradientXTypes<T>, Plato::MSIMP >; \
extern template class C<Plato::Geometric::GradientXTypes<T>, Plato::RAMP >; \
extern template class C<Plato::Geometric::GradientXTypes<T>, Plato::Heaviside >; \
extern template class C<Plato::Geometric::GradientXTypes<T>, Plato::NoPenalty >; \
extern template class C<Plato::Geometric::GradientZTypes<T>, Plato::MSIMP >; \
extern template class C<Plato::Geometric::GradientZTypes<T>, Plato::RAMP >; \
extern template class C<Plato::Geometric::GradientZTypes<T>, Plato::Heaviside >; \
extern template class C<Plato::Geometric::GradientZTypes<T>, Plato::NoPenalty >;

#define PLATO_GEOMETRIC_DEF(C, T) \
PLATO_GEOMETRIC_DEF_(C, T<Plato::Tet4>); \
PLATO_GEOMETRIC_DEF_(C, T<Plato::Tri3>); \
PLATO_GEOMETRIC_DEF_(C, T<Plato::Tet10>); \
PLATO_GEOMETRIC_DEF_(C, T<Plato::Hex8>); \
PLATO_GEOMETRIC_DEF_(C, T<Plato::Quad4>); \
PLATO_GEOMETRIC_DEF_(C, T<Plato::Hex27>);

#define PLATO_GEOMETRIC_DEC(C, T) \
PLATO_GEOMETRIC_DEC_(C, T<Plato::Tet4>); \
PLATO_GEOMETRIC_DEC_(C, T<Plato::Tri3>); \
PLATO_GEOMETRIC_DEC_(C, T<Plato::Tet10>); \
PLATO_GEOMETRIC_DEC_(C, T<Plato::Hex8>); \
PLATO_GEOMETRIC_DEC_(C, T<Plato::Quad4>); \
PLATO_GEOMETRIC_DEC_(C, T<Plato::Hex27>);

#define PLATO_GEOMETRIC_DEF_2(C, T) \
template class C<T<Plato::Tet4>>; \
template class C<T<Plato::Tri3>>; \
template class C<T<Plato::Tet10>>; \
template class C<T<Plato::Hex8>>; \
template class C<T<Plato::Quad4>>; \
template class C<T<Plato::Hex27>>;

#define PLATO_GEOMETRIC_DEC_2(C, T) \
extern template class C<T<Plato::Tet4>>; \
extern template class C<T<Plato::Tri3>>; \
extern template class C<T<Plato::Tet10>>; \
extern template class C<T<Plato::Hex8>>; \
extern template class C<T<Plato::Quad4>>; \
extern template class C<T<Plato::Hex27>>;

#define PLATO_GEOMETRIC_DEF_3_(C, T) \
extern template class C<Plato::Geometric::ResidualTypes<T>>; \
extern template class C<Plato::Geometric::GradientXTypes<T>>; \
extern template class C<Plato::Geometric::GradientZTypes<T>>;

#define PLATO_GEOMETRIC_DEC_3_(C, T) \
template class C<Plato::Geometric::ResidualTypes<T>>; \
template class C<Plato::Geometric::GradientXTypes<T>>; \
template class C<Plato::Geometric::GradientZTypes<T>>;

#define PLATO_GEOMETRIC_DEF_3(C, T) \
PLATO_GEOMETRIC_DEF_3_(C, T<Plato::Tet4>); \
PLATO_GEOMETRIC_DEF_3_(C, T<Plato::Tri3>); \
PLATO_GEOMETRIC_DEF_3_(C, T<Plato::Tet10>); \
PLATO_GEOMETRIC_DEF_3_(C, T<Plato::Hex8>); \
PLATO_GEOMETRIC_DEF_3_(C, T<Plato::Quad4>); \
PLATO_GEOMETRIC_DEF_3_(C, T<Plato::Hex27>);

#define PLATO_GEOMETRIC_DEC_3(C, T) \
PLATO_GEOMETRIC_DEC_3_(C, T<Plato::Tet4>); \
PLATO_GEOMETRIC_DEC_3_(C, T<Plato::Tri3>); \
PLATO_GEOMETRIC_DEC_3_(C, T<Plato::Tet10>); \
PLATO_GEOMETRIC_DEC_3_(C, T<Plato::Hex8>); \
PLATO_GEOMETRIC_DEC_3_(C, T<Plato::Quad4>); \
PLATO_GEOMETRIC_DEC_3_(C, T<Plato::Hex27>);

#endif
