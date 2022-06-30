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

#include "elliptic/EvaluationTypes.hpp"

#define PLATO_ELLIPTIC_EXP_INST_(C, T) \
template class C<Plato::Elliptic::ResidualTypes<T>, Plato::MSIMP >; \
template class C<Plato::Elliptic::ResidualTypes<T>, Plato::RAMP >; \
template class C<Plato::Elliptic::ResidualTypes<T>, Plato::Heaviside >; \
template class C<Plato::Elliptic::ResidualTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Elliptic::JacobianTypes<T>, Plato::MSIMP >; \
template class C<Plato::Elliptic::JacobianTypes<T>, Plato::RAMP >; \
template class C<Plato::Elliptic::JacobianTypes<T>, Plato::Heaviside >; \
template class C<Plato::Elliptic::JacobianTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Elliptic::GradientXTypes<T>, Plato::MSIMP >; \
template class C<Plato::Elliptic::GradientXTypes<T>, Plato::RAMP >; \
template class C<Plato::Elliptic::GradientXTypes<T>, Plato::Heaviside >; \
template class C<Plato::Elliptic::GradientXTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Elliptic::GradientZTypes<T>, Plato::MSIMP >; \
template class C<Plato::Elliptic::GradientZTypes<T>, Plato::RAMP >; \
template class C<Plato::Elliptic::GradientZTypes<T>, Plato::Heaviside >; \
template class C<Plato::Elliptic::GradientZTypes<T>, Plato::NoPenalty >;

#define PLATO_ELLIPTIC_EXP_INST_2_(C, T) \
template class C<Plato::Elliptic::ResidualTypes<T>>; \
template class C<Plato::Elliptic::JacobianTypes<T>>; \
template class C<Plato::Elliptic::GradientXTypes<T>>; \
template class C<Plato::Elliptic::GradientZTypes<T>>;


#ifdef PLATO_HEX_ELEMENTS
  #define PLATO_ELLIPTIC_EXP_INST(C, T) \
  PLATO_ELLIPTIC_EXP_INST_(C, T<Plato::Tet4>); \
  PLATO_ELLIPTIC_EXP_INST_(C, T<Plato::Tri3>); \
  PLATO_ELLIPTIC_EXP_INST_(C, T<Plato::Tet10>); \
  PLATO_ELLIPTIC_EXP_INST_(C, T<Plato::Hex8>); \
  PLATO_ELLIPTIC_EXP_INST_(C, T<Plato::Quad4>); \
  PLATO_ELLIPTIC_EXP_INST_(C, T<Plato::Hex27>);

  #define PLATO_ELLIPTIC_EXP_INST_2(C, T) \
  PLATO_ELLIPTIC_EXP_INST_2_(C, T<Plato::Tet4>); \
  PLATO_ELLIPTIC_EXP_INST_2_(C, T<Plato::Tri3>); \
  PLATO_ELLIPTIC_EXP_INST_2_(C, T<Plato::Tet10>); \
  PLATO_ELLIPTIC_EXP_INST_2_(C, T<Plato::Hex8>); \
  PLATO_ELLIPTIC_EXP_INST_2_(C, T<Plato::Quad4>); \
  PLATO_ELLIPTIC_EXP_INST_2_(C, T<Plato::Hex27>);
#else
  #define PLATO_ELLIPTIC_EXP_INST(C, T) \
  PLATO_ELLIPTIC_EXP_INST_(C, T<Plato::Tet4>); \
  PLATO_ELLIPTIC_EXP_INST_(C, T<Plato::Tri3>); \
  PLATO_ELLIPTIC_EXP_INST_(C, T<Plato::Tet10>);

  #define PLATO_ELLIPTIC_EXP_INST_2(C, T) \
  PLATO_ELLIPTIC_EXP_INST_2_(C, T<Plato::Tet4>); \
  PLATO_ELLIPTIC_EXP_INST_2_(C, T<Plato::Tri3>); \
  PLATO_ELLIPTIC_EXP_INST_2_(C, T<Plato::Tet10>);
#endif

#define SKIP_ELLIPTIC_EXP_INST

#ifdef SKIP_ELLIPTIC_EXP_INST

#define PLATO_ELLIPTIC_DEF_(C, T)
#define PLATO_ELLIPTIC_DEC_(C, T)
#define PLATO_ELLIPTIC_DEF(C, T)
#define PLATO_ELLIPTIC_DEC(C, T)
#define PLATO_ELLIPTIC_DEF_3_(C, T)
#define PLATO_ELLIPTIC_DEC_3_(C, T)
#define PLATO_ELLIPTIC_DEF_3(C, T)
#define PLATO_ELLIPTIC_DEC_3(C, T)

#else

#define PLATO_ELLIPTIC_DEF_(C, T) \
template class C<Plato::Elliptic::ResidualTypes<T>, Plato::MSIMP >; \
template class C<Plato::Elliptic::ResidualTypes<T>, Plato::RAMP >; \
template class C<Plato::Elliptic::ResidualTypes<T>, Plato::Heaviside >; \
template class C<Plato::Elliptic::ResidualTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Elliptic::JacobianTypes<T>, Plato::MSIMP >; \
template class C<Plato::Elliptic::JacobianTypes<T>, Plato::RAMP >; \
template class C<Plato::Elliptic::JacobianTypes<T>, Plato::Heaviside >; \
template class C<Plato::Elliptic::JacobianTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Elliptic::GradientXTypes<T>, Plato::MSIMP >; \
template class C<Plato::Elliptic::GradientXTypes<T>, Plato::RAMP >; \
template class C<Plato::Elliptic::GradientXTypes<T>, Plato::Heaviside >; \
template class C<Plato::Elliptic::GradientXTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Elliptic::GradientZTypes<T>, Plato::MSIMP >; \
template class C<Plato::Elliptic::GradientZTypes<T>, Plato::RAMP >; \
template class C<Plato::Elliptic::GradientZTypes<T>, Plato::Heaviside >; \
template class C<Plato::Elliptic::GradientZTypes<T>, Plato::NoPenalty >;

#define PLATO_ELLIPTIC_DEC_(C, T) \
extern template class C<Plato::Elliptic::ResidualTypes<T>, Plato::MSIMP >; \
extern template class C<Plato::Elliptic::ResidualTypes<T>, Plato::RAMP >; \
extern template class C<Plato::Elliptic::ResidualTypes<T>, Plato::Heaviside >; \
extern template class C<Plato::Elliptic::ResidualTypes<T>, Plato::NoPenalty >; \
extern template class C<Plato::Elliptic::JacobianTypes<T>, Plato::MSIMP >; \
extern template class C<Plato::Elliptic::JacobianTypes<T>, Plato::RAMP >; \
extern template class C<Plato::Elliptic::JacobianTypes<T>, Plato::Heaviside >; \
extern template class C<Plato::Elliptic::JacobianTypes<T>, Plato::NoPenalty >; \
extern template class C<Plato::Elliptic::GradientXTypes<T>, Plato::MSIMP >; \
extern template class C<Plato::Elliptic::GradientXTypes<T>, Plato::RAMP >; \
extern template class C<Plato::Elliptic::GradientXTypes<T>, Plato::Heaviside >; \
extern template class C<Plato::Elliptic::GradientXTypes<T>, Plato::NoPenalty >; \
extern template class C<Plato::Elliptic::GradientZTypes<T>, Plato::MSIMP >; \
extern template class C<Plato::Elliptic::GradientZTypes<T>, Plato::RAMP >; \
extern template class C<Plato::Elliptic::GradientZTypes<T>, Plato::Heaviside >; \
extern template class C<Plato::Elliptic::GradientZTypes<T>, Plato::NoPenalty >;

#define PLATO_ELLIPTIC_DEF(C, T) \
PLATO_ELLIPTIC_DEF_(C, T<Plato::Tet4>); \
PLATO_ELLIPTIC_DEF_(C, T<Plato::Tri3>); \
PLATO_ELLIPTIC_DEF_(C, T<Plato::Tet10>); \
PLATO_ELLIPTIC_DEF_(C, T<Plato::Hex8>); \
PLATO_ELLIPTIC_DEF_(C, T<Plato::Quad4>); \
PLATO_ELLIPTIC_DEF_(C, T<Plato::Hex27>);

#define PLATO_ELLIPTIC_DEC(C, T) \
PLATO_ELLIPTIC_DEC_(C, T<Plato::Tet4>); \
PLATO_ELLIPTIC_DEC_(C, T<Plato::Tri3>); \
PLATO_ELLIPTIC_DEC_(C, T<Plato::Tet10>); \
PLATO_ELLIPTIC_DEC_(C, T<Plato::Hex8>); \
PLATO_ELLIPTIC_DEC_(C, T<Plato::Quad4>); \
PLATO_ELLIPTIC_DEC_(C, T<Plato::Hex27>);

#define PLATO_ELLIPTIC_DEF_3_(C, T) \
extern template class C<Plato::Elliptic::ResidualTypes<T>>; \
extern template class C<Plato::Elliptic::JacobianTypes<T>>; \
extern template class C<Plato::Elliptic::GradientXTypes<T>>; \
extern template class C<Plato::Elliptic::GradientZTypes<T>>;

#define PLATO_ELLIPTIC_DEC_3_(C, T) \
template class C<Plato::Elliptic::ResidualTypes<T>>; \
template class C<Plato::Elliptic::JacobianTypes<T>>; \
template class C<Plato::Elliptic::GradientXTypes<T>>; \
template class C<Plato::Elliptic::GradientZTypes<T>>;

#define PLATO_ELLIPTIC_DEF_3(C, T) \
PLATO_ELLIPTIC_DEF_3_(C, T<Plato::Tet4>); \
PLATO_ELLIPTIC_DEF_3_(C, T<Plato::Tri3>); \
PLATO_ELLIPTIC_DEF_3_(C, T<Plato::Tet10>); \
PLATO_ELLIPTIC_DEF_3_(C, T<Plato::Hex8>); \
PLATO_ELLIPTIC_DEF_3_(C, T<Plato::Quad4>); \
PLATO_ELLIPTIC_DEF_3_(C, T<Plato::Hex27>);

#define PLATO_ELLIPTIC_DEC_3(C, T) \
PLATO_ELLIPTIC_DEC_3_(C, T<Plato::Tet4>); \
PLATO_ELLIPTIC_DEC_3_(C, T<Plato::Tri3>); \
PLATO_ELLIPTIC_DEC_3_(C, T<Plato::Tet10>); \
PLATO_ELLIPTIC_DEC_3_(C, T<Plato::Hex8>); \
PLATO_ELLIPTIC_DEC_3_(C, T<Plato::Quad4>); \
PLATO_ELLIPTIC_DEC_3_(C, T<Plato::Hex27>);

#endif
