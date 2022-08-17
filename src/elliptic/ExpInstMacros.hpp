#pragma once

#include "Simp.hpp"
#include "NoPenalty.hpp"

#ifdef PLATO_ALL_PENALTY
#include "Ramp.hpp"
#include "Heaviside.hpp"
#endif

#include "Tet4.hpp"
#include "Tri3.hpp"
#include "Bar2.hpp"
#include "Tet10.hpp"
#include "Tri6.hpp"

#ifdef PLATO_HEX_ELEMENTS
#include "Hex8.hpp"
#include "Quad4.hpp"
#include "Hex27.hpp"
#include "Quad9.hpp"
#endif

#include "elliptic/EvaluationTypes.hpp"


#ifdef PLATO_ALL_PENALTY
#define PLATO_ELLIPTIC_EXP_INST_(C, T) \
template class C<Plato::Elliptic::ResidualTypes<T>, Plato::MSIMP >; \
template class C<Plato::Elliptic::ResidualTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Elliptic::ResidualTypes<T>, Plato::RAMP >; \
template class C<Plato::Elliptic::ResidualTypes<T>, Plato::Heaviside >; \
template class C<Plato::Elliptic::JacobianTypes<T>, Plato::MSIMP >; \
template class C<Plato::Elliptic::JacobianTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Elliptic::JacobianTypes<T>, Plato::RAMP >; \
template class C<Plato::Elliptic::JacobianTypes<T>, Plato::Heaviside >; \
template class C<Plato::Elliptic::GradientXTypes<T>, Plato::MSIMP >; \
template class C<Plato::Elliptic::GradientXTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Elliptic::GradientXTypes<T>, Plato::RAMP >; \
template class C<Plato::Elliptic::GradientXTypes<T>, Plato::Heaviside >; \
template class C<Plato::Elliptic::GradientZTypes<T>, Plato::MSIMP >; \
template class C<Plato::Elliptic::GradientZTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Elliptic::GradientZTypes<T>, Plato::RAMP >; \
template class C<Plato::Elliptic::GradientZTypes<T>, Plato::Heaviside >;
#else
#define PLATO_ELLIPTIC_EXP_INST_(C, T) \
template class C<Plato::Elliptic::ResidualTypes<T>, Plato::MSIMP >; \
template class C<Plato::Elliptic::ResidualTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Elliptic::JacobianTypes<T>, Plato::MSIMP >; \
template class C<Plato::Elliptic::JacobianTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Elliptic::GradientXTypes<T>, Plato::MSIMP >; \
template class C<Plato::Elliptic::GradientXTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Elliptic::GradientZTypes<T>, Plato::MSIMP >; \
template class C<Plato::Elliptic::GradientZTypes<T>, Plato::NoPenalty >;
#endif

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

