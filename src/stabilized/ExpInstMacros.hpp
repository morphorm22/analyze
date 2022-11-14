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

#include "stabilized/EvaluationTypes.hpp"


#ifdef PLATO_ALL_PENALTY
#define PLATO_STABILIZED_EXP_INST_(C, T) \
template class C<Plato::Stabilized::ResidualTypes<T>, Plato::MSIMP >; \
template class C<Plato::Stabilized::ResidualTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Stabilized::JacobianTypes<T>, Plato::MSIMP >; \
template class C<Plato::Stabilized::JacobianTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Stabilized::JacobianNTypes<T>, Plato::MSIMP >; \
template class C<Plato::Stabilized::JacobianNTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Stabilized::GradientXTypes<T>, Plato::MSIMP >; \
template class C<Plato::Stabilized::GradientXTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Stabilized::GradientZTypes<T>, Plato::MSIMP >; \
template class C<Plato::Stabilized::GradientZTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Stabilized::ResidualTypes<T>, Plato::RAMP >; \
template class C<Plato::Stabilized::ResidualTypes<T>, Plato::Heaviside >; \
template class C<Plato::Stabilized::JacobianTypes<T>, Plato::RAMP >; \
template class C<Plato::Stabilized::JacobianTypes<T>, Plato::Heaviside >; \
template class C<Plato::Stabilized::JacobianNTypes<T>, Plato::RAMP >; \
template class C<Plato::Stabilized::JacobianNTypes<T>, Plato::Heaviside >; \
template class C<Plato::Stabilized::GradientXTypes<T>, Plato::RAMP >; \
template class C<Plato::Stabilized::GradientXTypes<T>, Plato::Heaviside >; \
template class C<Plato::Stabilized::GradientZTypes<T>, Plato::RAMP >; \
template class C<Plato::Stabilized::GradientZTypes<T>, Plato::Heaviside >;
#else
#define PLATO_STABILIZED_EXP_INST_(C, T) \
template class C<Plato::Stabilized::ResidualTypes<T>, Plato::MSIMP >; \
template class C<Plato::Stabilized::ResidualTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Stabilized::JacobianTypes<T>, Plato::MSIMP >; \
template class C<Plato::Stabilized::JacobianTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Stabilized::JacobianNTypes<T>, Plato::MSIMP >; \
template class C<Plato::Stabilized::JacobianNTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Stabilized::GradientXTypes<T>, Plato::MSIMP >; \
template class C<Plato::Stabilized::GradientXTypes<T>, Plato::NoPenalty >; \
template class C<Plato::Stabilized::GradientZTypes<T>, Plato::MSIMP >; \
template class C<Plato::Stabilized::GradientZTypes<T>, Plato::NoPenalty >;
#endif


#ifdef PLATO_HEX_ELEMENTS
  #define PLATO_STABILIZED_EXP_INST(C, T) \
  PLATO_STABILIZED_EXP_INST_(C, T<Plato::Tet4>); \
  PLATO_STABILIZED_EXP_INST_(C, T<Plato::Tri3>); \
  PLATO_STABILIZED_EXP_INST_(C, T<Plato::Tet10>); \
  PLATO_STABILIZED_EXP_INST_(C, T<Plato::Hex8>); \
  PLATO_STABILIZED_EXP_INST_(C, T<Plato::Quad4>); \
  PLATO_STABILIZED_EXP_INST_(C, T<Plato::Hex27>);
#else
  #define PLATO_STABILIZED_EXP_INST(C, T) \
  PLATO_STABILIZED_EXP_INST_(C, T<Plato::Tet4>); \
  PLATO_STABILIZED_EXP_INST_(C, T<Plato::Tri3>); \
  PLATO_STABILIZED_EXP_INST_(C, T<Plato::Tet10>);
#endif


#ifdef PLATO_ALL_PENALTY
#define PLATO_STABILIZED_EXP_INST_2_(C, T, P, D) \
template class C<Plato::Stabilized::ResidualTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::MSIMP >; \
template class C<Plato::Stabilized::ResidualTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::NoPenalty >; \
template class C<Plato::Stabilized::JacobianTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::MSIMP >; \
template class C<Plato::Stabilized::JacobianTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::NoPenalty >; \
template class C<Plato::Stabilized::JacobianNTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::MSIMP >; \
template class C<Plato::Stabilized::JacobianNTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::NoPenalty >; \
template class C<Plato::Stabilized::GradientXTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::MSIMP >; \
template class C<Plato::Stabilized::GradientXTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::NoPenalty >; \
template class C<Plato::Stabilized::GradientZTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::MSIMP >; \
template class C<Plato::Stabilized::GradientZTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::NoPenalty >; \
template class C<Plato::Stabilized::ResidualTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::RAMP >; \
template class C<Plato::Stabilized::ResidualTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::Heaviside >; \
template class C<Plato::Stabilized::JacobianTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::RAMP >; \
template class C<Plato::Stabilized::JacobianTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::Heaviside >; \
template class C<Plato::Stabilized::JacobianNTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::RAMP >; \
template class C<Plato::Stabilized::JacobianNTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::Heaviside >; \
template class C<Plato::Stabilized::GradientXTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::RAMP >; \
template class C<Plato::Stabilized::GradientXTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::Heaviside >; \
template class C<Plato::Stabilized::GradientZTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::RAMP >; \
template class C<Plato::Stabilized::GradientZTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::Heaviside >;
#else
#define PLATO_STABILIZED_EXP_INST_2_(C, T, P, D) \
template class C<Plato::Stabilized::ResidualTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::MSIMP >; \
template class C<Plato::Stabilized::ResidualTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::NoPenalty >; \
template class C<Plato::Stabilized::JacobianTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::MSIMP >; \
template class C<Plato::Stabilized::JacobianTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::NoPenalty >; \
template class C<Plato::Stabilized::JacobianNTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::MSIMP >; \
template class C<Plato::Stabilized::JacobianNTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::NoPenalty >; \
template class C<Plato::Stabilized::GradientXTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::MSIMP >; \
template class C<Plato::Stabilized::GradientXTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::NoPenalty >; \
template class C<Plato::Stabilized::GradientZTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::MSIMP >; \
template class C<Plato::Stabilized::GradientZTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, Plato::NoPenalty >;
#endif

#ifdef PLATO_HEX_ELEMENTS
  #define PLATO_STABILIZED_EXP_INST_2(C, T, P) \
  PLATO_STABILIZED_EXP_INST_2_(C, T, P, Plato::Tet4); \
  PLATO_STABILIZED_EXP_INST_2_(C, T, P, Plato::Tri3); \
  PLATO_STABILIZED_EXP_INST_2_(C, T, P, Plato::Tet10); \
  PLATO_STABILIZED_EXP_INST_2_(C, T, P, Plato::Hex8); \
  PLATO_STABILIZED_EXP_INST_2_(C, T, P, Plato::Quad4); \
  PLATO_STABILIZED_EXP_INST_2_(C, T, P, Plato::Hex27);
#else
  #define PLATO_STABILIZED_EXP_INST_2(C, T, P) \
  PLATO_STABILIZED_EXP_INST_2_(C, T, P, Plato::Tet4); \
  PLATO_STABILIZED_EXP_INST_2_(C, T, P, Plato::Tri3); \
  PLATO_STABILIZED_EXP_INST_2_(C, T, P, Plato::Tet10);
#endif
