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

#ifdef PLATO_HEX_ELEMENTS
#include "Hex8.hpp"
#include "Quad4.hpp"

#include "Hex27.hpp"
#include "Quad9.hpp"
#endif

#include "geometric/EvaluationTypes.hpp"

#define PLATO_GEOMETRIC_EXP_INST_2_(C, T) \
template class C<Plato::Geometric::ResidualTypes<T>>; \
template class C<Plato::Geometric::GradientXTypes<T>>; \
template class C<Plato::Geometric::GradientZTypes<T>>;

#ifdef PLATO_HEX_ELEMENTS
  #define PLATO_GEOMETRIC_EXP_INST(C, T) \
  PLATO_GEOMETRIC_EXP_INST_(C, T<Plato::Tet4>); \
  PLATO_GEOMETRIC_EXP_INST_(C, T<Plato::Tri3>); \
  PLATO_GEOMETRIC_EXP_INST_(C, T<Plato::Tet10>); \
  PLATO_GEOMETRIC_EXP_INST_(C, T<Plato::Hex8>); \
  PLATO_GEOMETRIC_EXP_INST_(C, T<Plato::Quad4>); \
  PLATO_GEOMETRIC_EXP_INST_(C, T<Plato::Hex27>);

  #define PLATO_GEOMETRIC_EXP_INST_2(C, T) \
  PLATO_GEOMETRIC_EXP_INST_2_(C, T<Plato::Tet4>); \
  PLATO_GEOMETRIC_EXP_INST_2_(C, T<Plato::Tri3>); \
  PLATO_GEOMETRIC_EXP_INST_2_(C, T<Plato::Tet10>); \
  PLATO_GEOMETRIC_EXP_INST_2_(C, T<Plato::Hex8>); \
  PLATO_GEOMETRIC_EXP_INST_2_(C, T<Plato::Quad4>); \
  PLATO_GEOMETRIC_EXP_INST_2_(C, T<Plato::Hex27>);
#else
  #define PLATO_GEOMETRIC_EXP_INST(C, T) \
  PLATO_GEOMETRIC_EXP_INST_(C, T<Plato::Tet4>); \
  PLATO_GEOMETRIC_EXP_INST_(C, T<Plato::Tri3>); \
  PLATO_GEOMETRIC_EXP_INST_(C, T<Plato::Tet10>);

  #define PLATO_GEOMETRIC_EXP_INST_2(C, T) \
  PLATO_GEOMETRIC_EXP_INST_2_(C, T<Plato::Tet4>); \
  PLATO_GEOMETRIC_EXP_INST_2_(C, T<Plato::Tri3>); \
  PLATO_GEOMETRIC_EXP_INST_2_(C, T<Plato::Tet10>);
#endif
