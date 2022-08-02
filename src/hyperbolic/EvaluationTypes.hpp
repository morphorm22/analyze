#pragma once

#include <Sacado.hpp>

#include "FadTypes.hpp"

namespace Plato {

namespace Hyperbolic {

template <typename ElementTypeT>
struct EvaluationTypes
{
    static constexpr int NumNodesPerCell = ElementTypeT::mNumNodesPerCell;
    static constexpr int NumControls = ElementTypeT::mNumControl;
    static constexpr int SpatialDim = ElementTypeT::mNumSpatialDims;

    using ElementType = ElementTypeT;
};

template <typename ElementType>
struct ResidualTypes : EvaluationTypes<ElementType>
{
  using StateScalarType       = Plato::Scalar;
  using StateDotScalarType    = Plato::Scalar;
  using StateDotDotScalarType = Plato::Scalar;
  using ControlScalarType     = Plato::Scalar;
  using ConfigScalarType      = Plato::Scalar;
  using ResultScalarType      = Plato::Scalar;
};

template <typename ElementType>
struct GradientUTypes : EvaluationTypes<ElementType>
{
  using SFadType = typename Plato::FadTypes<ElementType>::StateFad;

  using StateScalarType       = SFadType;
  using StateDotScalarType    = Plato::Scalar;
  using StateDotDotScalarType = Plato::Scalar;
  using ControlScalarType     = Plato::Scalar;
  using ConfigScalarType      = Plato::Scalar;
  using ResultScalarType      = SFadType;
};

template <typename ElementType>
struct GradientVTypes : EvaluationTypes<ElementType>
{
  using SFadType = typename Plato::FadTypes<ElementType>::StateFad;

  using StateScalarType       = Plato::Scalar;
  using StateDotScalarType    = SFadType;
  using StateDotDotScalarType = Plato::Scalar;
  using ControlScalarType     = Plato::Scalar;
  using ConfigScalarType      = Plato::Scalar;
  using ResultScalarType      = SFadType;
};

template <typename ElementType>
struct GradientATypes : EvaluationTypes<ElementType>
{
  using SFadType = typename Plato::FadTypes<ElementType>::StateFad;

  using StateScalarType       = Plato::Scalar;
  using StateDotScalarType    = Plato::Scalar;
  using StateDotDotScalarType = SFadType;
  using ControlScalarType     = Plato::Scalar;
  using ConfigScalarType      = Plato::Scalar;
  using ResultScalarType      = SFadType;
};

template <typename ElementType>
struct GradientXTypes : EvaluationTypes<ElementType>
{
  using SFadType = typename Plato::FadTypes<ElementType>::ConfigFad;

  using StateScalarType       = Plato::Scalar;
  using StateDotScalarType    = Plato::Scalar;
  using StateDotDotScalarType = Plato::Scalar;
  using ControlScalarType     = Plato::Scalar;
  using ConfigScalarType      = SFadType;
  using ResultScalarType      = SFadType;
};

template <typename ElementType>
struct GradientZTypes : EvaluationTypes<ElementType>
{
  using SFadType = typename Plato::FadTypes<ElementType>::ControlFad;

  using StateScalarType       = Plato::Scalar;
  using StateDotScalarType    = Plato::Scalar;
  using StateDotDotScalarType = Plato::Scalar;
  using ControlScalarType     = SFadType;
  using ConfigScalarType      = Plato::Scalar;
  using ResultScalarType      = SFadType;
};

template <typename ElementTypeT>
struct Evaluation {
   using Residual  = ResidualTypes<ElementTypeT>;
   using GradientU = GradientUTypes<ElementTypeT>;
   using GradientV = GradientVTypes<ElementTypeT>;
   using GradientA = GradientATypes<ElementTypeT>;
   using GradientX = GradientXTypes<ElementTypeT>;
   using GradientZ = GradientZTypes<ElementTypeT>;
};

} // namespace Hyperbolic
  
} // namespace Plato
