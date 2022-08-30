#pragma once

#include <Sacado.hpp>

#include "FadTypes.hpp"

namespace Plato
{

namespace Parabolic
{

template <typename ElementTypeT>
struct EvaluationTypes
{
    static constexpr int NumNodesPerCell = ElementTypeT::mNumNodesPerCell;
    static constexpr int NumControls     = ElementTypeT::mNumControl;
    static constexpr int SpatialDim      = ElementTypeT::mNumSpatialDims;

    using ElementType = ElementTypeT;
};

template <typename ElementTypeT>
struct ResidualTypes : EvaluationTypes<ElementTypeT>
{
  using StateScalarType    = Plato::Scalar;
  using StateDotScalarType = Plato::Scalar;
  using ControlScalarType  = Plato::Scalar;
  using ConfigScalarType   = Plato::Scalar;
  using ResultScalarType   = Plato::Scalar;
};

template <typename ElementTypeT>
struct GradientUTypes : EvaluationTypes<ElementTypeT>
{
  using SFadType = typename Plato::FadTypes<ElementTypeT>::StateFad;

  using StateScalarType    = SFadType;
  using StateDotScalarType = Plato::Scalar;
  using ControlScalarType  = Plato::Scalar;
  using ConfigScalarType   = Plato::Scalar;
  using ResultScalarType   = SFadType;
};

template <typename ElementTypeT>
struct GradientVTypes : EvaluationTypes<ElementTypeT>
{
  using SFadType = typename Plato::FadTypes<ElementTypeT>::StateFad;

  using StateScalarType    = Plato::Scalar;
  using StateDotScalarType = SFadType;
  using ControlScalarType  = Plato::Scalar;
  using ConfigScalarType   = Plato::Scalar;
  using ResultScalarType   = SFadType;
};

template <typename ElementTypeT>
struct GradientXTypes : EvaluationTypes<ElementTypeT>
{
  using SFadType = typename Plato::FadTypes<ElementTypeT>::ConfigFad;

  using StateScalarType    = Plato::Scalar;
  using StateDotScalarType = Plato::Scalar;
  using ControlScalarType  = Plato::Scalar;
  using ConfigScalarType   = SFadType;
  using ResultScalarType   = SFadType;
};

template <typename ElementTypeT>
struct GradientZTypes : EvaluationTypes<ElementTypeT>
{
  using SFadType = typename Plato::FadTypes<ElementTypeT>::ControlFad;

  using StateScalarType    = Plato::Scalar;
  using StateDotScalarType = Plato::Scalar;
  using ControlScalarType  = SFadType;
  using ConfigScalarType   = Plato::Scalar;
  using ResultScalarType   = SFadType;
};


template <typename ElementTypeT>
struct Evaluation {
   using Residual  = ResidualTypes<ElementTypeT>;
   using GradientU = GradientUTypes<ElementTypeT>;
   using GradientV = GradientVTypes<ElementTypeT>;
   using GradientZ = GradientZTypes<ElementTypeT>;
   using GradientX = GradientXTypes<ElementTypeT>;
};
  
} // namespace Parabolic

} // namespace Plato
