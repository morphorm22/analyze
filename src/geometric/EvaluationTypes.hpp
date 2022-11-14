#pragma once

#include "geometric/FadTypes.hpp"


namespace Plato
{

namespace Geometric
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
  using ControlScalarType = Plato::Scalar;
  using ConfigScalarType  = Plato::Scalar;
  using ResultScalarType  = Plato::Scalar;
};

template <typename ElementTypeT>
struct GradientXTypes : EvaluationTypes<ElementTypeT>
{
  using SFadType = typename Plato::Geometry::FadTypes<ElementTypeT>::ConfigFad;

  using ControlScalarType = Plato::Scalar;
  using ConfigScalarType  = SFadType;
  using ResultScalarType  = SFadType;
};

template <typename ElementTypeT>
struct GradientZTypes : EvaluationTypes<ElementTypeT>
{
  using SFadType = typename Plato::Geometry::FadTypes<ElementTypeT>::ControlFad;

  using ControlScalarType = SFadType;
  using ConfigScalarType  = Plato::Scalar;
  using ResultScalarType  = SFadType;
};

template <typename ElementTypeT>
struct Evaluation {
   using Residual  = ResidualTypes<ElementTypeT>;
   using GradientZ = GradientZTypes<ElementTypeT>;
   using GradientX = GradientXTypes<ElementTypeT>;
};

} // namespace Geometric

} // namespace Plato
