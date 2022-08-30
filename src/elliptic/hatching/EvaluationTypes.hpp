#pragma once

#include <Sacado.hpp>

#include "FadTypes.hpp"

namespace Plato
{

namespace Elliptic
{

namespace Hatching
{

template <typename ElementTypeT>
struct EvaluationTypes
{
    static constexpr int NumNodesPerCell = ElementTypeT::mNumNodesPerCell;
    static constexpr int NumControls     = ElementTypeT::mNumControl;
    static constexpr int SpatialDim      = ElementTypeT::mNumSpatialDims;

    using ElementType = ElementTypeT;
};

template <typename ElementType>
struct ResidualTypes : EvaluationTypes<ElementType>
{
  using GlobalStateScalarType = Plato::Scalar;
  using LocalStateScalarType  = Plato::Scalar;
  using ControlScalarType     = Plato::Scalar;
  using ConfigScalarType      = Plato::Scalar;
  using ResultScalarType      = Plato::Scalar;
};

template <typename ElementType>
struct JacobianTypes : EvaluationTypes<ElementType>
{
  using SFadType = typename FadTypes<ElementType>::StateFad;

  using GlobalStateScalarType = SFadType;
  using LocalStateScalarType  = Plato::Scalar;
  using ControlScalarType     = Plato::Scalar;
  using ConfigScalarType      = Plato::Scalar;
  using ResultScalarType      = SFadType;
};

template <typename ElementType>
struct GradientCTypes : EvaluationTypes<ElementType>
{
  using SFadType = typename FadTypes<ElementType>::LocalStateFad;

  using GlobalStateScalarType = Plato::Scalar;
  using LocalStateScalarType  = SFadType;
  using ControlScalarType     = Plato::Scalar;
  using ConfigScalarType      = Plato::Scalar;
  using ResultScalarType      = SFadType;
};

template <typename ElementType>
struct GradientXTypes : EvaluationTypes<ElementType>
{
  using SFadType = typename FadTypes<ElementType>::ConfigFad;

  using GlobalStateScalarType = Plato::Scalar;
  using LocalStateScalarType  = Plato::Scalar;
  using ControlScalarType     = Plato::Scalar;
  using ConfigScalarType      = SFadType;
  using ResultScalarType      = SFadType;
};

template <typename ElementType>
struct GradientZTypes : EvaluationTypes<ElementType>
{
  using SFadType = typename FadTypes<ElementType>::ControlFad;

  using GlobalStateScalarType = Plato::Scalar;
  using LocalStateScalarType  = Plato::Scalar;
  using ControlScalarType     = SFadType;
  using ConfigScalarType      = Plato::Scalar;
  using ResultScalarType      = SFadType;
};

template <typename ElementTypeT>
struct Evaluation {
   using Residual  = ResidualTypes<ElementTypeT>;
   using Jacobian  = JacobianTypes<ElementTypeT>;
   using GradientC = GradientCTypes<ElementTypeT>;
   using GradientZ = GradientZTypes<ElementTypeT>;
   using GradientX = GradientXTypes<ElementTypeT>;
};

} // namespace Hatching

} // namespace Elliptic

} // namespace Plato
