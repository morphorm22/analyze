#pragma once


#include <Sacado.hpp>

#include "FadTypes.hpp"

namespace Plato
{

namespace Elliptic
{

template <typename ElementType>
struct EvaluationTypes
{
    static constexpr int NumNodesPerCell = ElementType::mNumNodesPerCell;
    static constexpr int NumControls     = ElementType::mNumControl;
    static constexpr int SpatialDim      = ElementType::mNumSpatialDims;
};

template <typename ElementType>
struct ResidualTypes : EvaluationTypes<ElementType>
{
  using StateScalarType   = Plato::Scalar;
  using ControlScalarType = Plato::Scalar;
  using ConfigScalarType  = Plato::Scalar;
  using ResultScalarType  = Plato::Scalar;
};

template <typename ElementType>
struct JacobianTypes : EvaluationTypes<ElementType>
{
  using SFadType = typename FadTypes<ElementType>::StateFad;

  using StateScalarType   = SFadType;
  using ControlScalarType = Plato::Scalar;
  using ConfigScalarType  = Plato::Scalar;
  using ResultScalarType  = SFadType;
};

template <typename ElementType>
struct GradientXTypes : EvaluationTypes<ElementType>
{
  using SFadType = typename FadTypes<ElementType>::ConfigFad;

  using StateScalarType   = Plato::Scalar;
  using ControlScalarType = Plato::Scalar;
  using ConfigScalarType  = SFadType;
  using ResultScalarType  = SFadType;
};

template <typename ElementType>
struct GradientZTypes : EvaluationTypes<ElementType>
{
  using SFadType = typename FadTypes<ElementType>::ControlFad;

  using StateScalarType   = Plato::Scalar;
  using ControlScalarType = SFadType;
  using ConfigScalarType  = Plato::Scalar;
  using ResultScalarType  = SFadType;
};

template <typename ElementType>
struct Evaluation {
   using Residual  = ResidualTypes<ElementType>;
   using Jacobian  = JacobianTypes<ElementType>;
   using GradientZ = GradientZTypes<ElementType>;
   using GradientX = GradientXTypes<ElementType>;
};

} // namespace Elliptic

} // namespace Plato
