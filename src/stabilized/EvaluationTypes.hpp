#pragma once

#include <Sacado.hpp>

#include "FadTypes.hpp"

namespace Plato {

namespace Stabilized {

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
  using StateScalarType     = Plato::Scalar;
  using ConfigScalarType    = Plato::Scalar;
  using ControlScalarType   = Plato::Scalar;
  using ResultScalarType    = Plato::Scalar;
  using NodeStateScalarType = Plato::Scalar;
};

template <typename ElementType>
struct JacobianTypes : EvaluationTypes<ElementType>
{
  using SFadType = typename Plato::FadTypes<ElementType>::StateFad;

  using StateScalarType     = SFadType;
  using ConfigScalarType    = Plato::Scalar;
  using ControlScalarType   = Plato::Scalar;
  using ResultScalarType    = SFadType;
  using NodeStateScalarType = Plato::Scalar;
};

template <typename ElementType>
struct JacobianNTypes : EvaluationTypes<ElementType>
{
  using SFadType = typename Plato::FadTypes<ElementType>::NodeStateFad;

  using StateScalarType     = Plato::Scalar;
  using ConfigScalarType    = Plato::Scalar;
  using ControlScalarType   = Plato::Scalar;
  using ResultScalarType    = SFadType;
  using NodeStateScalarType = SFadType;
};

template <typename ElementType>
struct GradientXTypes : EvaluationTypes<ElementType>
{
  using SFadType = typename Plato::FadTypes<ElementType>::ConfigFad;

  using StateScalarType     = Plato::Scalar;
  using ConfigScalarType    = SFadType;
  using ControlScalarType   = Plato::Scalar;
  using ResultScalarType    = SFadType;
  using NodeStateScalarType = Plato::Scalar;
};

template <typename ElementType>
struct GradientZTypes : EvaluationTypes<ElementType>
{
  using SFadType = typename Plato::FadTypes<ElementType>::ControlFad;

  using StateScalarType     = Plato::Scalar;
  using ConfigScalarType    = Plato::Scalar;
  using ControlScalarType   = SFadType;
  using ResultScalarType    = SFadType;
  using NodeStateScalarType = Plato::Scalar;
};


template <typename ElementTypeT>
struct Evaluation {
   using Residual  = ResidualTypes<ElementTypeT>;
   using Jacobian  = JacobianTypes<ElementTypeT>;
   using JacobianN = JacobianNTypes<ElementTypeT>;
   using GradientZ = GradientZTypes<ElementTypeT>;
   using GradientX = GradientXTypes<ElementTypeT>;
};
  
} // namespace Stabilized
} // namespace Plato
