#ifndef PARABOLIC_SIMPLEX_FAD_TYPES
#define PARABOLIC_SIMPLEX_FAD_TYPES

#include <Sacado.hpp>

#include "PlatoTypes.hpp"
#include "SimplexFadTypes.hpp"

namespace Plato
{

namespace Parabolic
{

template <typename SimplexPhysicsT>
struct EvaluationTypes
{
    static constexpr int NumNodesPerCell = SimplexPhysicsT::mNumNodesPerCell;
    static constexpr int NumControls = SimplexPhysicsT::mNumControl;
    static constexpr int SpatialDim = SimplexPhysicsT::mNumSpatialDims;
};

template <typename SimplexPhysicsT>
struct ResidualTypes : EvaluationTypes<SimplexPhysicsT>
{
  using StateScalarType    = Plato::Scalar;
  using StateDotScalarType = Plato::Scalar;
  using ControlScalarType  = Plato::Scalar;
  using ConfigScalarType   = Plato::Scalar;
  using ResultScalarType   = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradientUTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::StateFad;

  using StateScalarType    = SFadType;
  using StateDotScalarType = Plato::Scalar;
  using ControlScalarType  = Plato::Scalar;
  using ConfigScalarType   = Plato::Scalar;
  using ResultScalarType   = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientVTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::StateFad;

  using StateScalarType    = Plato::Scalar;
  using StateDotScalarType = SFadType;
  using ControlScalarType  = Plato::Scalar;
  using ConfigScalarType   = Plato::Scalar;
  using ResultScalarType   = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientXTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::ConfigFad;

  using StateScalarType    = Plato::Scalar;
  using StateDotScalarType = Plato::Scalar;
  using ControlScalarType  = Plato::Scalar;
  using ConfigScalarType   = SFadType;
  using ResultScalarType   = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientZTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::ControlFad;

  using StateScalarType    = Plato::Scalar;
  using StateDotScalarType = Plato::Scalar;
  using ControlScalarType  = SFadType;
  using ConfigScalarType   = Plato::Scalar;
  using ResultScalarType   = SFadType;
};


template <typename SimplexPhysicsT>
struct Evaluation {
   using Residual  = ResidualTypes<SimplexPhysicsT>;
   using GradientU = GradientUTypes<SimplexPhysicsT>;
   using GradientV = GradientVTypes<SimplexPhysicsT>;
   using GradientZ = GradientZTypes<SimplexPhysicsT>;
   using GradientX = GradientXTypes<SimplexPhysicsT>;
};
  
} // namespace Parabolic

} // namespace Plato

#endif
