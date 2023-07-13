/*
 * ComputeStrainTensor.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include "PlatoMathTypes.hpp"

namespace Plato
{

/// @class ComputeStrainTensor
/// @brief compute strain tensor for small strains:
///
/// \f[
///   \epsilon_{ij}=\left( \frac{\partial u_i}{partial x_j} + \frac{\partial u_j}{partial x_i} \right)
/// \f]
///
/// where \f$u_i\f$ is the i-th displacement component and \f$x_i\f$ is the i-th spatial coordinate
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class ComputeStrainTensor
{
private:
  /// @brief local topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;
  /// @brief number of nodes per element
  static constexpr auto mNumNodesPerCell = ElementType::mNumNodesPerCell; 

public:
  /// @fn operator()
  /// @brief compute test (virtual) strain tensor, a constant value of 1.0 is assumed 
  /// for each virtual displacement component
  /// @tparam StrainScalarType   strains scalar type
  /// @tparam GradientScalarType gradient scalar type
  /// @param [in]     aCellOrdinal    local element ordinal
  /// @param [in]     aGradient       interpolation function gradient matrix
  /// @param [in,out] aVirtualStrains test strain tensor
  template
  <typename StrainScalarType,
   typename GradientScalarType>
  KOKKOS_INLINE_FUNCTION 
  void
  operator()(
    const Plato::OrdinalType                                                  & aCellOrdinal,
    const Plato::Matrix<mNumNodesPerCell,mNumSpatialDims, GradientScalarType> & aGradient,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims, StrainScalarType>    & aVirtualStrains
  ) const
  {
    constexpr Plato::Scalar tDeltaState = 1.0;
    constexpr Plato::Scalar tOneOverTwo = 0.5;
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
    {
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
      {
        for(Plato::OrdinalType tNodeI = 0; tNodeI < mNumNodesPerCell; tNodeI++)
        {
          aVirtualStrains(tDimI,tDimJ) += tOneOverTwo *
              ( ( tDeltaState * aGradient(tNodeI, tDimI) )
              + ( tDeltaState * aGradient(tNodeI, tDimJ) ) );
        }
      }
    }
  }

  /// @fn operator()
  /// @brief compute trial strain tensor
  /// @tparam StrainScalarType   trial strain scalar type
  /// @tparam DispScalarType     displacement scalar type
  /// @tparam GradientScalarType gradient scalar type
  /// @param [in]     aCellOrdinal local element ordinal
  /// @param [in]     aState       displacement workset
  /// @param [in]     aGradient    interpolation function gradient matrix
  /// @param [in,out] aStrains     trial strain tensor
  template
  <typename StrainScalarType,
   typename DispScalarType,
   typename GradientScalarType>
  KOKKOS_INLINE_FUNCTION 
  void
  operator()(
    const Plato::OrdinalType                                                  & aCellOrdinal,
    const Plato::ScalarMultiVectorT<DispScalarType>                           & aState,
    const Plato::Matrix<mNumNodesPerCell,mNumSpatialDims, GradientScalarType> & aGradient,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims, StrainScalarType>    & aStrains
  ) const
  {
    constexpr Plato::Scalar tOneOverTwo = 0.5;
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
    {
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
      {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
          auto tLocalOrdinalI = tNodeIndex * mNumSpatialDims + tDimI;
          auto tLocalOrdinalJ = tNodeIndex * mNumSpatialDims + tDimJ;
          aStrains(tDimI,tDimJ) += tOneOverTwo *
            ( aState(aCellOrdinal, tLocalOrdinalJ) * aGradient(tNodeIndex, tDimI)
            + aState(aCellOrdinal, tLocalOrdinalI) * aGradient(tNodeIndex, tDimJ));
        }
      }
    }
  }
};

} // namespace Plato
