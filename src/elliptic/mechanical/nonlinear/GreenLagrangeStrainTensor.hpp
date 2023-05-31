/*
 * GreenLagrangeStrainTensor.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

namespace Plato
{

/// @class GreenLagrangeStrainTensor
/// @brief Computes Green-Lagrange strain tensor:
/// \f[ 
///   E_{ij}=\frac{1}{2}\left(F_{ik}^{T}F_{kj}-\delta_{ij}\right)
/// \f]
/// where \f$F_{ij}\f$ is the deformation gradient and \f$\delta_{ij}\f$ is the Kronecker delta
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class GreenLagrangeStrainTensor
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types associated with input evaluation template type
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims  = ElementType::mNumSpatialDims;

public:
  /// @fn operator()()
  /// @brief computes the green-lagrange strain tensor
  /// @param [in]     aStateGrad    state gradient
  /// @param [in,out] aStrainTensor green-lagrange strain tesnor
  /// @return 
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aStateGrad,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aStrainTensor
  ) const
  {
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++)
    {
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++)
      {
        aStrainTensor(tDimI,tDimJ) += 0.5 * ( aStateGrad(tDimI,tDimJ) + aStateGrad(tDimJ,tDimI) );
        for(Plato::OrdinalType tDimK = 0; tDimK < mNumSpatialDims; tDimK++)
        {
          aStrainTensor(tDimI,tDimJ) += 0.5 * ( aStateGrad(tDimK,tDimI) * aStateGrad(tDimK,tDimJ) );
        }
      }
    }
  }
};

}