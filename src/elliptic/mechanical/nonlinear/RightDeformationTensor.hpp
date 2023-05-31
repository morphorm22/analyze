/*
 * RightDeformationTensor.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

namespace Plato
{

/// @class RightDeformationTensor
/// @brief Computes right deformation tensor: 
/// \f[ 
///   C_{ij}=F_{ik}^{T}F_{kj}
/// \f]
/// where \f$F_{ij}\f$ is the deformation gradient \n
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class RightDeformationTensor
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
  /// @brief 
  /// @param [in]     aDefGradT  deformation gradient transpose
  /// @param [in]     aDefGrad   deformation gradient 
  /// @param [in,out] aDefTensor deformation tensor 
  /// @return 
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aDefGradT,
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aDefGrad,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aDefTensor
  ) const
  {
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
        for(Plato::OrdinalType tDimK = 0; tDimK < mNumSpatialDims; tDimK++){
          aDefTensor(tDimI,tDimJ) += aDefGradT(tDimI,tDimK) * aDefGrad(tDimK,tDimJ);
        }
      }
    }
  }
};

}