/*
 * DeformationGradient.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

namespace Plato
{

/// @class DeformationGradient
/// @brief Computes deformation gradient:
/// \f[ 
///   F_{ij}=\frac{\partial{x}_i}{\partial{X}_j}=\frac{\partial{u}_i}{\partial{X}_j} + \delta_{ij}
/// \f]
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class DeformationGradient
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types associated with input evaluation template type
  using StateScalarType  = typename EvaluationType::StateScalarType;
  using ConfigScalarType = typename EvaluationType::ConfigScalarType;
  using StrainScalarType = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;

public:
  /// @fn operator()()
  /// @brief compute deformation gradient
  /// @param [in]     aStateGrad state gradient
  /// @param [in,out] aDefGrad   deformation gradient
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aStateGrad,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>  & aDefGrad
  ) const
  {
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
        aDefGrad(tDimI,tDimJ) += aStateGrad(tDimI,tDimJ);
      }
      aDefGrad(tDimI,tDimI) += 1.0;
    }
  }
};

}