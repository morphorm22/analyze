/*
 * KineticPullBackOperation.hpp
 *
 *  Created on: June 17, 2023
 */

#pragma once

namespace Plato
{

/// @class KineticPullBackOperation
/// @brief apply pull back operation to second order kinetic tensor:
/// \f[ 
///   S_{ij}=F^{-1}_{ik}\sigma_{kl}F^{-1}_{jl} J or \mathbf{S}=\mathbf{F}^{-1}\mathbf{\sigma}\mathbf{F}^{-T} J, 
/// \f]
/// where \f$J=\det(F)\f$, \f$S\f$ and \f$\sigma\f$ are generic second order kinetic tensors (e.g., stress), 
/// \f$F\f$ is a generic deformation gradient 
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class KineticPullBackOperation
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;

public:
  /// @brief class constructor
  KineticPullBackOperation(){}

  /// @fn operator()
  /// @brief apply pull back operation to second order kinetic tensor
  /// @tparam DefGradScalarType scalar type for deformation gradient
  /// @tparam KineticScalarType scalar type for second order kinetic tensor
  /// @tparam OutputScalarType  scalar type for output second order kinetic tensor
  /// @param [in]     aDefGrad          deformation gradient
  /// @param [in]     aInKineticTensor  input second order kinetic tensor
  /// @param [in,out] aOutKineticTensor output second order kinetic tensor
  template<typename DefGradScalarType, 
           typename KineticScalarType, 
           typename OutputScalarType>
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,DefGradScalarType> & aDefGrad,
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,KineticScalarType> & aInKineticTensor,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,OutputScalarType>  & aOutKineticTensor
  ) const
  {
    // compute deformation gradient inverse
    Plato::Matrix<mNumSpatialDims,mNumSpatialDims,DefGradScalarType> aInvDefGrad = Plato::invert(aDefGrad);
    // compute determinant of deformation gradient 
    DefGradScalarType tDeterminantDefGrad = Plato::determinant(aDefGrad);
    // apply pull-back operation to kinetic (stress) tensor
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
        for(Plato::OrdinalType tDimK = 0; tDimK < mNumSpatialDims; tDimK++){
          for(Plato::OrdinalType tDimL = 0; tDimL < mNumSpatialDims; tDimL++){
            aOutKineticTensor(tDimI,tDimJ) += tDeterminantDefGrad * 
              aInvDefGrad(tDimI,tDimK) * aInKineticTensor(tDimK,tDimL) * aInvDefGrad(tDimJ,tDimL);
          }
        }
      }
    }
  }
};

} // namespace Plato
