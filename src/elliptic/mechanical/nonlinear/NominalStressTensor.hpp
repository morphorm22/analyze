/*
 * NominalStressTensor.hpp
 *
 *  Created on: June 17, 2023
 */

#pragma once

namespace Plato
{

/// @class NominalStressTensor
/// @brief compute nominal (first piola-kirchhoff) stress tensor:
/// \f[ 
///   P_{ij}=S_{ik}F_{jk} or \mathbf{P}=\mathbf{S}\mathbf{F}^{T}, 
/// \f]
/// where \f$\mathbf{P}\f$ is the first piola-kirchhoff stress, \f$\mathbf{S}\f$ is the second 
/// piola-kirchhoff stress, and \f$\mathbf{F}\f$ is the deformation gradient 
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class NominalStressTensor
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;

public:
  /// @brief class constructor
  NominalStressTensor(){}
  /// @brief class destructor
  ~NominalStressTensor(){}

  /// @fn operator()
  /// @brief compute nominal stress
  /// @tparam DefGradScalarType scalar type for deformation gradient
  /// @tparam KineticScalarType scalar type for second piola-kirchhoff stress
  /// @tparam OutputScalarType  scalar type for first piola-kirchhoff stress (nominal stress)
  /// @param [in]     aDefGrad deformation gradient
  /// @param [in]     a2PKS    second piola-kirchhoff stress
  /// @param [in,out] a1PKS    first piola-kirchhoff stress
  template<typename DefGradScalarType, 
           typename KineticScalarType, 
           typename OutputScalarType>
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,DefGradScalarType> & aDefGrad,
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,KineticScalarType> & a2PKS,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,OutputScalarType>  & a1PKS
  ) const
  {
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
        for(Plato::OrdinalType tDimK = 0; tDimK < mNumSpatialDims; tDimK++){
          a1PKS(tDimI,tDimJ) += a2PKS(tDimI,tDimK) * aDefGrad(tDimJ,tDimK);
        }
      }
    }
  }
};

} // namespace Plato
