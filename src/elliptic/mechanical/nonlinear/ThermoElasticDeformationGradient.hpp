/*
* ThermoElasticDeformationGradient.hpp
 *
 *  Created on: June 17, 2023
 */

#pragma once

namespace Plato
{

/// @brief compute thermo-elastic deformation gradient via a multiplicative decomposition of the form:
/// \f[ 
///   F_{ij}=F^{\theta}_{ik}F^{u}_{kj},
/// \f]
/// where \f$F_{ij}\f$ is the deformation gradient, \f$F^{\theta}_{ij}\f$ is the thermal deformation gradient,
/// and \f$F^{u}_{ij}\f$ is the mechanical deformation gradient
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types
template<typename EvaluationType>
class ThermoElasticDeformationGradient
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types associated with input evaluation template type
  using StateScalarType     = typename EvaluationType::StateScalarType;
  using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
  using ResultScalarType    = typename EvaluationType::ResultScalarType;
  using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
  using StrainScalarType    = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;

public:
  /// @fn operator()()
  /// @brief compute thermo-elastic deformation gradient
  /// @param [in]     aThermalGrad    thermal deformation gradient
  /// @param [in]     aMechanicalGrad mechanical deformation gradient
  /// @param [in,out] aDefGrad thermo-elastic deformation gradient
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,NodeStateScalarType> & aThermalGrad,
    const Plato::Matrix<mNumSpatialDims,mNumSpatialDims,StrainScalarType>    & aMechanicalGrad,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,ResultScalarType>    & aDefGrad
  ) const
  {
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
        for(Plato::OrdinalType tDimK = 0; tDimK < mNumSpatialDims; tDimK++){
          aDefGrad(tDimI,tDimJ) += aThermalGrad(tDimI,tDimK) * aMechanicalGrad(tDimK,tDimJ);
        }
      }
    }
  }
};

} // namespace Plato
