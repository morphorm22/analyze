/*
 * ThermalDeformationGradient.hpp
 *
 *  Created on: June 17, 2023
 */

#pragma once

#include "elliptic/thermal/FactoryThermalConductionMaterial.hpp"

namespace Plato
{

/// @class ThermalDeformationGradient
/// @brief evaluate thermal deformation gradient:
/// \f[ 
///   F^{\theta}_{ij}=1+\alpha[T-T_0]\mathbf{I}_{ij},\quad i,j=1,\dots,d
/// \f]
/// where \f$d\in\{1,2,3\}, \f$\f$T\f$ denotes temperature, \f$T_0\f$ is the reference 
/// temperature, \f$\mathbf{I}_{ij}\f$ is the second-order identity tensor, \f$\alpha\f$ 
/// is the thermal expansion coefficient, and \f$F^{\theta}_{ij}\f$ is the thermal
/// deformation gradient.
/// @tparam EvaluationType automatic differentiation evaluation type, which sets scalar types 
template<typename EvaluationType>
class ThermalDeformationGradient
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief evaluation scalar types for function range and domain
  using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims = ElementType::mNumSpatialDims;

public:
  /// @brief reference temperature 
  Plato::Scalar mReferenceTemperature;
  /// @brief coefficient of thermal expansion
  Plato::Scalar mThermalExpansivity;

public:
  /// @brief class constructor
  /// @param [in] aMaterialName name of input material parameter list
  /// @param [in] aParamList    input problem parameters
  ThermalDeformationGradient(
    const std::string            & aMaterialName,
    const Teuchos::ParameterList & aParamList
  )
  {
    Plato::FactoryThermalConductionMaterial<EvaluationType> tMaterialFactory(aParamList);
    auto tMaterialModel = tMaterialFactory.create(aMaterialName);
    if( !tMaterialModel->scalarConstantExists("Thermal Expansivity") ){
      auto tMsg = std::string("Material parameter ('Thermal Expansivity') is not defined, thermal deformation ") 
        + "gradient cannot be computed";
      ANALYZE_THROWERR(tMsg)
    }
    mThermalExpansivity = tMaterialModel->getScalarConstant("Thermal Expansivity");

    if( !tMaterialModel->scalarConstantExists("Reference Temperature") ){
      auto tMsg = std::string("Material parameter ('Reference Temperature') is not defined, thermal deformation ") 
        + "gradient cannot be computed";
      ANALYZE_THROWERR(tMsg)
    }
    mReferenceTemperature = tMaterialModel->getScalarConstant("Reference Temperature");
  }

  /// @fn operator()()
  /// @brief compute thermal deformation gradient
  /// @param [in]     aTemp    temperature
  /// @param [in,out] aDefGrad deformation gradient
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    const NodeStateScalarType                                                & aTemp,
          Plato::Matrix<mNumSpatialDims,mNumSpatialDims,NodeStateScalarType> & aDefGrad
  ) const
  {
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      aDefGrad(tDimI,tDimI) += 1.0 + ( mThermalExpansivity * (aTemp - mReferenceTemperature) );
    }
  }
};

} // namespace Plato