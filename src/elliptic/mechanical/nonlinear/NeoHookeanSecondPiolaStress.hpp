/*
 * NeoHookeanSecondPiolaStress.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include "PlatoMathTypes.hpp"
#include "materials/MaterialModel.hpp"
#include "elliptic/mechanical/nonlinear/RightDeformationTensor.hpp"

namespace Plato
{

/// @class NeoHookeanSecondPiolaStress
/// @brief Evaluate second Piola-Kirchhoff stress tensor for a Neo-Hookean material: \n
///  \f[
///    \mathbf{S}=\lambda\ln(J)\mathbf{C}^{-1} + \mu(\mathbf{I}-\mathbf{C}^{-1}),
///  \f]
/// where \f$\lambda\f$ and \f$\mu\f$ are the Lame constants, \f$J=\det(\mathbf{F})\f$, \n 
/// \f$\mathbf{F}\f is the deformation gradient, \f$\mathbf{C}\f$ is the right deformation \n
/// tensor, and \f$\mathbf{I}\f$ is the second order identity tensor. \n
/// @tparam EvaluationType 
template<typename EvaluationType>
class NeoHookeanSecondPiolaStress
{
private:
  /// @brief topological element typename
  using ElementType = typename EvaluationType::ElementType;
  /// @brief scalar types for an evaluation type
  using StateScalarType   = typename EvaluationType::StateScalarType;
  using ControlScalarType = typename EvaluationType::ControlScalarType;
  using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
  using ResultScalarType  = typename EvaluationType::ResultScalarType;
  using StrainScalarType  = typename Plato::fad_type_t<ElementType,StateScalarType,ConfigScalarType>;
  /// @brief Lame constant \f$\mu\f$
  Plato::Scalar mMu;
  /// @brief Lame constant \f$\lambda\f$
  Plato::Scalar mLambda;
  /// @brief computes right deformation tensor 
  Plato::RightDeformationTensor<EvaluationType> mComputeRightDeformationTensor;

public:
  /// @brief class constructor
  /// @param [in] aMaterial material model interface
  NeoHookeanSecondPiolaStress(
    const Plato::MaterialModel<EvaluationType> & aMaterial
  )
  {
    mMu     = std::stod(aMaterial.property("lame mu").front());
    mLambda = std::stod(aMaterial.property("lame lambda").front());
  }

  /// @brief class destructor
  ~NeoHookeanSecondPiolaStress(){}

  /// @fn operator()()
  /// @brief Compute second Piola-Kirchhoff stress tensor
  /// @param [in]     aDefGradient  deformation gradient  
  /// @param [in,out] aStressTensor second Piola-Kirchhoff stress tensor
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainScalarType> & aDefGradient, 
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultScalarType> & aStressTensor 
  ) const
  {
    // apply transpose to deformation gradient
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainScalarType> tDefGradientT = 
      Plato::transpose(aDefGradient);
    // compute cauchy-green deformation tensor
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainScalarType> 
      tRightDeformationTensor(StrainScalarType(0.));
    mComputeRightDeformationTensor(tDefGradientT,aDefGradient,tRightDeformationTensor);
    // compute determinant of deformation gradient
    StrainScalarType tDetDefGrad = Plato::determinant(aDefGradient);
    // invert right deformation tensor
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainScalarType> 
      tInverseRightDeformationTensor = Plato::invert(tRightDeformationTensor);
    // compute stress tensor
    for(Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      aStressTensor(tDimI,tDimI) += mMu;
      for(Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        aStressTensor(tDimI,tDimJ) += mLambda*log(tDetDefGrad)*tInverseRightDeformationTensor(tDimI,tDimJ);
        aStressTensor(tDimI,tDimJ) -= mMu*tInverseRightDeformationTensor(tDimI,tDimJ);
      }
    }
  }
};

}