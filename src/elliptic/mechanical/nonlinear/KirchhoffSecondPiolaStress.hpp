/*
 * FactoryNonlinearElasticMaterial.hpp
 *
 *  Created on: May 31, 2023
 */

#pragma once

#include <string>

#include "MaterialModel.hpp"
#include "elliptic/EvaluationTypes.hpp"

namespace Plato
{

/// @class KirchhoffSecondPiolaStress
/// @brief Evaluate second Piola-Kirchhoff stress tensor for a Kirchhoff material: \n
///  \f[
///    \mathbf{S}=\lambda\mbox{trace}(\mathbf{E})\mathbf{I} + 2\mu\mathbf{E},
///  \f]
/// where \f$\lambda\f$ and \f$\mu\f$ are the Lame constants, \f$\mathbf{E}\f$ is \n
/// the Green-Lagrange strain tensor, and \f$\mathbf{I}\f$ is the second order \n
/// identity tensor.
/// @tparam EvaluationType 
template<typename EvaluationType>
class KirchhoffSecondPiolaStress
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

public:
  /// @brief class constructor
  /// @param [in] aMaterial material model interface
  KirchhoffSecondPiolaStress(
    const Plato::MaterialModel<EvaluationType> & aMaterial
  )
  {
    mMu     = std::stod(aMaterial.property("lame mu").front());
    mLambda = std::stod(aMaterial.property("lame lambda").front());
  }

  /// @brief class destructor
  ~KirchhoffSecondPiolaStress(){}

  /// @fn operator()()
  /// @brief Compute second Piola-Kirchhoff stress tensor
  /// @param [in]     aStrainTensor Green-Lagrange strain tensor 
  /// @param [in,out] aStressTensor second Piola-Kirchhoff stress tensor
  KOKKOS_INLINE_FUNCTION
  void 
  operator()(
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,StrainScalarType> & aStrainTensor, 
    Plato::Matrix<ElementType::mNumSpatialDims,ElementType::mNumSpatialDims,ResultScalarType> & aStressTensor 
  ) const
  {
    StrainScalarType tTrace = Plato::trace(aStrainTensor);
    for(Plato::OrdinalType tDimI = 0; tDimI < ElementType::mNumSpatialDims; tDimI++){
      aStressTensor(tDimI,tDimI) += mLambda*tTrace;
      for(Plato::OrdinalType tDimJ = 0; tDimJ < ElementType::mNumSpatialDims; tDimJ++){
        aStressTensor(tDimI,tDimJ) += 2.0*mMu*aStrainTensor(tDimI,tDimJ);
      }
    }
  }
};

}