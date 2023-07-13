#pragma once

#include <limits>

#include "AnalyzeMacros.hpp"
#include "PlatoMathTypes.hpp"
#include "materials/MaterialModel.hpp"

namespace Plato
{

/// @brief compute seond-order isotropic stress tensor:
///
/// \f[
///   \sigma_{ij}=C_{ijkl}\epsilon_{kl}\quad (i,j,k,l)\in\{1,2,3\}\in,
/// \f]
///
/// where \f$\sigma_{ij}\f$ is the second-order stress tensor, \f$\epsilon_{kl}\f$ is the second-order strain tensor, 
/// and \f$C_{ijkl}\f$ is the fourth-order material tensor for isotropic elastic materials
/// @tparam EvaluationType 
template< typename EvaluationType>
class ComputeIsotropicElasticStressTensor
{
private:
  /// @brief topological element type
  using ElementType = typename EvaluationType::ElementType;
  /// @brief number of spatial dimensions
  static constexpr auto mNumSpatialDims   = ElementType::mNumSpatialDims;
  /// @brief number of nodes per cell
  static constexpr auto mNumNodesPerCell  = ElementType::mNumNodesPerCell;
  /// @brief lame constants
  Plato::Scalar mMu = 1.0;
  Plato::Scalar mLambda = 1.0;
public:
  /// @brief class constructor
  /// @param [in] aMaterialModel material constitutive model interface 
  ComputeIsotropicElasticStressTensor(
    Plato::MaterialModel<EvaluationType> & aMaterialModel
  )
  {
    mMu = aMaterialModel.getScalarConstant("mu");
    if(mMu <= std::numeric_limits<Plato::Scalar>::epsilon())
    {
      ANALYZE_THROWERR(std::string("Error: Lame constant 'mu' is less than the machine epsilon. ")
          + "The input material properties were not parsed properly.");
    }
    mLambda = aMaterialModel.getScalarConstant("lambda");
    if(mLambda <= std::numeric_limits<Plato::Scalar>::epsilon())
    {
      ANALYZE_THROWERR(std::string("Error: Lame constant 'lambda' is less than the machine epsilon. ")
          + "The input material properties were not parsed properly.");
    }
  }

  /// @fn operator()
  /// @brief compute stress tensor
  /// @tparam ResultScalarType result scalar type
  /// @tparam StrainScalarType strain scalar type
  /// @param [in]     aStrainTensor trial strain tensor
  /// @param [in,out] aStressTensor trial stress tensor
  template<typename ResultScalarType,
           typename StrainScalarType>
  KOKKOS_INLINE_FUNCTION 
  void
  operator()
  (const Plato::Matrix<mNumSpatialDims, mNumSpatialDims, StrainScalarType> & aStrainTensor,
         Plato::Matrix<mNumSpatialDims, mNumSpatialDims, ResultScalarType> & aStressTensor) const
  {
    // compute first strain invariant
    StrainScalarType tFirstStrainInvariant(0.0);
    for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++){
      tFirstStrainInvariant += aStrainTensor(tDim,tDim);
    }
    // add contribution from first stress invariant to the stress tensor
    for(Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; tDim++){
      aStressTensor(tDim,tDim) += mLambda * tFirstStrainInvariant;
    }
    // add shear stress contribution to the stress tensor
    for(Plato::OrdinalType tDimI = 0; tDimI < mNumSpatialDims; tDimI++){
      for(Plato::OrdinalType tDimJ = 0; tDimJ < mNumSpatialDims; tDimJ++){
        aStressTensor(tDimI,tDimJ) += 2.0 * mMu * aStrainTensor(tDimI,tDimJ);
      }
    }
  }
};

} // namespace Plato
