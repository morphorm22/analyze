#pragma once

#include "LinearThermoelasticMaterial.hpp"
#include "VoigtMap.hpp"
#include "MaterialModel.hpp"

namespace Plato
{

namespace Stabilized
{

/******************************************************************************/
/*! Two-field thermoelastics functor.

    given: strain, pressure gradient, temperature gradient, fine scale
    displacement, pressure, and temperature

    compute: deviatoric stress, volume flux, cell stabilization, and thermal flux
*/
/******************************************************************************/
template<typename ElementType>
class TMKinetics : public ElementType
{
  private:

    using ElementType::mNumSpatialDims;
    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;

    const Plato::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellStiffness;
    const Plato::Array<mNumSpatialDims> mCellThermalExpansionCoef;
    const Plato::Matrix<mNumSpatialDims, mNumSpatialDims> mCellThermalConductivity;
    const Plato::Scalar mCellReferenceTemperature;
    Plato::Scalar mBulkModulus, mShearModulus;

    const Plato::Scalar mTemperatureScaling;
    const Plato::Scalar mTemperatureScaling2;

    const Plato::Scalar mPressureScaling;
    const Plato::Scalar mPressureScaling2;

  public:

    TMKinetics( const Teuchos::RCP<Plato::LinearThermoelasticMaterial<mNumSpatialDims>> materialModel ) :
            mCellStiffness(materialModel->getStiffnessMatrix()),
            mCellThermalExpansionCoef(materialModel->getThermalExpansion()),
            mCellThermalConductivity(materialModel->getThermalConductivity()),
            mCellReferenceTemperature(materialModel->getReferenceTemperature()),
            mBulkModulus(0.0), mShearModulus(0.0),
            mTemperatureScaling(materialModel->getTemperatureScaling()),
            mTemperatureScaling2(mTemperatureScaling*mTemperatureScaling),
            mPressureScaling(materialModel->getPressureScaling()),
            mPressureScaling2(mPressureScaling*mPressureScaling)
    {
        for( int iDim=0; iDim<mNumSpatialDims; iDim++ )
        {
            mBulkModulus  += mCellStiffness(0, iDim);
        }
        mBulkModulus /= mNumSpatialDims;

        int tNumShear = mNumVoigtTerms - mNumSpatialDims;
        for( int iShear=0; iShear<tNumShear; iShear++ )
        {
            mShearModulus += mCellStiffness(iShear+mNumSpatialDims, iShear+mNumSpatialDims);
        }
        mShearModulus /= tNumShear;
    }



    /***********************************************************************************
     * \brief Compute deviatoric stress, volume flux, cell stabilization, and thermal flux
     * \param [in] aStrain infinitesimal strain tensor
     * \param [in] aTGrad temperature gradient
     * \param [in] aTemperature temperature
     * \param [out] aStress Cauchy stress tensor
     * \param [out] aFlux thermal flux vector
     **********************************************************************************/
    template<
      typename KineticsScalarType,
      typename KinematicsScalarType,
      typename StateScalarType,
      typename NodeStateScalarType,
      typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        VolumeScalarType                                    const & aVolume,
        Plato::Array<mNumSpatialDims, NodeStateScalarType>  const & aProjectedPGrad,
        Plato::Array<mNumVoigtTerms, KinematicsScalarType>  const & aStrain,
        Plato::Array<mNumSpatialDims, KinematicsScalarType> const & aPressureGrad,
        Plato::Array<mNumSpatialDims, KinematicsScalarType> const & aTGrad,
        StateScalarType                                     const & aTemperature,
        KineticsScalarType                                        & aPressure,
        Plato::Array<mNumVoigtTerms, KineticsScalarType>          & aDevStress,
        KineticsScalarType                                        & aVolumeFlux,
        Plato::Array<mNumSpatialDims, KineticsScalarType>         & aTFlux,
        Plato::Array<mNumSpatialDims, KineticsScalarType>         & aCellStabilization
    ) const
    {
      // compute thermal strain and volume strain
      //
      StateScalarType tstrain[mNumVoigtTerms] = {0};
      StateScalarType tThermalVolStrain = 0.0;
      KinematicsScalarType tVolStrain = 0.0;
      for( int iDim=0; iDim<mNumSpatialDims; iDim++ ){
        tstrain[iDim] = mTemperatureScaling * mCellThermalExpansionCoef(iDim) * (aTemperature - mCellReferenceTemperature);
        tThermalVolStrain += tstrain[iDim];
        tVolStrain += aStrain(iDim);
      }

      // compute deviatoric stress
      //
      for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
        aDevStress(iVoigt) = 0.0;
        for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
          aDevStress(iVoigt) += ( (aStrain(jVoigt)-tstrain[jVoigt]) ) *mCellStiffness(iVoigt, jVoigt);
        }
      }
      KineticsScalarType trace(0.0);
      for( int iDim=0; iDim<mNumSpatialDims; iDim++ ){
        trace += aDevStress(iDim);
      }
      for( int iDim=0; iDim<mNumSpatialDims; iDim++ ){
        aDevStress(iDim) -= trace/3.0;
      }

      // compute flux
      //
      for( int iDim=0; iDim<mNumSpatialDims; iDim++){
        aTFlux(iDim) = 0.0;
        for( int jDim=0; jDim<mNumSpatialDims; jDim++){
          aTFlux(iDim) += mTemperatureScaling2 * aTGrad(jDim)*mCellThermalConductivity(iDim, jDim);
        }
      }

      // compute volume difference
      //
      aPressure *= mPressureScaling;
      aVolumeFlux = mPressureScaling * (tVolStrain - tThermalVolStrain - aPressure/mBulkModulus);

      // compute cell stabilization
      //
      KinematicsScalarType tTau = pow(aVolume,2.0/3.0)/(2.0*mShearModulus);
      for( int iDim=0; iDim<mNumSpatialDims; iDim++){
          aCellStabilization(iDim) = mPressureScaling * tTau *
            (mPressureScaling*aPressureGrad(iDim) - aProjectedPGrad(iDim));
      }
    }
};
} // namespace Stabilized
} // namespace Plato
