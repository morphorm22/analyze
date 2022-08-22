#ifndef PLATO_TMKINETICS_HPP
#define PLATO_TMKINETICS_HPP

#include "SimplexThermomechanics.hpp"
#include "LinearThermoelasticMaterial.hpp"
#include "VoigtMap.hpp"
#include "MaterialModel.hpp"

namespace Plato
{

/******************************************************************************/
/*! Thermoelastics functor.

    given a strain, temperature gradient, and temperature, compute the stress and flux
*/
/******************************************************************************/
template<int SpaceDim>
class TMKinetics : public Plato::SimplexThermomechanics<SpaceDim>
{
  private:
    Plato::MaterialModelType mModelType;

    using Plato::SimplexThermomechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumDofsPerNode;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumDofsPerCell;

    Plato::Rank4VoigtConstant<SpaceDim> mElasticStiffnessConstant;
    Plato::Rank4VoigtFunctor<SpaceDim>  mElasticStiffnessFunctor;

    Plato::TensorConstant<SpaceDim> mThermalExpansivityConstant;
    Plato::TensorFunctor<SpaceDim>  mThermalExpansivityFunctor;

    Plato::TensorConstant<SpaceDim> mThermalConductivityConstant;
    Plato::TensorFunctor<SpaceDim>  mThermalConductivityFunctor;

    Plato::Scalar mRefTemperature;

    const Plato::Scalar mScaling;
    const Plato::Scalar mScaling2;

    Plato::VoigtMap<SpaceDim> cVoigtMap;

  public:

    TMKinetics(const Teuchos::RCP<Plato::MaterialModel<SpaceDim>> aMaterialModel) :
      mRefTemperature(aMaterialModel->getScalarConstant("Reference Temperature")),
      mScaling(aMaterialModel->getScalarConstant("Temperature Scaling")),
      mScaling2(mScaling*mScaling)
    {
        mModelType = aMaterialModel->type();
        if (mModelType == Plato::MaterialModelType::Nonlinear)
        {
            mElasticStiffnessFunctor = aMaterialModel->getRank4VoigtFunctor("Elastic Stiffness");
            mThermalExpansivityFunctor = aMaterialModel->getTensorFunctor("Thermal Expansivity");
            mThermalConductivityFunctor = aMaterialModel->getTensorFunctor("Thermal Conductivity");
        } else
        if (mModelType == Plato::MaterialModelType::Linear)
        {
            mElasticStiffnessConstant = aMaterialModel->getRank4VoigtConstant("Elastic Stiffness");
            mThermalExpansivityConstant = aMaterialModel->getTensorConstant("Thermal Expansivity");
            mThermalConductivityConstant = aMaterialModel->getTensorConstant("Thermal Conductivity");
        }
    }

    /***********************************************************************************
     * \brief Compute stress and thermal flux from strain, temperature, and temperature gradient
     * \param [in] aStrain infinitesimal strain tensor
     * \param [in] aTGrad temperature gradient
     * \param [in] aTemperature temperature
     * \param [out] aStress Cauchy stress tensor
     * \param [out] aFlux thermal flux vector
     **********************************************************************************/
    template<typename KineticsScalarType, typename KinematicsScalarType, typename StateScalarType>
    KOKKOS_FUNCTION inline void
    operator()( int cellOrdinal,
                Kokkos::View<KineticsScalarType**,   Plato::Layout, Plato::MemSpace> const& aStress,
                Kokkos::View<KineticsScalarType**,   Plato::Layout, Plato::MemSpace> const& aFlux,
                Kokkos::View<KinematicsScalarType**, Plato::Layout, Plato::MemSpace> const& aStrain,
                Kokkos::View<KinematicsScalarType**, Plato::Layout, Plato::MemSpace> const& aTGrad,
                Kokkos::View<StateScalarType*,       Plato::MemSpace> const& aTemperature) const
    {
        StateScalarType tTemperature = aTemperature(cellOrdinal);
        if (mModelType == Plato::MaterialModelType::Linear)
        {
            // compute thermal strain
            //
            StateScalarType tstrain[mNumVoigtTerms] = {0};
            for( int iDim=0; iDim<SpaceDim; iDim++ ){
                tstrain[iDim] = mScaling * mThermalExpansivityConstant(cVoigtMap.I[iDim], cVoigtMap.J[iDim])
                              * (tTemperature - mRefTemperature);
            }

            // compute stress
            //
            for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
                aStress(cellOrdinal,iVoigt) = 0.0;
                for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
                    aStress(cellOrdinal,iVoigt) += (aStrain(cellOrdinal,jVoigt)-tstrain[jVoigt])*mElasticStiffnessConstant(iVoigt, jVoigt);
                }
            }

            // compute flux
            //
            for( int iDim=0; iDim<SpaceDim; iDim++){
                aFlux(cellOrdinal,iDim) = 0.0;
                for( int jDim=0; jDim<SpaceDim; jDim++){
                    aFlux(cellOrdinal,iDim) += mScaling2 * aTGrad(cellOrdinal,jDim)*mThermalConductivityConstant(iDim, jDim);
                }
            }
        }
        else
        {
            // compute thermal strain
            //
            StateScalarType tstrain[mNumVoigtTerms] = {0};
            for( int iDim=0; iDim<SpaceDim; iDim++ ){
                tstrain[iDim] = mScaling * mThermalExpansivityFunctor(tTemperature, cVoigtMap.I[iDim], cVoigtMap.J[iDim])
                              * (tTemperature - mRefTemperature);
            }

            // compute stress
            //
            for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
                aStress(cellOrdinal,iVoigt) = 0.0;
                for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
                    aStress(cellOrdinal,iVoigt) += (aStrain(cellOrdinal,jVoigt)-tstrain[jVoigt])
                                                  *mElasticStiffnessFunctor(tTemperature, iVoigt, jVoigt);
                }
            }

            // compute flux
            //
            for( int iDim=0; iDim<SpaceDim; iDim++){
                aFlux(cellOrdinal,iDim) = 0.0;
                for( int jDim=0; jDim<SpaceDim; jDim++){
                    aFlux(cellOrdinal,iDim) += mScaling2 * aTGrad(cellOrdinal,jDim)*mThermalConductivityFunctor(tTemperature, iDim, jDim);
                }
            }
        }
    }

};
// class TMKinetics


/******************************************************************************/
/*! Two-field thermoelastics functor.

    given: strain, pressure gradient, temperature gradient, fine scale
    displacement, pressure, and temperature

    compute: deviatoric stress, volume flux, cell stabilization, and thermal flux
*/
/******************************************************************************/
template<int SpaceDim>
class StabilizedTMKinetics : public Plato::SimplexStabilizedThermomechanics<SpaceDim>
{
  private:

    using Plato::SimplexStabilizedThermomechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexStabilizedThermomechanics<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexStabilizedThermomechanics<SpaceDim>::mNumDofsPerNode;
    using Plato::SimplexStabilizedThermomechanics<SpaceDim>::mNumDofsPerCell;

    const Plato::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellStiffness;
    const Plato::Array<SpaceDim> mCellThermalExpansionCoef;
    const Plato::Matrix<SpaceDim, SpaceDim> mCellThermalConductivity;
    const Plato::Scalar mCellReferenceTemperature;
    Plato::Scalar mBulkModulus, mShearModulus;

    const Plato::Scalar mTemperatureScaling;
    const Plato::Scalar mTemperatureScaling2;

    const Plato::Scalar mPressureScaling;
    const Plato::Scalar mPressureScaling2;

  public:

    StabilizedTMKinetics( const Teuchos::RCP<Plato::LinearThermoelasticMaterial<SpaceDim>> materialModel ) :
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
        for( int iDim=0; iDim<SpaceDim; iDim++ )
        {
            mBulkModulus  += mCellStiffness(0, iDim);
        }
        mBulkModulus /= SpaceDim;

        int tNumShear = mNumVoigtTerms - SpaceDim;
        for( int iShear=0; iShear<tNumShear; iShear++ )
        {
            mShearModulus += mCellStiffness(iShear+SpaceDim, iShear+SpaceDim);
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
    KOKKOS_FUNCTION inline void
    operator()( int cellOrdinal,
                Plato::ScalarVectorT      <VolumeScalarType>     const& aCellVolume,
                Plato::ScalarMultiVectorT <NodeStateScalarType>  const& aProjectedPGrad,
                Plato::ScalarVectorT      <KineticsScalarType>   const& aPressure,
                Plato::ScalarVectorT      <StateScalarType>      const& aTemperature,
                Plato::ScalarMultiVectorT <KinematicsScalarType> const& aStrain,
                Plato::ScalarMultiVectorT <KinematicsScalarType> const& aPressureGrad,
                Plato::ScalarMultiVectorT <KinematicsScalarType> const& aTGrad,
                Plato::ScalarMultiVectorT <KineticsScalarType>   const& aDevStress,
                Plato::ScalarVectorT      <KineticsScalarType>   const& aVolumeFlux,
                Plato::ScalarMultiVectorT <KineticsScalarType>   const& aTFlux,
                Plato::ScalarMultiVectorT <KineticsScalarType>   const& aCellStabilization) const {

      // compute thermal strain and volume strain
      //
      StateScalarType tstrain[mNumVoigtTerms] = {0};
      StateScalarType tThermalVolStrain = 0.0;
      KinematicsScalarType tVolStrain = 0.0;
      for( int iDim=0; iDim<SpaceDim; iDim++ ){
        tstrain[iDim] = mTemperatureScaling * mCellThermalExpansionCoef(iDim) * (aTemperature(cellOrdinal) - mCellReferenceTemperature);
        tThermalVolStrain += tstrain[iDim];
        tVolStrain += aStrain(cellOrdinal,iDim);
      }

      // compute deviatoric stress
      //
      for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
        aDevStress(cellOrdinal,iVoigt) = 0.0;
        for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
          aDevStress(cellOrdinal,iVoigt) += ( (aStrain(cellOrdinal,jVoigt)-tstrain[jVoigt]) ) *mCellStiffness(iVoigt, jVoigt);
        }
      }
      KineticsScalarType trace(0.0);
      for( int iDim=0; iDim<SpaceDim; iDim++ ){
        trace += aDevStress(cellOrdinal,iDim);
      }
      for( int iDim=0; iDim<SpaceDim; iDim++ ){
        aDevStress(cellOrdinal,iDim) -= trace/3.0;
      }

      // compute flux
      //
      for( int iDim=0; iDim<SpaceDim; iDim++){
        aTFlux(cellOrdinal,iDim) = 0.0;
        for( int jDim=0; jDim<SpaceDim; jDim++){
          aTFlux(cellOrdinal,iDim) += mTemperatureScaling2 * aTGrad(cellOrdinal,jDim)*mCellThermalConductivity(iDim, jDim);
        }
      }

      // compute volume difference
      //
      aPressure(cellOrdinal) *= mPressureScaling;
      aVolumeFlux(cellOrdinal) = mPressureScaling * (tVolStrain - tThermalVolStrain - aPressure(cellOrdinal)/mBulkModulus);

      // compute cell stabilization
      //
      KinematicsScalarType tTau = pow(aCellVolume(cellOrdinal),2.0/3.0)/(2.0*mShearModulus);
      for( int iDim=0; iDim<SpaceDim; iDim++){
          aCellStabilization(cellOrdinal,iDim) = mPressureScaling * tTau *
            (mPressureScaling*aPressureGrad(cellOrdinal,iDim) - aProjectedPGrad(cellOrdinal,iDim));
      }
    }
};

} // namespace Plato

#endif
