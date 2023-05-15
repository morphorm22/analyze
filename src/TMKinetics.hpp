#ifndef PLATO_TMKINETICS_HPP
#define PLATO_TMKINETICS_HPP

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
template<typename EvaluationType>
class TMKinetics
{
  private:
      // set local element type
    using ElementType = typename EvaluationType::ElementType;
    
    Plato::MaterialModelType mModelType;

    using ElementType::mNumSpatialDims;
    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;

    Plato::Rank4VoigtConstant<mNumSpatialDims> mElasticStiffnessConstant;
    Plato::Rank4VoigtFunctor<mNumSpatialDims>  mElasticStiffnessFunctor;

    Plato::TensorConstant<mNumSpatialDims> mThermalExpansivityConstant;
    Plato::TensorFunctor<mNumSpatialDims>  mThermalExpansivityFunctor;

    Plato::TensorConstant<mNumSpatialDims> mThermalConductivityConstant;
    Plato::TensorFunctor<mNumSpatialDims>  mThermalConductivityFunctor;

    Plato::Scalar mRefTemperature;

    const Plato::Scalar mScaling;
    const Plato::Scalar mScaling2;

    Plato::VoigtMap<mNumSpatialDims> cVoigtMap;

  public:

    TMKinetics(const Teuchos::RCP<Plato::MaterialModel<EvaluationType>> & aMaterialModel) :
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
    KOKKOS_INLINE_FUNCTION void
    operator()(
        Plato::Array<mNumVoigtTerms,  KineticsScalarType>         & aStress,
        Plato::Array<mNumSpatialDims, KineticsScalarType>         & aFlux,
        Plato::Array<mNumVoigtTerms,  KinematicsScalarType> const & aStrain,
        Plato::Array<mNumSpatialDims, KinematicsScalarType> const & aTGrad,
        StateScalarType                                     const & aTemperature
    ) const
    {
        if (mModelType == Plato::MaterialModelType::Linear)
        {
            // compute thermal strain
            //
            StateScalarType tstrain[mNumVoigtTerms] = {0};
            for( int iDim=0; iDim<mNumSpatialDims; iDim++ ){
                tstrain[iDim] = mScaling * mThermalExpansivityConstant(cVoigtMap.I[iDim], cVoigtMap.J[iDim])
                              * (aTemperature - mRefTemperature);
            }

            // compute stress
            //
            for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
                aStress(iVoigt) = 0.0;
                for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
                    aStress(iVoigt) += (aStrain(jVoigt)-tstrain[jVoigt])*mElasticStiffnessConstant(iVoigt, jVoigt);
                }
            }

            // compute flux
            //
            for( int iDim=0; iDim<mNumSpatialDims; iDim++){
                aFlux(iDim) = 0.0;
                for( int jDim=0; jDim<mNumSpatialDims; jDim++){
                    aFlux(iDim) += mScaling2 * aTGrad(jDim)*mThermalConductivityConstant(iDim, jDim);
                }
            }
        }
        else
        {
            // compute thermal strain
            //
            StateScalarType tstrain[mNumVoigtTerms] = {0};
            for( int iDim=0; iDim<mNumSpatialDims; iDim++ ){
                tstrain[iDim] = mScaling * mThermalExpansivityFunctor(aTemperature, cVoigtMap.I[iDim], cVoigtMap.J[iDim])
                              * (aTemperature - mRefTemperature);
            }

            // compute stress
            //
            for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
                aStress(iVoigt) = 0.0;
                for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
                    aStress(iVoigt) += (aStrain(jVoigt)-tstrain[jVoigt])
                                                  *mElasticStiffnessFunctor(aTemperature, iVoigt, jVoigt);
                }
            }

            // compute flux
            //
            for( int iDim=0; iDim<mNumSpatialDims; iDim++){
                aFlux(iDim) = 0.0;
                for( int jDim=0; jDim<mNumSpatialDims; jDim++){
                    aFlux(iDim) += mScaling2 * aTGrad(jDim)*mThermalConductivityFunctor(aTemperature, iDim, jDim);
                }
            }
        }
    }

};
// class TMKinetics
} // namespace Plato

#endif
