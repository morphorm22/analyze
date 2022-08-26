#pragma once

#include "PlatoMathTypes.hpp"
#include "hyperbolic/micromorphic/MicromorphicLinearElasticMaterial.hpp"
#include "hyperbolic/micromorphic/MicromorphicInertiaMaterial.hpp"

namespace Plato
{

namespace Hyperbolic
{

template<typename ElementType>
class MicromorphicKinetics : public ElementType
{
  private:

    using ElementType::mNumSpatialDims;
    using ElementType::mNumVoigtTerms;
    using ElementType::mNumSkwTerms;

    const Plato::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellMesoStressSymmetricMaterialTensor; 
    const Plato::Matrix<mNumSkwTerms,mNumSkwTerms>     mCellMesoStressSkewMaterialTensor; 
    const Plato::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellMicroStressSymmetricMaterialTensor; 
    const Plato::Matrix<mNumSkwTerms,mNumSkwTerms>     mCellMicroStressSkewMaterialTensor; 

  public:

    MicromorphicKinetics(const Teuchos::RCP<Plato::MicromorphicLinearElasticMaterial<mNumSpatialDims>> aMaterialModel) :
      mCellMesoStressSymmetricMaterialTensor(aMaterialModel->getStiffnessMatrixCe()),
      mCellMesoStressSkewMaterialTensor(aMaterialModel->getStiffnessMatrixCc()),
      mCellMicroStressSymmetricMaterialTensor(aMaterialModel->getStiffnessMatrixCm())
    {
    }
    
    MicromorphicKinetics(const Teuchos::RCP<Plato::MicromorphicInertiaMaterial<mNumSpatialDims>> aMaterialModel) :
      mCellMesoStressSymmetricMaterialTensor(aMaterialModel->getInertiaMatrixTe()),
      mCellMesoStressSkewMaterialTensor(aMaterialModel->getInertiaMatrixTc()),
      mCellMicroStressSymmetricMaterialTensor(aMaterialModel->getInertiaMatrixJm()),
      mCellMicroStressSkewMaterialTensor(aMaterialModel->getInertiaMatrixJc())
    {
    }

    // overloaded for cauchy and micro stresses
    template<typename KineticsScalarType, typename KinematicsScalarType, typename StateScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()( 
              Plato::Array<mNumVoigtTerms, KineticsScalarType>   & aSymmetricMesoStress,
              Plato::Array<mNumVoigtTerms, KineticsScalarType>   & aSkewMesoStress,
              Plato::Array<mNumVoigtTerms, KineticsScalarType>   & aSymmetricMicroStress,
        const Plato::Array<mNumVoigtTerms, KinematicsScalarType> & aSymmetricGradientStrain,
        const Plato::Array<mNumSkwTerms, KinematicsScalarType>   & aSkewGradientStrain,
        const Plato::Array<mNumVoigtTerms, StateScalarType>      & aSymmetricMicroStrain,
        const Plato::Array<mNumSkwTerms, StateScalarType>        & aSkewMicroStrain) const
    {
        this->initializeStressesWithZeros(aSymmetricMesoStress,aSkewMesoStress,aSymmetricMicroStress);
        this->addSymmetricMesoStressTerm(aSymmetricMesoStress,aSymmetricGradientStrain,1.0);
        this->addSymmetricMesoStressTerm(aSymmetricMesoStress,aSymmetricMicroStrain,-1.0);
        this->addSkewMesoStressTerm(aSkewMesoStress,aSkewGradientStrain,1.0);
        this->addSkewMesoStressTerm(aSkewMesoStress,aSkewMicroStrain,-1.0);
        this->addSymmetricMicroStressTerm(aSymmetricMicroStress,aSymmetricMicroStrain);
    }
    
    // overloaded for inertia stresses
    template<typename KineticsScalarType, typename KinematicsScalarType, typename StateScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()( 
              Plato::Array<mNumVoigtTerms, KineticsScalarType>   & aSymmetricMesoStress,
              Plato::Array<mNumVoigtTerms, KineticsScalarType>   & aSkewMesoStress,
              Plato::Array<mNumVoigtTerms, KineticsScalarType>   & aSymmetricMicroStress,
              Plato::Array<mNumVoigtTerms, KineticsScalarType>   & aSkewMicroStress,
        const Plato::Array<mNumVoigtTerms, KinematicsScalarType> & aSymmetricGradientStrain,
        const Plato::Array<mNumSkwTerms, KinematicsScalarType>   & aSkewGradientStrain,
        const Plato::Array<mNumVoigtTerms, StateScalarType>      & aSymmetricMicroStrain,
        const Plato::Array<mNumSkwTerms, StateScalarType>        & aSkewMicroStrain) const
    {
        this->initializeStressesWithZeros(aSymmetricMesoStress,aSkewMesoStress,aSymmetricMicroStress);
        this->addSymmetricMesoStressTerm(aSymmetricMesoStress,aSymmetricGradientStrain,1.0);
        this->addSkewMesoStressTerm(aSkewMesoStress,aSkewGradientStrain,1.0);
        this->addSymmetricMicroStressTerm(aSymmetricMicroStress,aSymmetricMicroStrain);
        this->addSkewMicroStressTerm(aSkewMicroStress,aSkewMicroStrain);
    }

  private:

    // overloaded for cauchy and micro stresses
    template<typename KineticsScalarType>
    KOKKOS_INLINE_FUNCTION void
    initializeStressesWithZeros( 
              Plato::Array<mNumVoigtTerms, KineticsScalarType>   & aSymmetricMesoStress,
              Plato::Array<mNumVoigtTerms, KineticsScalarType>   & aSkewMesoStress,
              Plato::Array<mNumVoigtTerms, KineticsScalarType>   & aSymmetricMicroStress) const
    {
        for( Plato::OrdinalType iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++)
        {
            aSymmetricMesoStress(iVoigt) = 0.0;
            aSymmetricMicroStress(iVoigt) = 0.0;
            aSkewMesoStress(iVoigt) = 0.0;
        }
    }
    
    // overloaded for inertia stresses
    template<typename KineticsScalarType>
    KOKKOS_INLINE_FUNCTION void
    initializeStressesWithZeros( 
              Plato::Array<mNumVoigtTerms, KineticsScalarType>   & aSymmetricMesoStress,
              Plato::Array<mNumVoigtTerms, KineticsScalarType>   & aSkewMesoStress,
              Plato::Array<mNumVoigtTerms, KineticsScalarType>   & aSymmetricMicroStress,
              Plato::Array<mNumVoigtTerms, KineticsScalarType>   & aSkewMicroStress) const
    {
        for( Plato::OrdinalType iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++)
        {
            aSymmetricMesoStress(iVoigt) = 0.0;
            aSymmetricMicroStress(iVoigt) = 0.0;
            aSkewMesoStress(iVoigt) = 0.0;
            aSkewMicroStress(iVoigt) = 0.0;
        }
    }

    template<typename StressScalarType, typename StrainScalarType>
    KOKKOS_INLINE_FUNCTION void
    addSymmetricMesoStressTerm( 
              Plato::Array<mNumVoigtTerms, StressScalarType> & aStress,
        const Plato::Array<mNumVoigtTerms, StrainScalarType> & aStrain,
        const Plato::Scalar aScale = 1.0) const
    {
        for( Plato::OrdinalType iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++)
        {
            for( Plato::OrdinalType jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
                aStress(iVoigt) += aScale*aStrain(jVoigt)*mCellMesoStressSymmetricMaterialTensor(iVoigt, jVoigt);
            }
        }
    }

    template<typename StressScalarType, typename StrainScalarType>
    KOKKOS_INLINE_FUNCTION void
    addSkewMesoStressTerm( 
              Plato::Array<mNumVoigtTerms, StressScalarType> & aStress,
        const Plato::Array<mNumSkwTerms, StrainScalarType>   & aStrain,
        const Plato::Scalar aScale = 1.0) const
    {
        for( Plato::OrdinalType iSkew=0; iSkew<mNumSkwTerms; iSkew++)
        {
            Plato::OrdinalType StressOrdinalI = mNumSpatialDims + iSkew;
            for( Plato::OrdinalType jSkew=0; jSkew<mNumSkwTerms; jSkew++){
                aStress(StressOrdinalI) += aScale*aStrain(jSkew)*mCellMesoStressSkewMaterialTensor(iSkew, jSkew);
            }
        }
    }

    template<typename StressScalarType, typename StrainScalarType>
    KOKKOS_INLINE_FUNCTION void
    addSymmetricMicroStressTerm( 
              Plato::Array<mNumVoigtTerms, StressScalarType> & aStress,
        const Plato::Array<mNumVoigtTerms, StrainScalarType> & aStrain) const
    {
        for( Plato::OrdinalType iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++)
        {
            for( Plato::OrdinalType jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
                aStress(iVoigt) += aStrain(jVoigt)*mCellMicroStressSymmetricMaterialTensor(iVoigt, jVoigt);
            }
        }
    }

    template<typename StressScalarType, typename StrainScalarType>
    KOKKOS_INLINE_FUNCTION void
    addSkewMicroStressTerm( 
              Plato::Array<mNumVoigtTerms, StressScalarType> & aStress,
        const Plato::Array<mNumSkwTerms, StrainScalarType>   & aStrain) const
    {
        for( Plato::OrdinalType iSkew=0; iSkew<mNumSkwTerms; iSkew++)
        {
            Plato::OrdinalType StressOrdinalI = mNumSpatialDims + iSkew;
            for( Plato::OrdinalType jSkew=0; jSkew<mNumSkwTerms; jSkew++){
                aStress(StressOrdinalI) += aStrain(jSkew)*mCellMicroStressSkewMaterialTensor(iSkew, jSkew);
            }
        }
    }

};

}

}

