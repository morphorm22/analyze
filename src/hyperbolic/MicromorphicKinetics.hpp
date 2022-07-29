#pragma once

#include "PlatoMathTypes.hpp"
#include "hyperbolic/SimplexMicromorphicMechanics.hpp"

namespace Plato
{

/******************************************************************************/
/*! micromorphic kinetics functor.

    Given a set of micromorphic strain measures, compute the stress measures
*/
/******************************************************************************/
template<int SpaceDim>
class MicromorphicKinetics : public Plato::SimplexMicromorphicMechanics<SpaceDim>
{
  private:

    using Plato::SimplexMicromorphicMechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexMicromorphicMechanics<SpaceDim>::mNumSkwTerms;

    const Plato::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellMesoStressSymmetricMaterialTensor; 
    const Plato::Matrix<mNumSkwTerms,mNumSkwTerms>     mCellMesoStressSkewMaterialTensor; 
    const Plato::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellMicroStressSymmetricMaterialTensor; 
    const Plato::Matrix<mNumSkwTerms,mNumSkwTerms>     mCellMicroStressSkewMaterialTensor; 

  public:

    // overloaded for cauchy and micro stresses
    MicromorphicKinetics(const Teuchos::RCP<Plato::MicromorphicLinearElasticMaterial<SpaceDim>> aMaterialModel) :
      mCellMesoStressSymmetricMaterialTensor(aMaterialModel->getStiffnessMatrixCe()),
      mCellMesoStressSkewMaterialTensor(aMaterialModel->getStiffnessMatrixCc()),
      mCellMicroStressSymmetricMaterialTensor(aMaterialModel->getStiffnessMatrixCm())
    {
    }
    
    // overloaded for inertia stresses
    MicromorphicKinetics(const Teuchos::RCP<Plato::MicromorphicInertiaMaterial<SpaceDim>> aMaterialModel) :
      mCellMesoStressSymmetricMaterialTensor(aMaterialModel->getInertiaMatrixTe()),
      mCellMesoStressSkewMaterialTensor(aMaterialModel->getInertiaMatrixTc()),
      mCellMicroStressSymmetricMaterialTensor(aMaterialModel->getInertiaMatrixJm()),
      mCellMicroStressSkewMaterialTensor(aMaterialModel->getInertiaMatrixJc())
    {
    }

    // overloaded for cauchy and micro stresses
    template<typename KineticsScalarType, typename KinematicsScalarType, typename StateScalarType>
    KOKKOS_FUNCTION inline void
    operator()( Plato::OrdinalType aCellOrdinal,
                Plato::ScalarMultiVectorT<KineticsScalarType> const& aSymmetricMesoStress,
                Plato::ScalarMultiVectorT<KineticsScalarType> const& aSkewMesoStress,
                Plato::ScalarMultiVectorT<KineticsScalarType> const& aSymmetricMicroStress,
                Plato::ScalarMultiVectorT<KinematicsScalarType> const& aSymmetricGradientStrain,
                Plato::ScalarMultiVectorT<KinematicsScalarType> const& aSkewGradientStrain,
                Plato::ScalarMultiVectorT<StateScalarType> const& aSymmetricMicroStrain,
                Plato::ScalarMultiVectorT<StateScalarType> const& aSkewMicroStrain) const
    {
        this->initializeStressesWithZeros(aCellOrdinal, aSymmetricMesoStress,aSkewMesoStress,aSymmetricMicroStress);
        this->addSymmetricMesoStressTerm(aCellOrdinal,aSymmetricMesoStress,aSymmetricGradientStrain,1.0);
        this->addSymmetricMesoStressTerm(aCellOrdinal,aSymmetricMesoStress,aSymmetricMicroStrain,-1.0);
        this->addSkewMesoStressTerm(aCellOrdinal,aSkewMesoStress,aSkewGradientStrain,1.0);
        this->addSkewMesoStressTerm(aCellOrdinal,aSkewMesoStress,aSkewMicroStrain,-1.0);
        this->addSymmetricMicroStressTerm(aCellOrdinal,aSymmetricMicroStress,aSymmetricMicroStrain);
    }
    
    // overloaded for inertia stresses
    template<typename KineticsScalarType, typename KinematicsScalarType, typename StateScalarType>
    KOKKOS_FUNCTION inline void
    operator()( Plato::OrdinalType aCellOrdinal,
                Plato::ScalarMultiVectorT<KineticsScalarType> const& aSymmetricMesoStress,
                Plato::ScalarMultiVectorT<KineticsScalarType> const& aSkewMesoStress,
                Plato::ScalarMultiVectorT<KineticsScalarType> const& aSymmetricMicroStress,
                Plato::ScalarMultiVectorT<KineticsScalarType> const& aSkewMicroStress,
                Plato::ScalarMultiVectorT<KinematicsScalarType> const& aSymmetricGradientStrain,
                Plato::ScalarMultiVectorT<KinematicsScalarType> const& aSkewGradientStrain,
                Plato::ScalarMultiVectorT<StateScalarType> const& aSymmetricMicroStrain,
                Plato::ScalarMultiVectorT<StateScalarType> const& aSkewMicroStrain) const
    {
        this->initializeStressesWithZeros(aCellOrdinal, aSymmetricMesoStress,aSkewMesoStress,aSymmetricMicroStress);
        this->addSymmetricMesoStressTerm(aCellOrdinal,aSymmetricMesoStress,aSymmetricGradientStrain,1.0);
        this->addSkewMesoStressTerm(aCellOrdinal,aSkewMesoStress,aSkewGradientStrain,1.0);
        this->addSymmetricMicroStressTerm(aCellOrdinal,aSymmetricMicroStress,aSymmetricMicroStrain);
        this->addSkewMicroStressTerm(aCellOrdinal,aSkewMicroStress,aSkewMicroStrain);
    }

  private:

    // overloaded for cauchy and micro stresses
    template<typename KineticsScalarType>
    KOKKOS_FUNCTION inline void
    initializeStressesWithZeros( Plato::OrdinalType aCellOrdinal,
                                 Plato::ScalarMultiVectorT<KineticsScalarType> const& aSymmetricMesoStress,
                                 Plato::ScalarMultiVectorT<KineticsScalarType> const& aSkewMesoStress,
                                 Plato::ScalarMultiVectorT<KineticsScalarType> const& aSymmetricMicroStress) const
    {
        for( Plato::OrdinalType iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++)
        {
            aSymmetricMesoStress(aCellOrdinal,iVoigt) = 0.0;
            aSymmetricMicroStress(aCellOrdinal,iVoigt) = 0.0;
            aSkewMesoStress(aCellOrdinal,iVoigt) = 0.0;
        }
    }
    
    // overloaded for inertia stresses
    template<typename KineticsScalarType>
    KOKKOS_FUNCTION inline void
    initializeStressesWithZeros( Plato::OrdinalType aCellOrdinal,
                                 Plato::ScalarMultiVectorT<KineticsScalarType> const& aSymmetricMesoStress,
                                 Plato::ScalarMultiVectorT<KineticsScalarType> const& aSkewMesoStress,
                                 Plato::ScalarMultiVectorT<KineticsScalarType> const& aSymmetricMicroStress,
                                 Plato::ScalarMultiVectorT<KineticsScalarType> const& aSkewMicroStress) const
    {
        for( Plato::OrdinalType iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++)
        {
            aSymmetricMesoStress(aCellOrdinal,iVoigt) = 0.0;
            aSymmetricMicroStress(aCellOrdinal,iVoigt) = 0.0;
            aSkewMesoStress(aCellOrdinal,iVoigt) = 0.0;
            aSkewMicroStress(aCellOrdinal,iVoigt) = 0.0;
        }
    }

    template<typename StressScalarType, typename StrainScalarType>
    KOKKOS_FUNCTION inline void
    addSymmetricMesoStressTerm( Plato::OrdinalType aCellOrdinal,
                                  Plato::ScalarMultiVectorT<StressScalarType> const& aStress,
                                  Plato::ScalarMultiVectorT<StrainScalarType> const& aStrain,
                                  const Plato::Scalar aScale = 1.0) const
    {
        for( Plato::OrdinalType iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++)
        {
            for( Plato::OrdinalType jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
                aStress(aCellOrdinal,iVoigt) += aScale*aStrain(aCellOrdinal,jVoigt)*mCellMesoStressSymmetricMaterialTensor(iVoigt, jVoigt);
            }
        }
    }

    template<typename StressScalarType, typename StrainScalarType>
    KOKKOS_FUNCTION inline void
    addSkewMesoStressTerm( Plato::OrdinalType aCellOrdinal,
                             Plato::ScalarMultiVectorT<StressScalarType> const& aStress,
                             Plato::ScalarMultiVectorT<StrainScalarType> const& aStrain,
                             const Plato::Scalar aScale = 1.0) const
    {
        for( Plato::OrdinalType iSkew=0; iSkew<mNumSkwTerms; iSkew++)
        {
            Plato::OrdinalType StressOrdinalI = SpaceDim + iSkew;
            for( Plato::OrdinalType jSkew=0; jSkew<mNumSkwTerms; jSkew++){
                aStress(aCellOrdinal,StressOrdinalI) += aScale*aStrain(aCellOrdinal,jSkew)*mCellMesoStressSkewMaterialTensor(iSkew, jSkew);
            }
        }
    }

    template<typename StressScalarType, typename StrainScalarType>
    KOKKOS_FUNCTION inline void
    addSymmetricMicroStressTerm( Plato::OrdinalType aCellOrdinal,
                                                Plato::ScalarMultiVectorT<StressScalarType> const& aStress,
                                                Plato::ScalarMultiVectorT<StrainScalarType> const& aStrain) const
    {
        for( Plato::OrdinalType iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++)
        {
            for( Plato::OrdinalType jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
                aStress(aCellOrdinal,iVoigt) += aStrain(aCellOrdinal,jVoigt)*mCellMicroStressSymmetricMaterialTensor(iVoigt, jVoigt);
            }
        }
    }

    template<typename StressScalarType, typename StrainScalarType>
    KOKKOS_FUNCTION inline void
    addSkewMicroStressTerm( Plato::OrdinalType aCellOrdinal,
                                                Plato::ScalarMultiVectorT<StressScalarType> const& aStress,
                                                Plato::ScalarMultiVectorT<StrainScalarType> const& aStrain) const
    {
        for( Plato::OrdinalType iSkew=0; iSkew<mNumSkwTerms; iSkew++)
        {
            Plato::OrdinalType StressOrdinalI = SpaceDim + iSkew;
            for( Plato::OrdinalType jSkew=0; jSkew<mNumSkwTerms; jSkew++){
                aStress(aCellOrdinal,StressOrdinalI) += aStrain(aCellOrdinal,jSkew)*mCellMicroStressSkewMaterialTensor(iSkew, jSkew);
            }
        }
    }

};
// class MicromorphicKinetics

} // namespace Plato

