#pragma once

#include "hyperbolic/micromorphic/SimplexMicromorphicMechanics.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Micromorphic kinematics functor.

 Given a shape function gradient matrix, shape function array,
 displacement array, and micro-distortion array, compute the relevant strains.

 Micromorphic state DOFs stored as (for e.g. 3D):
  [u1, u2, u3, X11, X22, X33, X23, X13, X12, X32, X31, X21] 
  u is displacement vector
  X is micro distortion tensor
 */
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class MicromorphicKinematics : public Plato::SimplexMicromorphicMechanics<SpaceDim>
{
private:

    using Plato::SimplexMicromorphicMechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexMicromorphicMechanics<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexMicromorphicMechanics<SpaceDim>::mNumDofsPerNode;

public:

    template<typename KinematicsScalarType, typename StateScalarType, typename GradientScalarType>
    DEVICE_TYPE inline void operator()(Plato::OrdinalType aCellOrdinal,
                                       Plato::ScalarMultiVectorT<KinematicsScalarType> const& aSymmetricGradientStrain,
                                       Plato::ScalarMultiVectorT<KinematicsScalarType> const& aSkewGradientStrain,
                                       Plato::ScalarMultiVectorT<StateScalarType> const& aSymmetricMicroStrain,
                                       Plato::ScalarMultiVectorT<StateScalarType> const& aSkewMicroStrain,
                                       Plato::ScalarMultiVectorT<StateScalarType> const& aState,
                                       Plato::ScalarVectorT<Plato::Scalar> const& aBasisFunctions,
                                       Plato::ScalarArray3DT<GradientScalarType> const& aGradient) const
    {
        this->computeSymmetricGradientStrain(aCellOrdinal,aSymmetricGradientStrain,aState,aGradient);
        this->computeSkewGradientStrain(aCellOrdinal,aSkewGradientStrain,aState,aGradient);
        this->computeSymmetricMicroStrain(aCellOrdinal,aSymmetricMicroStrain,aState,aBasisFunctions);
        this->computeSkewMicroStrain(aCellOrdinal,aSkewMicroStrain,aState,aBasisFunctions);
    }

private:

    template<typename KinematicsScalarType, typename StateScalarType, typename GradientScalarType>
    DEVICE_TYPE inline void computeSymmetricGradientStrain(Plato::OrdinalType aCellOrdinal,
                                       Plato::ScalarMultiVectorT<KinematicsScalarType> const& aSymmetricGradientStrain,
                                       Plato::ScalarMultiVectorT<StateScalarType> const& aState,
                                       Plato::ScalarArray3DT<GradientScalarType> const& aGradient) const
    {
        Plato::OrdinalType tVoigtTerm = 0;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < SpaceDim; tDofIndex++)
        {
            aSymmetricGradientStrain(aCellOrdinal, tVoigtTerm) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + tDofIndex;
                aSymmetricGradientStrain(aCellOrdinal, tVoigtTerm) += aState(aCellOrdinal, tLocalOrdinal) * aGradient(aCellOrdinal, tNodeIndex, tDofIndex);
            }
            tVoigtTerm++;
        }
        for(Plato::OrdinalType tDofIndexJ = SpaceDim - 1; tDofIndexJ >= 1; tDofIndexJ--)
        {
            for(Plato::OrdinalType tDofIndexI = tDofIndexJ - 1; tDofIndexI >= 0; tDofIndexI--)
            {
                aSymmetricGradientStrain(aCellOrdinal, tVoigtTerm) = 0.0;
                for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
                {
                    Plato::OrdinalType tLocalOrdinalI = tNodeIndex * mNumDofsPerNode + tDofIndexI;
                    Plato::OrdinalType tLocalOrdinalJ = tNodeIndex * mNumDofsPerNode + tDofIndexJ;
                    aSymmetricGradientStrain(aCellOrdinal, tVoigtTerm) += (aState(aCellOrdinal, tLocalOrdinalJ)
                            * aGradient(aCellOrdinal, tNodeIndex, tDofIndexI)
                            + aState(aCellOrdinal, tLocalOrdinalI) * aGradient(aCellOrdinal, tNodeIndex, tDofIndexJ));
                }
                tVoigtTerm++;
            }
        }
    }

    template<typename KinematicsScalarType, typename StateScalarType, typename GradientScalarType>
    DEVICE_TYPE inline void computeSkewGradientStrain(Plato::OrdinalType aCellOrdinal,
                                       Plato::ScalarMultiVectorT<KinematicsScalarType> const& aSkewGradientStrain,
                                       Plato::ScalarMultiVectorT<StateScalarType> const& aState,
                                       Plato::ScalarArray3DT<GradientScalarType> const& aGradient) const
    {
        Plato::OrdinalType tSkwTerm = 0;
        for(Plato::OrdinalType tDofIndexJ = SpaceDim - 1; tDofIndexJ >= 1; tDofIndexJ--)
        {
            for(Plato::OrdinalType tDofIndexI = tDofIndexJ - 1; tDofIndexI >= 0; tDofIndexI--)
            {
                aSkewGradientStrain(aCellOrdinal, tSkwTerm) = 0.0;
                for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
                {
                    Plato::OrdinalType tLocalOrdinalI = tNodeIndex * mNumDofsPerNode + tDofIndexI;
                    Plato::OrdinalType tLocalOrdinalJ = tNodeIndex * mNumDofsPerNode + tDofIndexJ;
                    aSkewGradientStrain(aCellOrdinal, tSkwTerm) += (aState(aCellOrdinal, tLocalOrdinalI)
                            * aGradient(aCellOrdinal, tNodeIndex, tDofIndexJ)
                            - aState(aCellOrdinal, tLocalOrdinalJ) * aGradient(aCellOrdinal, tNodeIndex, tDofIndexI));
                }
                tSkwTerm++;
            }
        }
    }

    template<typename StateScalarType>
    DEVICE_TYPE inline void computeSymmetricMicroStrain(Plato::OrdinalType aCellOrdinal,
                                       Plato::ScalarMultiVectorT<StateScalarType> const& aSymmetricMicroStrain,
                                       Plato::ScalarMultiVectorT<StateScalarType> const& aState,
                                       Plato::ScalarVectorT<Plato::Scalar> const& aBasisFunctions) const
    {
        Plato::OrdinalType tDofOffset = SpaceDim;
        Plato::OrdinalType tVoigtTerm = 0;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < SpaceDim; tDofIndex++)
        {
            aSymmetricMicroStrain(aCellOrdinal, tVoigtTerm) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tDofOffset + tNodeIndex * mNumDofsPerNode + tDofIndex;
                aSymmetricMicroStrain(aCellOrdinal, tVoigtTerm) += aState(aCellOrdinal, tLocalOrdinal) * aBasisFunctions(tNodeIndex);
            }
            tVoigtTerm++;
        }
        Plato::OrdinalType tDofIndexTerm2 = mNumVoigtTerms;
        for(Plato::OrdinalType tDofIndexTerm1 = SpaceDim; tDofIndexTerm1 < mNumVoigtTerms; tDofIndexTerm1++)
        {
            aSymmetricMicroStrain(aCellOrdinal, tVoigtTerm) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinalTerm1 = tDofOffset + tNodeIndex * mNumDofsPerNode + tDofIndexTerm1; 
                Plato::OrdinalType tLocalOrdinalTerm2 = tDofOffset + tNodeIndex * mNumDofsPerNode + tDofIndexTerm2;
                aSymmetricMicroStrain(aCellOrdinal, tVoigtTerm) += (aState(aCellOrdinal, tLocalOrdinalTerm1)
                        * aBasisFunctions(tNodeIndex)
                        + aState(aCellOrdinal, tLocalOrdinalTerm2) * aBasisFunctions(tNodeIndex));
            }
            tVoigtTerm++;
            tDofIndexTerm2++;
        }
    }

    template<typename StateScalarType>
    DEVICE_TYPE inline void computeSkewMicroStrain(Plato::OrdinalType aCellOrdinal,
                                       Plato::ScalarMultiVectorT<StateScalarType> const& aSkewMicroStrain,
                                       Plato::ScalarMultiVectorT<StateScalarType> const& aState,
                                       Plato::ScalarVectorT<Plato::Scalar> const& aBasisFunctions) const
    {
        Plato::OrdinalType tDofOffset = SpaceDim;
        Plato::OrdinalType tSkwTerm = 0;
        Plato::OrdinalType tDofIndexTerm2 = mNumVoigtTerms;
        for(Plato::OrdinalType tDofIndexTerm1 = SpaceDim; tDofIndexTerm1 < mNumVoigtTerms; tDofIndexTerm1++)
        {
            aSkewMicroStrain(aCellOrdinal, tSkwTerm) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinalTerm1 = tDofOffset + tNodeIndex * mNumDofsPerNode + tDofIndexTerm1; 
                Plato::OrdinalType tLocalOrdinalTerm2 = tDofOffset + tNodeIndex * mNumDofsPerNode + tDofIndexTerm2;
                aSkewMicroStrain(aCellOrdinal, tSkwTerm) += (aState(aCellOrdinal, tLocalOrdinalTerm1)
                        * aBasisFunctions(tNodeIndex)
                        - aState(aCellOrdinal, tLocalOrdinalTerm2) * aBasisFunctions(tNodeIndex));
            }
            tSkwTerm++;
            tDofIndexTerm2++;
        }
    }

};
// class MicromorphicKinematics

} // namespace Plato

