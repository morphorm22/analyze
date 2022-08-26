#pragma once

#include "PlatoStaticsTypes.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato
{

namespace Hyperbolic
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
template<typename ElementType>
class MicromorphicKinematics : public ElementType
{
private:

    using ElementType::mNumSpatialDims;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumSkwTerms;

public:

    template<typename KinematicsScalarType, typename StateScalarType, typename GradientScalarType>
    KOKKOS_INLINE_FUNCTION void operator()(
              Plato::OrdinalType                                                     aCellOrdinal,
              Plato::Array<mNumVoigtTerms, KinematicsScalarType>                   & aSymmetricGradientStrain,
              Plato::Array<mNumSkwTerms, KinematicsScalarType>                     & aSkewGradientStrain,
              Plato::Array<mNumVoigtTerms, StateScalarType>                        & aSymmetricMicroStrain,
              Plato::Array<mNumSkwTerms, StateScalarType>                          & aSkewMicroStrain,
        const Plato::ScalarMultiVectorT<StateScalarType>                           & aState,
        const Plato::Array<mNumNodesPerCell, Plato::Scalar>                        & aBasisFunctions,
        const Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, GradientScalarType> & aGradient) const
    {
        this->computeSymmetricGradientStrain(aCellOrdinal,aSymmetricGradientStrain,aState,aGradient);
        this->computeSkewGradientStrain(aCellOrdinal,aSkewGradientStrain,aState,aGradient);
        this->computeSymmetricMicroStrain(aCellOrdinal,aSymmetricMicroStrain,aState,aBasisFunctions);
        this->computeSkewMicroStrain(aCellOrdinal,aSkewMicroStrain,aState,aBasisFunctions);
    }

private:

    template<typename KinematicsScalarType, typename StateScalarType, typename GradientScalarType>
    KOKKOS_INLINE_FUNCTION void computeSymmetricGradientStrain(
              Plato::OrdinalType                                                     aCellOrdinal,
              Plato::Array<mNumVoigtTerms, KinematicsScalarType>                   & aSymmetricGradientStrain,
        const Plato::ScalarMultiVectorT<StateScalarType>                           & aState,
        const Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, GradientScalarType> & aGradient) const
    {
        Plato::OrdinalType tVoigtTerm = 0;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumSpatialDims; tDofIndex++)
        {
            aSymmetricGradientStrain(tVoigtTerm) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + tDofIndex;
                aSymmetricGradientStrain(tVoigtTerm) += aState(aCellOrdinal, tLocalOrdinal) * aGradient(tNodeIndex, tDofIndex);
            }
            tVoigtTerm++;
        }
        for(Plato::OrdinalType tDofIndexJ = mNumSpatialDims - 1; tDofIndexJ >= 1; tDofIndexJ--)
        {
            for(Plato::OrdinalType tDofIndexI = tDofIndexJ - 1; tDofIndexI >= 0; tDofIndexI--)
            {
                aSymmetricGradientStrain(tVoigtTerm) = 0.0;
                for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
                {
                    Plato::OrdinalType tLocalOrdinalI = tNodeIndex * mNumDofsPerNode + tDofIndexI;
                    Plato::OrdinalType tLocalOrdinalJ = tNodeIndex * mNumDofsPerNode + tDofIndexJ;
                    aSymmetricGradientStrain(tVoigtTerm) += (aState(aCellOrdinal, tLocalOrdinalJ)
                            * aGradient(tNodeIndex, tDofIndexI)
                            + aState(aCellOrdinal, tLocalOrdinalI) * aGradient(tNodeIndex, tDofIndexJ));
                }
                tVoigtTerm++;
            }
        }
    }

    template<typename KinematicsScalarType, typename StateScalarType, typename GradientScalarType>
    KOKKOS_INLINE_FUNCTION void computeSkewGradientStrain(
              Plato::OrdinalType                                                     aCellOrdinal,
              Plato::Array<mNumSkwTerms, KinematicsScalarType>                     & aSkewGradientStrain,
        const Plato::ScalarMultiVectorT<StateScalarType>                           & aState,
        const Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, GradientScalarType> & aGradient) const
    {
        Plato::OrdinalType tSkwTerm = 0;
        for(Plato::OrdinalType tDofIndexJ = mNumSpatialDims - 1; tDofIndexJ >= 1; tDofIndexJ--)
        {
            for(Plato::OrdinalType tDofIndexI = tDofIndexJ - 1; tDofIndexI >= 0; tDofIndexI--)
            {
                aSkewGradientStrain(tSkwTerm) = 0.0;
                for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
                {
                    Plato::OrdinalType tLocalOrdinalI = tNodeIndex * mNumDofsPerNode + tDofIndexI;
                    Plato::OrdinalType tLocalOrdinalJ = tNodeIndex * mNumDofsPerNode + tDofIndexJ;
                    aSkewGradientStrain(tSkwTerm) += (aState(aCellOrdinal, tLocalOrdinalI)
                            * aGradient(tNodeIndex, tDofIndexJ)
                            - aState(aCellOrdinal, tLocalOrdinalJ) * aGradient(tNodeIndex, tDofIndexI));
                }
                tSkwTerm++;
            }
        }
    }

    template<typename StateScalarType>
    KOKKOS_INLINE_FUNCTION void computeSymmetricMicroStrain(
              Plato::OrdinalType                                                     aCellOrdinal,
              Plato::Array<mNumVoigtTerms, StateScalarType> & aSymmetricMicroStrain,
        const Plato::ScalarMultiVectorT<StateScalarType>    & aState,
        const Plato::Array<mNumNodesPerCell, Plato::Scalar> & aBasisFunctions) const
    {
        Plato::OrdinalType tDofOffset = mNumSpatialDims;
        Plato::OrdinalType tVoigtTerm = 0;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumSpatialDims; tDofIndex++)
        {
            aSymmetricMicroStrain(tVoigtTerm) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tDofOffset + tNodeIndex * mNumDofsPerNode + tDofIndex;
                aSymmetricMicroStrain(tVoigtTerm) += aState(aCellOrdinal, tLocalOrdinal) * aBasisFunctions(tNodeIndex);
            }
            tVoigtTerm++;
        }
        Plato::OrdinalType tDofIndexTerm2 = mNumVoigtTerms;
        for(Plato::OrdinalType tDofIndexTerm1 = mNumSpatialDims; tDofIndexTerm1 < mNumVoigtTerms; tDofIndexTerm1++)
        {
            aSymmetricMicroStrain(tVoigtTerm) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinalTerm1 = tDofOffset + tNodeIndex * mNumDofsPerNode + tDofIndexTerm1; 
                Plato::OrdinalType tLocalOrdinalTerm2 = tDofOffset + tNodeIndex * mNumDofsPerNode + tDofIndexTerm2;
                aSymmetricMicroStrain(tVoigtTerm) += (aState(aCellOrdinal, tLocalOrdinalTerm1)
                        * aBasisFunctions(tNodeIndex)
                        + aState(aCellOrdinal, tLocalOrdinalTerm2) * aBasisFunctions(tNodeIndex));
            }
            tVoigtTerm++;
            tDofIndexTerm2++;
        }
    }

    template<typename StateScalarType>
    KOKKOS_INLINE_FUNCTION void computeSkewMicroStrain(
              Plato::OrdinalType                                                     aCellOrdinal,
              Plato::Array<mNumSkwTerms, StateScalarType>   & aSkewMicroStrain,
        const Plato::ScalarMultiVectorT<StateScalarType>    & aState,
        const Plato::Array<mNumNodesPerCell, Plato::Scalar> & aBasisFunctions) const
    {
        Plato::OrdinalType tDofOffset = mNumSpatialDims;
        Plato::OrdinalType tSkwTerm = 0;
        Plato::OrdinalType tDofIndexTerm2 = mNumVoigtTerms;
        for(Plato::OrdinalType tDofIndexTerm1 = mNumSpatialDims; tDofIndexTerm1 < mNumVoigtTerms; tDofIndexTerm1++)
        {
            aSkewMicroStrain(tSkwTerm) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinalTerm1 = tDofOffset + tNodeIndex * mNumDofsPerNode + tDofIndexTerm1; 
                Plato::OrdinalType tLocalOrdinalTerm2 = tDofOffset + tNodeIndex * mNumDofsPerNode + tDofIndexTerm2;
                aSkewMicroStrain(tSkwTerm) += (aState(aCellOrdinal, tLocalOrdinalTerm1)
                        * aBasisFunctions(tNodeIndex)
                        - aState(aCellOrdinal, tLocalOrdinalTerm2) * aBasisFunctions(tNodeIndex));
            }
            tSkwTerm++;
            tDofIndexTerm2++;
        }
    }

};

}

}
