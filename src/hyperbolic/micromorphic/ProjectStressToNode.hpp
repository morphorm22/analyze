#pragma once

#include "PlatoStaticsTypes.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato
{

namespace Hyperbolic
{

/***************************************************************************//**
 * \brief Functor for projecting full stress tensor (not just symmetric) to nodes. 
*******************************************************************************/
template<typename ElementType,
         Plato::OrdinalType DofOffset = 0>
class ProjectStressToNode : ElementType
{
private:
    using ElementType::mNumSpatialDims;
    using ElementType::mNumNodesPerCell; 
    using ElementType::mNumDofsPerNode;  

    using ElementType::mNumVoigtTerms;  
    using ElementType::mNumFullTerms;  

    // 2-D Example: mVoigtMap[0] = 0, mVoigtMap[1] = 1, mVoigtMap[2] = 2, mVoigtMap[3] = 2,
    // where the stress tensor in Voigt notation is given s = {s_11, s_22, s_12} (plane strain)
    Plato::OrdinalType mVoigtMap[mNumFullTerms]; /*!< matrix with indices to stress tensor entries in Voigt notation */
    // 2-D Example: mSkewScale[0] = 0, mSkewScale[1] = 0, mSkewScale[2] = 1, mSkewScale[3] = -1,
    // where the skew stress tensor in Voigt storage is given s = {0, 0, s_12} (plane strain)
    Plato::Scalar mSkewScale[mNumFullTerms]; /*!< matrix with indices to scalr factors for skew terms with Voigt storage */

public:
    ProjectStressToNode()
    {
        this->initializeVoigtMap();
        this->initializeSkewScale();
    }

    // overloaded for cauchy and micro stresses
    template<typename ProjectedScalarType, typename StressScalarType, typename VolumeScalarType>
    DEVICE_TYPE inline void 
    operator()(
              Plato::OrdinalType                               aCellOrdinal,
        const Plato::ScalarMultiVectorT<ProjectedScalarType> & aOutput,
        const Plato::Array<mNumVoigtTerms, StressScalarType> & aSymmetricMesoStress,
        const Plato::Array<mNumVoigtTerms, StressScalarType> & aSkewMesoStress,
        const Plato::Array<mNumVoigtTerms, StressScalarType> & aSymmetricMicroStress,
        const Plato::Array<mNumNodesPerCell, Plato::Scalar>  & aBasisFunctions,
        const VolumeScalarType                               & aVolume) const
    {
        this->addSymmetricStressAtNodes(aCellOrdinal,aOutput,aSymmetricMicroStress,aBasisFunctions,aVolume,1.0);
        this->addSymmetricStressAtNodes(aCellOrdinal,aOutput,aSymmetricMesoStress,aBasisFunctions,aVolume,-1.0);
        this->addSkewStressAtNodes(aCellOrdinal,aOutput,aSkewMesoStress,aBasisFunctions,aVolume,-1.0);
    }

    // overloaded for inertia stresses
    template<typename ProjectedScalarType, typename StressScalarType, typename VolumeScalarType>
    DEVICE_TYPE inline void 
    operator()(
              Plato::OrdinalType                               aCellOrdinal,
        const Plato::ScalarMultiVectorT<ProjectedScalarType> & aOutput,
        const Plato::Array<mNumVoigtTerms, StressScalarType> & aSymmetricMicroStress,
        const Plato::Array<mNumVoigtTerms, StressScalarType> & aSkewMicroStress,
        const Plato::Array<mNumNodesPerCell, Plato::Scalar>  & aBasisFunctions,
        const VolumeScalarType                               & aVolume) const
    {
        this->addSymmetricStressAtNodes(aCellOrdinal,aOutput,aSymmetricMicroStress,aBasisFunctions,aVolume,1.0);
        this->addSkewStressAtNodes(aCellOrdinal,aOutput,aSkewMicroStress,aBasisFunctions,aVolume,1.0);
    }

private:

    inline void 
    initializeVoigtMap()
    {
        Plato::OrdinalType tVoigtTerm = 0;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumVoigtTerms; tDofIndex++)
        {
            mVoigtMap[tDofIndex] = tVoigtTerm++;
        }
        tVoigtTerm = mNumSpatialDims;
        for(Plato::OrdinalType tDofIndex = mNumVoigtTerms; tDofIndex < mNumFullTerms; tDofIndex++)
        {
            mVoigtMap[tDofIndex] = tVoigtTerm++;
        }
    }

    inline void 
    initializeSkewScale()
    {
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumSpatialDims; tDofIndex++)
        {
            mSkewScale[tDofIndex] = 0.0;
        }
        for(Plato::OrdinalType tDofIndex = mNumSpatialDims; tDofIndex < mNumVoigtTerms; tDofIndex++)
        {
            mSkewScale[tDofIndex] = 1.0;
        }
        for(Plato::OrdinalType tDofIndex = mNumVoigtTerms; tDofIndex < mNumFullTerms; tDofIndex++)
        {
            mSkewScale[tDofIndex] = -1.0;
        }
    }

    template<typename ProjectedScalarType, typename StressScalarType, typename VolumeScalarType>
    DEVICE_TYPE inline void 
    addSymmetricStressAtNodes(
              Plato::OrdinalType                               aCellOrdinal,
        const Plato::ScalarMultiVectorT<ProjectedScalarType> & aOutput,
        const Plato::Array<mNumVoigtTerms, StressScalarType> & aStress,
        const Plato::Array<mNumNodesPerCell, Plato::Scalar>  & aBasisFunctions,
        const VolumeScalarType                               & aVolume,
              Plato::Scalar                                    aScale) const
    {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumFullTerms; tDofIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + tDofIndex + DofOffset;
                ProjectedScalarType tResult = aScale * aVolume * aStress(mVoigtMap[tDofIndex]) * aBasisFunctions(tNodeIndex);
                Kokkos::atomic_add(&aOutput(aCellOrdinal, tLocalOrdinal), tResult);
            }
        }
    }

    template<typename ProjectedScalarType, typename StressScalarType, typename VolumeScalarType>
    DEVICE_TYPE inline void 
    addSkewStressAtNodes(
              Plato::OrdinalType                               aCellOrdinal,
        const Plato::ScalarMultiVectorT<ProjectedScalarType> & aOutput,
        const Plato::Array<mNumVoigtTerms, StressScalarType> & aStress,
        const Plato::Array<mNumNodesPerCell, Plato::Scalar>  & aBasisFunctions,
        const VolumeScalarType                               & aVolume,
              Plato::Scalar                                    aScale) const
    {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumFullTerms; tDofIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + tDofIndex + DofOffset;
                ProjectedScalarType tResult = aScale * mSkewScale[tDofIndex] * aVolume * aStress(mVoigtMap[tDofIndex]) * aBasisFunctions(tNodeIndex);
                Kokkos::atomic_add(&aOutput(aCellOrdinal, tLocalOrdinal), tResult);
            }
        }
    }

};

} 

} 

