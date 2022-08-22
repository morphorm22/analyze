#pragma once

#include "SimplexMicromorphicMechanics.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Functor for projecting full stress tensor (not just symmetric) to nodes. 
 *
 * \tparam SpaceDim       spatial dimensions
 * \tparam DofOffset      offset apply to degree of freedom indexing
*******************************************************************************/
template<Plato::OrdinalType SpaceDim,
         Plato::OrdinalType DofOffset = 0>
class ProjectStressToNode : public Plato::SimplexMicromorphicMechanics<SpaceDim>
{
private:
    using Plato::SimplexMicromorphicMechanics<SpaceDim>::mNumNodesPerCell; 
    using Plato::SimplexMicromorphicMechanics<SpaceDim>::mNumDofsPerNode;  
    using Plato::SimplexMicromorphicMechanics<SpaceDim>::mNumVoigtTerms;  
    using Plato::SimplexMicromorphicMechanics<SpaceDim>::mNumFullTerms;  

    // 2-D Example: mVoigtMap[0] = 0, mVoigtMap[1] = 1, mVoigtMap[2] = 2, mVoigtMap[3] = 2,
    // where the stress tensor in Voigt notation is given s = {s_11, s_22, s_12} (plane strain)
    Plato::OrdinalType mVoigtMap[mNumFullTerms]; /*!< matrix with indices to stress tensor entries in Voigt notation */
    // 2-D Example: mSkewScale[0] = 0, mSkewScale[1] = 0, mSkewScale[2] = 1, mSkewScale[3] = -1,
    // where the skew stress tensor in Voigt storage is given s = {0, 0, s_12} (plane strain)
    Plato::Scalar mSkewScale[mNumFullTerms]; /*!< matrix with indices to scalr factors for skew terms with Voigt storage */

public:
    /***************************************************************************//**
     * \brief Constructor
    *******************************************************************************/
    ProjectStressToNode()
    {
        this->initializeVoigtMap();
        this->initializeSkewScale();
    }

    /***************************************************************************//**
     * \brief Project stress to node
     * overloaded for cauchy and micro stresses
     *
    *******************************************************************************/
    template<typename ForcingScalarType,
             typename StressScalarType,
             typename VolumeScalarType>
    KOKKOS_FUNCTION inline void 
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const Plato::ScalarMultiVectorT<ForcingScalarType> & aOutput,
               const Plato::ScalarMultiVectorT<StressScalarType> & aSymmetricMesoStress,
               const Plato::ScalarMultiVectorT<StressScalarType> & aSkewMesoStress,
               const Plato::ScalarMultiVectorT<StressScalarType> & aSymmetricMicroStress,
               const Plato::ScalarVectorT<Plato::Scalar> & aBasisFunctions,
               const Plato::ScalarVectorT<VolumeScalarType> & aCellVolume) const
    {
        this->addSymmetricStressAtNodes(aCellOrdinal,aOutput,aSymmetricMicroStress,aBasisFunctions,aCellVolume,1.0);
        this->addSymmetricStressAtNodes(aCellOrdinal,aOutput,aSymmetricMesoStress,aBasisFunctions,aCellVolume,-1.0);
        this->addSkewStressAtNodes(aCellOrdinal,aOutput,aSkewMesoStress,aBasisFunctions,aCellVolume,-1.0);
    }

    /***************************************************************************//**
     * \brief Project stress to node
     * overloaded for inertia stresses
     *
    *******************************************************************************/
    template<typename ForcingScalarType,
             typename StressScalarType,
             typename VolumeScalarType>
    KOKKOS_FUNCTION inline void 
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const Plato::ScalarMultiVectorT<ForcingScalarType> & aOutput,
               const Plato::ScalarMultiVectorT<StressScalarType> & aSymmetricMicroStress,
               const Plato::ScalarMultiVectorT<StressScalarType> & aSkewMicroStress,
               const Plato::ScalarVectorT<Plato::Scalar> & aBasisFunctions,
               const Plato::ScalarVectorT<VolumeScalarType> & aCellVolume) const
    {
        this->addSymmetricStressAtNodes(aCellOrdinal,aOutput,aSymmetricMicroStress,aBasisFunctions,aCellVolume,1.0);
        this->addSkewStressAtNodes(aCellOrdinal,aOutput,aSkewMicroStress,aBasisFunctions,aCellVolume,1.0);
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
        tVoigtTerm = SpaceDim;
        for(Plato::OrdinalType tDofIndex = mNumVoigtTerms; tDofIndex < mNumFullTerms; tDofIndex++)
        {
            mVoigtMap[tDofIndex] = tVoigtTerm++;
        }
    }

    inline void 
    initializeSkewScale()
    {
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < SpaceDim; tDofIndex++)
        {
            mSkewScale[tDofIndex] = 0.0;
        }
        for(Plato::OrdinalType tDofIndex = SpaceDim; tDofIndex < mNumVoigtTerms; tDofIndex++)
        {
            mSkewScale[tDofIndex] = 1.0;
        }
        for(Plato::OrdinalType tDofIndex = mNumVoigtTerms; tDofIndex < mNumFullTerms; tDofIndex++)
        {
            mSkewScale[tDofIndex] = -1.0;
        }
    }

    template<typename ForcingScalarType,
             typename StressScalarType,
             typename VolumeScalarType>
    KOKKOS_FUNCTION inline void 
    addSymmetricStressAtNodes(const Plato::OrdinalType & aCellOrdinal,
                              const Plato::ScalarMultiVectorT<ForcingScalarType> & aOutput,
                              const Plato::ScalarMultiVectorT<StressScalarType> & aStress,
                              const Plato::ScalarVectorT<Plato::Scalar> & aBasisFunctions,
                              const Plato::ScalarVectorT<VolumeScalarType> & aCellVolume,
                              const Plato::Scalar aScale) const
    {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumFullTerms; tDofIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + tDofIndex + DofOffset;
                aOutput(aCellOrdinal, tLocalOrdinal) +=
                    aScale * aCellVolume(aCellOrdinal) * aStress(aCellOrdinal, mVoigtMap[tDofIndex]) * aBasisFunctions(tNodeIndex);
            }
        }
    }

    template<typename ForcingScalarType,
             typename StressScalarType,
             typename VolumeScalarType>
    KOKKOS_FUNCTION inline void 
    addSkewStressAtNodes(const Plato::OrdinalType & aCellOrdinal,
                         const Plato::ScalarMultiVectorT<ForcingScalarType> & aOutput,
                         const Plato::ScalarMultiVectorT<StressScalarType> & aStress,
                         const Plato::ScalarVectorT<Plato::Scalar> & aBasisFunctions,
                         const Plato::ScalarVectorT<VolumeScalarType> & aCellVolume,
                         const Plato::Scalar aScale) const
    {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumFullTerms; tDofIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + tDofIndex + DofOffset;
                aOutput(aCellOrdinal, tLocalOrdinal) +=
                    aScale * mSkewScale[tDofIndex] * aCellVolume(aCellOrdinal) * aStress(aCellOrdinal, mVoigtMap[tDofIndex]) * aBasisFunctions(tNodeIndex);
            }
        }
    }

};

} // namespace Plato

