#pragma once

#include "hyperbolic/micromorphic/SimplexMicromorphicMechanics.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Stress Divergence functor for full stress tensor (not just symmetric). 
 *  Given the symmetric and skew parts of a stress tensor, apply the divergence
 *  operator to the sum of the symmetric and skew parts.
 *
 * \tparam SpaceDim       spatial dimensions
 * \tparam DofOffset      offset apply to degree of freedom indexing
*******************************************************************************/
template<Plato::OrdinalType SpaceDim,
         Plato::OrdinalType DofOffset = 0>
class FullStressDivergence : public Plato::SimplexMicromorphicMechanics<SpaceDim>
{
private:
    using Plato::SimplexMicromorphicMechanics<SpaceDim>::mNumNodesPerCell; 
    using Plato::SimplexMicromorphicMechanics<SpaceDim>::mNumDofsPerNode;  

    // 2-D Example: mVoigtMap[0][0] = 0, mVoigtMap[0][1] = 2, mVoigtMap[1][0] = 2, mVoigtMap[1][1] = 1,
    // where the stress tensor in Voigt notation is given s = {s_11, s_22, s_12} (plane strain)
    Plato::OrdinalType mVoigtMap[SpaceDim][SpaceDim]; /*!< matrix with indices to stress tensor entries in Voigt notation */
    // 2-D Example: mSkewScale[0][0] = 0, mSkewScale[0][1] = 1, mSkewScale[1][0] = -1, mSkewScale[1][1] = 0,
    // where the skew stress tensor in Voigt storage is given s = {0, 0, s_12} (plane strain)
    Plato::Scalar mSkewScale[SpaceDim][SpaceDim]; /*!< matrix with indices to scalr factors for skew terms with Voigt storage */

public:
    /***************************************************************************//**
     * \brief Constructor
    *******************************************************************************/
    FullStressDivergence()
    {
        this->initializeVoigtMap();
        this->initializeSkewScale();
    }

    /***************************************************************************//**
     * \brief Apply stress divergence operator to stress tensor
     *
     * \tparam ForcingScalarType   Kokkos::View POD type
     * \tparam StressScalarType    Kokkos::View POD type
     * \tparam GradientScalarType  Kokkos::View POD type
     * \tparam VolumeScalarType    Kokkos::View POD type
     *
     * \param aCellOrdinal cell index
     * \param aOutput      stress divergence
     * \param aStress      stress tensor
     * \param aGradient    spatial gradient tensor
     * \param aCellVolume  cell volume
     * \param aScale       multiplier (default = 1.0)
     *
    *******************************************************************************/
    template<typename ForcingScalarType,
             typename StressScalarType,
             typename GradientScalarType,
             typename VolumeScalarType>
    DEVICE_TYPE inline void 
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const Plato::ScalarMultiVectorT<ForcingScalarType> & aOutput,
               const Plato::ScalarMultiVectorT<StressScalarType> & aSymmetricStress,
               const Plato::ScalarMultiVectorT<StressScalarType> & aSkewStress,
               const Plato::ScalarArray3DT<GradientScalarType> & aGradient,
               const Plato::ScalarVectorT<VolumeScalarType> & aCellVolume,
               const Plato::Scalar aScale = 1.0) const
    {
        this->addSymmetricStressDivergence(aCellOrdinal,aOutput,aSymmetricStress,aGradient,aCellVolume,aScale);
        this->addSkewStressDivergence(aCellOrdinal,aOutput,aSkewStress,aGradient,aCellVolume,aScale);
    }

private:

    inline void 
    initializeVoigtMap()
    {
        Plato::OrdinalType tVoigtTerm = 0;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < SpaceDim; tDofIndex++)
        {
            mVoigtMap[tDofIndex][tDofIndex] = tVoigtTerm++;
        }
        for(Plato::OrdinalType tDofIndexJ = SpaceDim - 1; tDofIndexJ >= 1; tDofIndexJ--)
        {
            for(Plato::OrdinalType tDofIndexI = tDofIndexJ - 1; tDofIndexI >= 0; tDofIndexI--)
            {
                mVoigtMap[tDofIndexI][tDofIndexJ] = tVoigtTerm;
                mVoigtMap[tDofIndexJ][tDofIndexI] = tVoigtTerm++;
            }
        }
    }

    inline void 
    initializeSkewScale()
    {
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < SpaceDim; tDofIndex++)
        {
            mSkewScale[tDofIndex][tDofIndex] = 0.0;
        }
        for(Plato::OrdinalType tDofIndexJ = SpaceDim - 1; tDofIndexJ >= 1; tDofIndexJ--)
        {
            for(Plato::OrdinalType tDofIndexI = tDofIndexJ - 1; tDofIndexI >= 0; tDofIndexI--)
            {
                mSkewScale[tDofIndexI][tDofIndexJ] = 1.0;
                mSkewScale[tDofIndexJ][tDofIndexI] = -1.0;
            }
        }
    }

    template<typename ForcingScalarType,
             typename StressScalarType,
             typename GradientScalarType,
             typename VolumeScalarType>
    DEVICE_TYPE inline void 
    addSymmetricStressDivergence(const Plato::OrdinalType & aCellOrdinal,
                                     const Plato::ScalarMultiVectorT<ForcingScalarType> & aOutput,
                                     const Plato::ScalarMultiVectorT<StressScalarType> & aStress,
                                     const Plato::ScalarArray3DT<GradientScalarType> & aGradient,
                                     const Plato::ScalarVectorT<VolumeScalarType> & aCellVolume,
                                     const Plato::Scalar aScale) const
    {
        for(Plato::OrdinalType tDimIndexI = 0; tDimIndexI < SpaceDim; tDimIndexI++)
        {
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + tDimIndexI + DofOffset;
                for(Plato::OrdinalType tDimIndexJ = 0; tDimIndexJ < SpaceDim; tDimIndexJ++)
                {
                    aOutput(aCellOrdinal, tLocalOrdinal) +=
                        aScale * aCellVolume(aCellOrdinal) * aStress(aCellOrdinal, mVoigtMap[tDimIndexI][tDimIndexJ]) * aGradient(aCellOrdinal, tNodeIndex, tDimIndexJ);
                }
            }
        }
    }

    template<typename ForcingScalarType,
             typename StressScalarType,
             typename GradientScalarType,
             typename VolumeScalarType>
    DEVICE_TYPE inline void 
    addSkewStressDivergence(const Plato::OrdinalType & aCellOrdinal,
                                const Plato::ScalarMultiVectorT<ForcingScalarType> & aOutput,
                                const Plato::ScalarMultiVectorT<StressScalarType> & aStress,
                                const Plato::ScalarArray3DT<GradientScalarType> & aGradient,
                                const Plato::ScalarVectorT<VolumeScalarType> & aCellVolume,
                                const Plato::Scalar aScale) const
    {
        for(Plato::OrdinalType tDimIndexI = 0; tDimIndexI < SpaceDim; tDimIndexI++)
        {
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + tDimIndexI + DofOffset;
                for(Plato::OrdinalType tDimIndexJ = 0; tDimIndexJ < SpaceDim; tDimIndexJ++)
                {
                    aOutput(aCellOrdinal, tLocalOrdinal) +=
                        aScale * mSkewScale[tDimIndexI][tDimIndexJ] *aCellVolume(aCellOrdinal) * aStress(aCellOrdinal, mVoigtMap[tDimIndexI][tDimIndexJ]) * aGradient(aCellOrdinal, tNodeIndex, tDimIndexJ);
                }
            }
        }
    }

};

} // namespace Plato

