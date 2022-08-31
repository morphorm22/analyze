#pragma once

#include "PlatoStaticsTypes.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato
{

namespace Hyperbolic
{

/***************************************************************************//**
 * \brief Stress Divergence functor for full stress tensor (not just symmetric). 
 *  Given the symmetric and skew parts of a stress tensor, apply the divergence
 *  operator to the sum of the symmetric and skew parts.
*******************************************************************************/
template<typename ElementType,
         Plato::OrdinalType DofOffset = 0>
class FullStressDivergence : public ElementType
{
private:
    using ElementType::mNumSpatialDims;
    using ElementType::mNumNodesPerCell; 
    using ElementType::mNumDofsPerNode;  

    using ElementType::mNumVoigtTerms;  

    // 2-D Example: mVoigtMap[0][0] = 0, mVoigtMap[0][1] = 2, mVoigtMap[1][0] = 2, mVoigtMap[1][1] = 1,
    // where the stress tensor in Voigt notation is given s = {s_11, s_22, s_12} (plane strain)
    Plato::OrdinalType mVoigtMap[mNumSpatialDims][mNumSpatialDims]; /*!< matrix with indices to stress tensor entries in Voigt notation */
    // 2-D Example: mSkewScale[0][0] = 0, mSkewScale[0][1] = 1, mSkewScale[1][0] = -1, mSkewScale[1][1] = 0,
    // where the skew stress tensor in Voigt storage is given s = {0, 0, s_12} (plane strain)
    Plato::Scalar mSkewScale[mNumSpatialDims][mNumSpatialDims]; /*!< matrix with indices to scalr factors for skew terms with Voigt storage */

public:
    FullStressDivergence()
    {
        this->initializeVoigtMap();
        this->initializeSkewScale();
    }

    template<typename ForcingScalarType,
             typename StressScalarType,
             typename GradientScalarType,
             typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void 
    operator()(
              Plato::OrdinalType                                                     aCellOrdinal,
        const Plato::ScalarMultiVectorT<ForcingScalarType>                         & aOutput,
        const Plato::Array<mNumVoigtTerms, StressScalarType>                       & aSymmetricStress,
        const Plato::Array<mNumVoigtTerms, StressScalarType>                       & aSkewStress,
        const Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, GradientScalarType> & aGradient,
        const VolumeScalarType                                                     & aCellVolume,
              Plato::Scalar                                                          aScale = 1.0) const
    {
        this->addSymmetricStressDivergence(aCellOrdinal,aOutput,aSymmetricStress,aGradient,aCellVolume,aScale);
        this->addSkewStressDivergence(aCellOrdinal,aOutput,aSkewStress,aGradient,aCellVolume,aScale);
    }

private:

    inline void 
    initializeVoigtMap()
    {
        Plato::OrdinalType tVoigtTerm = 0;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumSpatialDims; tDofIndex++)
        {
            mVoigtMap[tDofIndex][tDofIndex] = tVoigtTerm++;
        }
        for(Plato::OrdinalType tDofIndexJ = mNumSpatialDims - 1; tDofIndexJ >= 1; tDofIndexJ--)
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
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumSpatialDims; tDofIndex++)
        {
            mSkewScale[tDofIndex][tDofIndex] = 0.0;
        }
        for(Plato::OrdinalType tDofIndexJ = mNumSpatialDims - 1; tDofIndexJ >= 1; tDofIndexJ--)
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
    KOKKOS_INLINE_FUNCTION void 
    addSymmetricStressDivergence(
              Plato::OrdinalType                                                     aCellOrdinal,
        const Plato::ScalarMultiVectorT<ForcingScalarType>                         & aOutput,
        const Plato::Array<mNumVoigtTerms, StressScalarType>                       & aStress,
        const Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, GradientScalarType> & aGradient,
        const VolumeScalarType                                                     & aCellVolume,
              Plato::Scalar                                                          aScale) const
    {
        for(Plato::OrdinalType tDimIndexI = 0; tDimIndexI < mNumSpatialDims; tDimIndexI++)
        {
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + tDimIndexI + DofOffset;
                for(Plato::OrdinalType tDimIndexJ = 0; tDimIndexJ < mNumSpatialDims; tDimIndexJ++)
                {
                    Kokkos::atomic_add(&aOutput(aCellOrdinal, tLocalOrdinal),
                        aScale * aCellVolume * aStress(mVoigtMap[tDimIndexI][tDimIndexJ]) * aGradient(tNodeIndex, tDimIndexJ));
                }
            }
        }
    }

    template<typename ForcingScalarType,
             typename StressScalarType,
             typename GradientScalarType,
             typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void 
    addSkewStressDivergence(
              Plato::OrdinalType                                                     aCellOrdinal,
        const Plato::ScalarMultiVectorT<ForcingScalarType>                         & aOutput,
        const Plato::Array<mNumVoigtTerms, StressScalarType>                       & aStress,
        const Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, GradientScalarType> & aGradient,
        const VolumeScalarType                                                     & aCellVolume,
              Plato::Scalar                                                          aScale) const
    {
        for(Plato::OrdinalType tDimIndexI = 0; tDimIndexI < mNumSpatialDims; tDimIndexI++)
        {
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + tDimIndexI + DofOffset;
                for(Plato::OrdinalType tDimIndexJ = 0; tDimIndexJ < mNumSpatialDims; tDimIndexJ++)
                {
                    Kokkos::atomic_add(&aOutput(aCellOrdinal, tLocalOrdinal),
                        aScale * mSkewScale[tDimIndexI][tDimIndexJ] * aCellVolume * aStress(mVoigtMap[tDimIndexI][tDimIndexJ]) * aGradient(tNodeIndex, tDimIndexJ));
                }
            }
        }
    }

};

}

}
