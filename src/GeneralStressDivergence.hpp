#pragma once

namespace Plato
{

/***************************************************************************//**
 * \brief Stress Divergence functor. Given the stress tensor, apply the divergence
 *  operator to the stress tensor.
 *
 * \tparam ElementType    Base element type
 * \tparam NumDofsPerNode number of degrees of freedom per node
 * \tparam DofOffset      offset apply to degree of freedom indexing
*******************************************************************************/
template<typename ElementType,
         Plato::OrdinalType NumDofsPerNode = ElementType::mNumSpatialDims,
         Plato::OrdinalType DofOffset = 0>
class GeneralStressDivergence : public ElementType
{
private:
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;
    using ElementType::mNumVoigtTerms;

    // 2-D Example: mVoigt[0][0] = 0, mVoigt[0][1] = 2, mVoigt[1][0] = 2, mVoigt[1][1] = 1,
    // where the stress tensor in Voigt notation is given s = {s_11, s_22, s_12, s_33} (plane strain)
    Plato::OrdinalType mVoigt[mNumSpatialDims][mNumSpatialDims]; /*!< matrix with indices to stress tensor entries in Voigt notation */

public:
    /***************************************************************************//**
     * \brief Constructor
    *******************************************************************************/
    GeneralStressDivergence()
    {
        Plato::OrdinalType tVoigtTerm = 0;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumSpatialDims; tDofIndex++)
        {
            mVoigt[tDofIndex][tDofIndex] = tVoigtTerm++;
        }
        for(Plato::OrdinalType tDofIndexJ = mNumSpatialDims - 1; tDofIndexJ >= 1; tDofIndexJ--)
        {
            for(Plato::OrdinalType tDofIndexI = tDofIndexJ - 1; tDofIndexI >= 0; tDofIndexI--)
            {
                mVoigt[tDofIndexI][tDofIndexJ] = tVoigtTerm;
                mVoigt[tDofIndexJ][tDofIndexI] = tVoigtTerm++;
            }
        }
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
    KOKKOS_INLINE_FUNCTION void
    operator()(
        const Plato::OrdinalType & aCellOrdinal,
        const Plato::ScalarMultiVectorT<ForcingScalarType> & aOutput,
        const Plato::Array<mNumVoigtTerms, StressScalarType> & aStress,
        const Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, GradientScalarType> & aGradient,
        const VolumeScalarType & aCellVolume,
        const Plato::Scalar aScale = 1.0) const
    {

        for(Plato::OrdinalType tDimIndexI = 0; tDimIndexI < mNumSpatialDims; tDimIndexI++)
        {
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * NumDofsPerNode + tDimIndexI + DofOffset;
                for(Plato::OrdinalType tDimIndexJ = 0; tDimIndexJ < mNumSpatialDims; tDimIndexJ++)
                {
                    Kokkos::atomic_add(&aOutput(aCellOrdinal, tLocalOrdinal),
                        aScale * aCellVolume * aStress(mVoigt[tDimIndexI][tDimIndexJ]) * aGradient(tNodeIndex, tDimIndexJ));
                }
            }
        }
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
    KOKKOS_INLINE_FUNCTION void
    operator()(
        const Plato::OrdinalType                           & aCellOrdinal,
        const Plato::OrdinalType                           & aGpOrdinal,
        const Plato::ScalarMultiVectorT<ForcingScalarType> & aOutput,
        const Plato::ScalarArray3DT<StressScalarType>      & aStress,
        const Plato::ScalarArray4DT<GradientScalarType>    & aGradient,
        const Plato::ScalarMultiVectorT<VolumeScalarType>  & aCellVolume,
        const Plato::Scalar aScale = 1.0
    ) const
    {

        for(Plato::OrdinalType tDimIndexI = 0; tDimIndexI < mNumSpatialDims; tDimIndexI++)
        {
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * NumDofsPerNode + tDimIndexI + DofOffset;
                for(Plato::OrdinalType tDimIndexJ = 0; tDimIndexJ < mNumSpatialDims; tDimIndexJ++)
                {
                    Kokkos::atomic_add(&aOutput(aCellOrdinal, tLocalOrdinal),
                        aScale * aCellVolume(aCellOrdinal, aGpOrdinal)
                               * aStress(aCellOrdinal, aGpOrdinal, mVoigt[tDimIndexI][tDimIndexJ])
                               * aGradient(aCellOrdinal, aGpOrdinal, tNodeIndex, tDimIndexJ));
                }
            }
        }
    }
};

} // namespace Plato
