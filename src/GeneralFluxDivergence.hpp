#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Flux divergence functor.

 Given a thermal flux, compute the flux divergence.
 */
/******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumDofsPerNode = 1, Plato::OrdinalType DofOffset = 0>
class GeneralFluxDivergence : public ElementType
{
private:
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;

public:
    /******************************************************************************//**
     * \brief Compute flux divergence
     * \param [in] aCellOrdinal cell (i.e. element) ordinal
     * \param [in/out] aOutput output, i.e. flux divergence
     * \param [in] aFlux input flux workset
     * \param [in] aGradient configuration gradients
     * \param [in] aStateValues 2D state values workset
     * \param [in] aScale scale parameter (default = 1.0)
    **********************************************************************************/
    template<typename ForcingScalarType, typename FluxScalarType, typename GradientScalarType, typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        Plato::OrdinalType aCellOrdinal,
        Plato::ScalarMultiVectorT<ForcingScalarType> aOutput,
        Plato::Array<mNumSpatialDims, FluxScalarType> aFlux,
        Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, GradientScalarType> aGradient,
        VolumeScalarType  aCellVolume,
        Plato::Scalar aScale = 1.0
    ) const
    {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            Plato::OrdinalType tLocalOrdinal = tNodeIndex * NumDofsPerNode + DofOffset;
            for(Plato::OrdinalType tDimIndex = 0; tDimIndex < mNumSpatialDims; tDimIndex++)
            {
                ForcingScalarType tValue = aScale * aFlux(tDimIndex) * aGradient(tNodeIndex, tDimIndex) * aCellVolume;
                Kokkos::atomic_add(&aOutput(aCellOrdinal, tLocalOrdinal), tValue);
            }
        }
    }

    /******************************************************************************//**
     * \brief Compute flux divergence
     * \param [in] aCellOrdinal cell (i.e. element) ordinal
     * \param [in/out] aOutput output, i.e. flux divergence
     * \param [in] aFlux input flux workset
     * \param [in] aGradient configuration gradients
     * \param [in] aStateValues 2D state values workset
     * \param [in] aScale scale parameter (default = 1.0)
    **********************************************************************************/
    template<typename ForcingScalarType, typename FluxScalarType, typename GradientScalarType, typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        Plato::OrdinalType aCellOrdinal,
        Plato::OrdinalType aGpOrdinal,
        Plato::ScalarMultiVectorT<ForcingScalarType> aOutput,
        Plato::ScalarArray3DT<FluxScalarType> aFlux,
        Plato::ScalarArray4DT<GradientScalarType> aGradient,
        Plato::ScalarMultiVectorT<VolumeScalarType>  aCellVolume,
        Plato::Scalar aScale = 1.0
    ) const
    {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            Plato::OrdinalType tLocalOrdinal = tNodeIndex * NumDofsPerNode + DofOffset;
            for(Plato::OrdinalType tDimIndex = 0; tDimIndex < mNumSpatialDims; tDimIndex++)
            {
                ForcingScalarType tValue = aScale * aFlux(aCellOrdinal, aGpOrdinal, tDimIndex)
                                                  * aGradient(aCellOrdinal, aGpOrdinal, tNodeIndex, tDimIndex)
                                                  * aCellVolume(aCellOrdinal, aGpOrdinal);
                Kokkos::atomic_add(&aOutput(aCellOrdinal, tLocalOrdinal), tValue);
            }
        }
    }
};
// class FluxDivergence

}
// namespace Plato
