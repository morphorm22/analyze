#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Helmholtz
{

/******************************************************************************/
/*! Add mass term functor.

 Given filtered density field, compute the "mass" term.
 */
/******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumDofsPerNode = 1, Plato::OrdinalType DofOffset = 0>
class AddMassTerm
{
public:
    /******************************************************************************//**
     * \brief Add mass term
     * \param [in] aCellOrdinal cell (i.e. element) ordinal
     * \param [in/out] aOutput output, i.e. Helmholtz residual
     * \param [in] aFilteredDensity input filtered density workset
     * \param [in] aUnfilteredDensity input unfiltered density workset
     * \param [in] aBasisFunctions basis functions
     * \param [in] aCellVolume cell volume
    **********************************************************************************/
    template<typename ResultScalarType, typename StateScalarType, typename ControlScalarType, typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
      const Plato::OrdinalType                          & aCellOrdinal,
      const Plato::ScalarMultiVectorT<ResultScalarType> & aOutput,
      const StateScalarType                             & aFilteredDensity,
      const ControlScalarType                           & aUnfilteredDensity,
      const Plato::Array<ElementType::mNumNodesPerCell> & aBasisFunctions,
      const VolumeScalarType                            & aCellVolume
    ) const
    {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < ElementType::mNumNodesPerCell; tNodeIndex++)
        {
            Plato::OrdinalType tLocalOrdinal = tNodeIndex * ElementType::mNumDofsPerNode + DofOffset;
            Kokkos::atomic_add(&aOutput(aCellOrdinal, tLocalOrdinal), aBasisFunctions(tNodeIndex) * ( aFilteredDensity - aUnfilteredDensity ) * aCellVolume);
        }
    }
};
// class AddMassTerm

} // namespace Helmholtz

} // namespace Plato
