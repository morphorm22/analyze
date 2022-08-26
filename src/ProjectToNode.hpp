#pragma once

namespace Plato
{

/******************************************************************************/
/*! Project to node functor.

 Given values at gauss points, multiply by the basis functions to project
 to the nodes.
 */
/******************************************************************************/
template< typename           ElementType,
          Plato::OrdinalType NumArgDofs=ElementType::mNumDofsPerNode,
          Plato::OrdinalType DofOffset=0>
class ProjectToNode : public ElementType
{
private:
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;

public:
    /******************************************************************************//**
     * \brief Project state node values to cubature points (i.e. Gauss points)
     * \param [in] aCellOrdinal cell (i.e. element) ordinal
     * \param [in] aVolume Gauss point volume
     * \param [in] aBasisValues basis function values
     * \param [in] aStateValues Array of values to be projected
     * \param [in/out] aResult output, state values at cubature points
     * \param [in] aScale scale parameter (default = 1.0)
    **********************************************************************************/
    template<typename GaussPointScalarType, typename ProjectedScalarType, typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
      const Plato::OrdinalType                             & aCellOrdinal,
      const VolumeScalarType                               & aVolume,
      const Plato::Array<mNumNodesPerCell>                 & aBasisValues,
      const Plato::Array<NumArgDofs, GaussPointScalarType> & aStateValues,
      const Plato::ScalarMultiVectorT<ProjectedScalarType> & aResult,
            Plato::Scalar aScale = 1.0
    ) const
    {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumArgDofs; tDofIndex++)
            {
                Plato::OrdinalType tMyDofIndex = (mNumDofsPerNode * tNodeIndex) + tDofIndex + DofOffset;
                ProjectedScalarType tResult = aScale * aBasisValues(tNodeIndex) * aStateValues(tDofIndex) * aVolume;
                Kokkos::atomic_add(&aResult(aCellOrdinal, tMyDofIndex), tResult);
            }
        }
    }

    /******************************************************************************//**
     * \brief Project state node values to cubature points (i.e. Gauss points)
     * \param [in] aCellOrdinal cell (i.e. element) ordinal
     * \param [in] aVolume Gauss point volume
     * \param [in] aBasisValues basis function values
     * \param [in] aStateValues 1D state values workset
     * \param [in/out] aResult output, state values at cubature points
     * \param [in] aScale scale parameter (default = 1.0)
    **********************************************************************************/
    template<typename GaussPointScalarType, typename ProjectedScalarType, typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
      const Plato::OrdinalType                             & aCellOrdinal,
      const VolumeScalarType                               & aVolume,
      const Plato::Array<mNumNodesPerCell>                 & aBasisValues,
      const GaussPointScalarType                           & aStateValue,
      const Plato::ScalarMultiVectorT<ProjectedScalarType> & aResult,
            Plato::Scalar aScale = 1.0
    ) const
    {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            Plato::OrdinalType tMyDofIndex = (mNumDofsPerNode * tNodeIndex) + DofOffset;
            ProjectedScalarType tResult = aScale * aBasisValues(tNodeIndex) * aStateValue * aVolume;
            Kokkos::atomic_add(&aResult(aCellOrdinal, tMyDofIndex), tResult);
        }
    }
};
// class ProjectToNode

}
// namespace Plato
